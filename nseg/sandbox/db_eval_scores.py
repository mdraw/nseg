"""
Create Latex table and seaborn plots from result summaries queried from db
"""

import functools

from pathlib import Path

import pymongo
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from types import SimpleNamespace

from nseg.evaluation.score_db_access import get_scores_df, _default_proj_keys
from nseg.conf import NConf


client = pymongo.MongoClient('cajalg001')

matching_setup_string = '_400k_benchmark'
tuning_metric = 'erl'

# Load eval run metadata document with manual experiment grouping, TODO: Make this configurable
nseg_dir = Path(__file__).parent.parent
nc_eval_run_groups = NConf.load(nseg_dir / 'evaluation' / 'eval_run_groups.yaml').groups
eval_run_groups = NConf.to_container(nc_eval_run_groups)

run_groups = {}
score_groups = {}  # scores_db_name -> group_name
group_scores = {}  # group_name -> list of scores_db_name
for group_name, group in eval_run_groups.items():
    group_scores[group_name] = []
    for run_name, run in group['runs'].items():
        scores_db_name = run.get('scores_db_name')
        if scores_db_name is None:
            print(f'Skipping {run_name} because it does not have a scores_db_name')
            continue
        # scores_db_name += '_test'  # Append test suffix
        run_groups[run_name] = group_name
        score_groups[scores_db_name] = group_name
        group_scores[group_name].append(scores_db_name)

test_score_query = client['scores']['best_thresh_results'].find(
    {'setup': {'$regex': matching_setup_string}},
    {
        '_id': 0,
        'erl.val': 1,
        'erl.test_best_val_threshold': 1,
        'voi.test_best_val_threshold': 1,
        'setup': 1
    }
)

test_scores = {}
test_scores_puredict = {}
for doc in test_score_query:
    # Get basic summary of test results
    setup_name = doc['setup']
    thresh = doc[tuning_metric]['val']  # threshold where best validation ERL score was achieved
    erl = doc['erl']['test_best_val_threshold']  # test ERL on this threshold
    voi = doc['voi']['test_best_val_threshold']  # test VOI on this threshold

    if setup_name not in score_groups:
        print(f'Skipping {setup_name} because it does not have a scores_db_name')
        continue

    # Split and merge components of VOI are not stored in the same collection,
    #  so we need to query them separately by setup name
    mscores = get_scores_df(
        collection_name=f'{setup_name}_test',
        proj_keys=_default_proj_keys,
    )

    scores_at_thresh = mscores.loc[mscores.threshold == thresh].squeeze()
    voi_merge = scores_at_thresh.voi_merge
    voi_split = scores_at_thresh.voi_split

    setup_score_entry = {
        'setup_name': setup_name,
        'group_name': score_groups[setup_name],
        'thresh': thresh,
        'erl': erl / 1000,  # test erl on this threshold (µm)
        'voi': voi,  # test voi on this threshold
        'voi_merge': voi_merge,
        'voi_split': voi_split,
    }

    test_scores[setup_name] = setup_score_entry

for setup_name, score_entry in test_scores.items():
    n_greater = 0
    for other_setup_name, other_score_entry in test_scores.items():
        if other_score_entry['group_name'] != score_entry['group_name'] or other_setup_name == setup_name:
            continue
        if other_score_entry['erl'] > score_entry['erl']:
            n_greater += 1
    assert n_greater < 3, f'Found {n_greater} setups with higher ERL than {setup_name}'
    ingroup_rank = {
        0: 1,
        1: 2,
        2: 3,
    }[n_greater]
    score_entry['ingroup_rank'] = ingroup_rank


# Generate latex table from results summary

LATEX_OUT = True

if LATEX_OUT:
    print('Method & ERL (µm) $\\uparrow$ & VOI $\\downarrow$ & VOI\\textsubscript{merge} $\\downarrow$ & VOI\\textsubscript{split} $\\downarrow$ \\\\\n')
for group_name, scores_db_names in sorted(group_scores.items()):
    # print(f'\n== {group_name} ==\n')
    print()
    group_score_lines = {}
    for i, scores_db_name in enumerate(scores_db_names):
        scd = test_scores.get(scores_db_name)
        if scd is None:
            print(f'No test scores found for {scores_db_name}')
            continue
        sc = SimpleNamespace(**scd)  # Enable dot notation
        ingroup_rank = sc.ingroup_rank

        # print(f'{scores_db_name}')
        if LATEX_OUT:
            score_line = f'& {sc.erl:.2f} & {sc.voi:.2f} & {sc.voi_merge:.2f} & {sc.voi_split:.2f} \\\\ % run: {sc.setup_name} % thresh: {sc.thresh:.2f}'
            # if i == 0:  # Only put group name on first line
            #     score_line = f'{group_name} {score_line}'
            # else:
            #     padding = ' ' * len(group_name)
            #     score_line = f'{padding} {score_line}'
            # score_line = f'python -m nseg.scripts.log_scores --enable-wandb {sc.setup_name} -o {group_name}_{ingroup_rank}'
            score_line = f'{group_name}\\textsubscript{{{ingroup_rank}}} {score_line}'
            # print(score_line)
        else:
            score_line = f'Threshold: {sc.thresh}, ERL: {sc.erl:.2f}, VOI: {sc.voi:.2f}, VOI_split: {sc.voi_merge:.2f}, VOI_merge: {sc.voi_split:.2f}\n'
        group_score_lines[ingroup_rank] = score_line

    for ingroup_rank in sorted(group_score_lines.keys()):
        print(group_score_lines[ingroup_rank])
    if LATEX_OUT:
        print('\\addlinespace')




### Create grouped bar plot with seaborn

plot_dir = Path('/cajal/scratch/projects/misc/mdraw/lsdex/tmp/grouped_plots')
plot_dir.mkdir(exist_ok=True)
# plot_ext = 'png'
plot_ext = 'pdf'

fmt = '%.2f'

figsize = (10, 5)

grouped_scores_df = pd.DataFrame.from_dict(test_scores, orient='index')
grouped_scores_df.to_csv('./grouped_scores.csv')


# exit(0)

# TODO: Groups are fine but we need a good way to map setup_name to hue (each repeated experiment could receive a number for example)

ddof = 1  # Delta degrees of freedom for std calculation, choose 0 for numpy default

group_erl_std = grouped_scores_df.groupby(['group_name'])['erl'].std(ddof=ddof).round(3)
group_voi_std = grouped_scores_df.groupby(['group_name'])['voi'].std(ddof=ddof).round(3)

print()
print(f'group_erl_std:\n{group_erl_std}')
print(f'group_voi_std:\n{group_voi_std}')
print()

plottable_df = grouped_scores_df.astype({'ingroup_rank': 'string'})

sorted_group_names = sorted(group_scores.keys())

# def grouped_barplot(plottable_df):

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.0})

# fig, axes = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
#
# erl_ax, voi_ax = axes



## ERL
fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

# erl_ax, voi_ax = axes

sns.barplot(
    # ax=erl_ax,
    data=plottable_df,
    x='group_name',
    order=sorted_group_names,
    y='erl',
    hue='ingroup_rank',
    hue_order=['1', '2', '3'],
)
for i in ax.containers:
    ax.bar_label(i, fmt=fmt)

# erl_ax.set_xlabel('')
# erl_ax.set_ylabel('ERL (µm)')
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('ERL (µm)')
# ax.legend_.remove()
# ax.legend_.set_title('Experiment run')
# ax.legend_.set_label(['1', '2', '2'])

plt.savefig(plot_dir / f'grouped_erl.{plot_ext}', dpi=300)
plt.close(fig)

## VOI
fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

sns.barplot(
    # ax=voi_ax,
    data=plottable_df,
    x='group_name',
    order=sorted_group_names,
    y='voi',
    hue='ingroup_rank',
    hue_order=['1', '2', '3'],
)
for i in ax.containers:
    ax.bar_label(i, fmt=fmt)

ax.set_xlabel('')
ax.set_ylabel('VOI')
ax.legend_.remove()
# ax.legend_.set_title('Experiment run')
# ax.legend_.set_label(['1', '2', '2'])


plt.savefig(plot_dir / f'grouped_voi.{plot_ext}', dpi=300)
