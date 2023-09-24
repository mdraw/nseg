from pathlib import Path

import pymongo

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
for doc in test_score_query:
    # Get basic summary of test results
    setup_name = doc['setup']
    thresh = doc[tuning_metric]['val']  # threshold where best validation ERL score was achieved
    erl = doc['erl']['test_best_val_threshold']  # test ERL on this threshold
    voi = doc['voi']['test_best_val_threshold']  # test VOI on this threshold

    # Split and merge components of VOI are not stored in the same collection,
    #  so we need to query them separately by setup name
    mscores = get_scores_df(
        collection_name=f'{setup_name}_test',
        proj_keys=_default_proj_keys,
    )

    scores_at_thresh = mscores.loc[mscores.threshold == thresh].squeeze()
    voi_merge = scores_at_thresh.voi_merge
    voi_split = scores_at_thresh.voi_split

    setup_score_entry = SimpleNamespace(**{
        'setup_name': setup_name,
        'thresh': thresh,
        'erl': erl / 1000,  # test erl on this threshold (µm)
        'voi': voi,  # test voi on this threshold
        'voi_merge': voi_merge,
        'voi_split': voi_split,
    })

    test_scores[setup_name] = setup_score_entry

LATEX_OUT = True

if LATEX_OUT:
    # print('Threshold & ERL & VOI & VOI_merge & VOI_split\n')
    print('Method & ERL (µm) & VOI & VOI\\textsubscript{merge} & VOI_\\textsubscript{merge}\n')
for group_name, scores_db_names in group_scores.items():
    # print(f'\n== {group_name} ==\n')
    print()
    for scores_db_name in scores_db_names:
        sc = test_scores.get(scores_db_name)
        if sc is None:
            # print(f'No test scores found for {scores_db_name}')
            continue
        # print(f'{scores_db_name}')
        if LATEX_OUT:
            # print(f'{group_name} & {sc.thresh:.0f} & {sc.erl:.3f} & {sc.voi:.3f} & {sc.voi_merge:.3f} & {sc.voi_split:.3f} \\\\')
            print(f'{group_name} & {sc.erl:.3f} & {sc.voi:.3f} & {sc.voi_merge:.3f} & {sc.voi_split:.3f} \\\\ % run: {sc.setup_name} % thresh: {sc.thresh:.2f}')
        else:
            print(f'Threshold: {sc.thresh}, ERL: {sc.erl:.3f}, VOI: {sc.voi:.3f}, VOI_split: {sc.voi_merge:.3f}, VOI_merge: {sc.voi_split:.3f}\n')
