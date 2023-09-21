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
        'thresh': thresh,
        'erl': erl,  # test erl on this threshold
        'voi': voi,  # test voi on this threshold
        'voi_merge': voi_merge,
        'voi_split': voi_split,
    })

    test_scores[setup_name] = setup_score_entry

LATEX_OUT = True

if LATEX_OUT:
    print('Threshold & ERL & VOI & VOI_merge & VOI_split \\')
for group_name, scores_db_names in group_scores.items():
    print(f'\n== {group_name} ==\n')
    for scores_db_name in scores_db_names:
        sc = test_scores.get(scores_db_name)
        if sc is None:
            print(f'No test scores found for {scores_db_name}')
            continue
        print(f'{scores_db_name}')
        if LATEX_OUT:
            print(f'{sc.thresh:.0f} & {sc.erl:.0f} & {sc.voi:.3f} & {sc.voi_merge:.3f} & {sc.voi_split:.3f} \\')
        else:
            print(f'Threshold: {sc.thresh}, ERL: {sc.erl:.0f}, VOI: {sc.voi:.3f}, VOI_split: {sc.voi_merge:.3f}, VOI_merge: {sc.voi_split:.3f}\n')












# for setup_name, e in test_scores.items():
#     print(f'{setup_name}')
#     print(f'Threshold: {e.threshold} \t ERL: {e.erl:.0f} \t VOI: {e.voi:.3f}, VOI_split: {e.voi_split:.3f}, VOI_merge: {e.voi_merge:.3f}\n')



# for setup_name, e in test_scores.items():
#     print(f'{setup_name}')
#     print(f'Threshold: {e.threshold} \t ERL: {e.erl:.0f} \t VOI: {e.voi:.3f}, VOI_split: {e.voi_split:.3f}, VOI_merge: {e.voi_merge:.3f}\n')


# for doc in test_score_query:
#     setup, thresh, erl, voi = doc['setup'], doc['thresh'], doc['erl']['test_best_val_threshold'], doc['voi']['test_best_val_threshold']
#     print(f'{setup}\t VOI: {voi:.3f}, ERL: {erl:.0f}')

"""
08-20_00-26_fat-micr__400k_benchmark     VOI: 3.220, ERL: 10350
08-18_17-42_glum-cof__400k_benchmark     VOI: 7.013, ERL: 3616
08-20_01-09_crispy-s__400k_benchmark     VOI: 3.992, ERL: 12211
09-11_21-54_largo-tu__400k_benchmark     VOI: 3.461, ERL: 14602
08-19_01-34_absolute__400k_benchmark_b   VOI: 5.279, ERL: 7445
08-20_00-23_plastic-__400k_benchmark_b   VOI: 5.497, ERL: 9487
09-11_22-18_forgivin__400k_benchmark     VOI: 7.159, ERL: 3518
09-11_22-05_matte-bl__400k_benchmark     VOI: 5.530, ERL: 8130
09-11_22-18_equidist__400k_benchmark     VOI: 6.414, ERL: 5872
09-11_22-05_savory-b__400k_benchmark     VOI: 4.135, ERL: 9330
09-11_22-12_obsolete__400k_benchmark     VOI: 7.802, ERL: 2342
09-11_22-05_similar-__400k_benchmark     VOI: 6.174, ERL: 6180
09-11_22-19_obvious-__400k_benchmark     VOI: 5.192, ERL: 11133
09-11_22-12_joint-ro__400k_benchmark     VOI: 7.895, ERL: 2743
"""

"""
MTLSD:

08-20_00-26_fat-micr__400k_benchmark     VOI: 3.220, ERL: 10350

08-19_01-34_absolute__400k_benchmark_b   VOI: 5.279, ERL: 7445
"""



"""

'MTLSD + BN + HW + DT + CE':

{'setup': '09-11_22-18_forgivin__400k_benchmark', 'erl': {'test_best_val_threshold': 3517.5893924132674}}
{'setup': '09-11_22-18_equidist__400k_benchmark', 'erl': {'test_best_val_threshold': 5872.3102967503555}}
{'setup': '09-11_22-19_obvious-__400k_benchmark', 'erl': {'test_best_val_threshold': 11132.722620534822}}


'MTLSD + BN + HL + HW + DT + CE':

{'setup': '09-11_22-05_similar-__400k_benchmark', 'erl': {'test_best_val_threshold': 6180.167871609523}}
{'setup': '09-11_22-05_savory-b__400k_benchmark', 'erl': {'test_best_val_threshold': 9329.968481718715}}
{'setup': '09-11_22-05_matte-bl__400k_benchmark', 'erl': {'test_best_val_threshold': 8130.479271716295}}

"""