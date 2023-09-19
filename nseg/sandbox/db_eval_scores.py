import pymongo; client = pymongo.MongoClient('cajalg001')


test_erl_query = client['scores']['best_thresh_results'].find(
    {},
    {'_id': 0, 'erl.test_best_val_threshold': 1, 'setup': 1}
)
print(test_erl_query)

"""
[{'setup': '08-18_17-42_glum-cof__400k_11_micron_a',
  'erl': {'test_best_val_threshold': 13942.985581862658}},
 {'setup': '08-20_00-26_fat-micr__400k_benchmark',
  'erl': {'test_best_val_threshold': 10349.556271387299}},
 {'setup': '08-18_17-42_glum-cof__400k_benchmark',
  'erl': {'test_best_val_threshold': 3615.513098630738}},
 {'setup': '08-20_01-09_crispy-s__400k_benchmark',
  'erl': {'test_best_val_threshold': 12210.68640063825}},
 {'setup': '09-11_21-54_largo-tu__400k_benchmark',
  'erl': {'test_best_val_threshold': 14601.832575882407}},
 {'setup': '08-19_01-34_absolute__400k_benchmark_b',
  'erl': {'test_best_val_threshold': 7444.5733509282045}},
 {'setup': '08-20_00-23_plastic-__400k_benchmark_b',
  'erl': {'test_best_val_threshold': 9486.723532213868}},
 {'setup': '09-11_22-18_forgivin__400k_benchmark',
  'erl': {'test_best_val_threshold': 3517.5893924132674}},
 {'setup': '09-11_22-05_matte-bl__400k_benchmark',
  'erl': {'test_best_val_threshold': 8130.479271716295}},
 {'setup': '09-11_22-18_equidist__400k_benchmark',
  'erl': {'test_best_val_threshold': 5872.3102967503555}},
 {'setup': '09-11_22-05_savory-b__400k_benchmark',
  'erl': {'test_best_val_threshold': 9329.968481718715}},
 {'setup': '09-11_22-12_obsolete__400k_benchmark',
  'erl': {'test_best_val_threshold': 2341.510204212119}},
 {'setup': '09-11_22-05_similar-__400k_benchmark',
  'erl': {'test_best_val_threshold': 6180.167871609523}}]
"""

