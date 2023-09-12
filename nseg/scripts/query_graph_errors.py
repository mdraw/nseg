"""
Experimental code for querying graph errors and fetching hardness data at the location of the error,
so that we can check if learned hardness is a good predictor of error locations.
"""


import pymongo
import numpy as np
import daisy

threshold = 0.5  # Agglomeration threshold.
voxel_size = daisy.Coordinate((20, 9, 9))
anno_name = 'test'
roi = '11_micron'
setup_name = f'08-18_17-42_glum-cof__400k_{roi}_a'
# scores_db_name = f'scores_{setup_name}_{anno_name}'  # Old storage format
scores_db_name = f'{setup_name}_{anno_name}'  # New storage format
annotations_db_name = f'{anno_name}_annotations'

dense_pred_path = '/cajal/scratch/projects/misc/mdraw/lsdex/v1/inference/08-18_17-42_glum-cof__400k_11_micron_a/affs.zarr'
hardness_dskey = 'volumes/pred_hardness'

client = pymongo.MongoClient('cajalg001')

pred_edges_coll = client[setup_name]['edges_hist_quant_75']
pred_nodes_coll = client[setup_name]['nodes']

anno_nodes_coll = client[annotations_db_name]['zebrafinch.nodes']
comp_coll = client[annotations_db_name][f'nodes_zebrafinch_components_{roi}_roi_masked']

# scores_coll = client['scores'][scores_db_name]  # Old storage format
scores_coll = client['scores'][scores_db_name]  # New storage format
split_merge_dict = scores_coll.find_one(
    {'threshold': threshold},
    projection={'_id': 0, 'merge_stats': 1, 'split_stats': 1}
)

split_stats, merge_stats = split_merge_dict['split_stats'], split_merge_dict['merge_stats']


# TODO: This is not meant for lots of calls. Could be made more efficient by looking
#  up several comp_ids at once with find() instead of find_one().
def get_node_from_comp_id(
        comp_id: int,
        comp_coll: pymongo.collection.Collection = comp_coll,
        anno_nodes_coll: pymongo.collection.Collection = anno_nodes_coll,
) -> dict[str, int]:
    """Get node gloabl node id from component ID"""
    node_id_entry = comp_coll.find_one(
        {'component_id': comp_id}, projection={'_id': 0, 'id': 1}
    )
    if node_id_entry is None:
        raise ValueError(f'Node with {comp_id=} not found in {comp_coll}')
    node_id = node_id_entry['id']
    node = anno_nodes_coll.find_one(
        {'id': node_id}, projection={'_id': 0}
    )
    assert node.keys() == {'id', 'x', 'y', 'z', 'neuron_id'}
    return node


rsplit = split_stats[-1]  # Example split error
rmerge = merge_stats[-1]  # Example merge error
print(f'{rsplit=}')
print(f'{rmerge=}')

# Split error node info




hardness_data = daisy.open_ds(dense_pred_path, hardness_dskey, mode='r')

# Example: The coordinates of component_id 183 (id 316691), which lies within the 11_micron ROI
example_coord = daisy.Coordinate((51760, 49266, 44172))
h_elem = hardness_data[example_coord]
print(hardness_data[example_coord])

# Initialize ROI as 1 voxel, centered on example_coord
rroi = daisy.Roi(example_coord, voxel_size)
# Add context by growing ROI by s voxels in each direction
rroi_grown = rroi.grow(voxel_size, voxel_size)

# Get hardness data for ROI
h_arr = hardness_data.to_ndarray(rroi_grown)
print(h_arr)


# TODO: Find a good approximation of actual error locations (not just component centroids / point annotations).
# TODO: Get hardness data at error locations.

# Idea: Collect hardness data along the connecting line between two falsely merged nodes.
#  This could be done by sampling points along the line, or by using a line integral.

# Example:

false_merge_nodes = np.array([3000, 3000, 3000]),  np.array([3200, 3200, 3200]) # TODO: Get from merge error data
a, b = false_merge_nodes

length = int(np.hypot(b - a))
zz, yy, xx = np.linspace(a[0], b[0], length), np.linspace(a[1], b[1], length), np.linspace(a[2], b[2], length)
connecting_line = h_arr[zz.astype(int), yy.astype(int), xx.astype(int)]

import IPython; IPython.embed(); raise SystemExit



"""
 'merge_stats': [{'seg_id': 24620, 'comp_ids': [13, 20]},
  {'seg_id': 0,
   'comp_ids': [20,
    43,
    47,
    116,
    183,
    229,
    669,
    1380,
    1570,
    1641,
    1886,
    1948,
    2900,
    3025,
    3358]},
  {'seg_id': 24133, 'comp_ids': [47, 183]},
  {'seg_id': 222431, 'comp_ids': [64, 549, 830]},
  {'seg_id': 70615, 'comp_ids': [787, 801]},
  {'seg_id': 106217, 'comp_ids': [1231, 2703]},
  {'seg_id': 398744, 'comp_ids': [1570, 3079]},
  {'seg_id': 9958, 'comp_ids': [1779, 2283, 2507]},
  {'seg_id': 347277, 'comp_ids': [1867, 2058]},
  {'seg_id': 90147, 'comp_ids': [2187, 3446]},
  {'seg_id': 348110, 'comp_ids': [2900, 3487]}],
 'split_stats': [{'comp_id': 1641,
   'seg_ids': [[186322, 186720],
    [186720, 186843],
    [401335, 596553],
    [596553, 401335]]},
  {'comp_id': 47, 'seg_ids': [[448783, 24133]]},
  {'comp_id': 1041, 'seg_ids': [[451139, 92818]]},
  {'comp_id': 1026, 'seg_ids': [[85402, 9943]]},
  {'comp_id': 2507, 'seg_ids': [[9958, 108973]]},
  {'comp_id': 2416, 'seg_ids': [[94414, 98317], [95528, 98317]]},
  {'comp_id': 43, 'seg_ids': [[44257, 14118]]},
  {'comp_id': 2231, 'seg_ids': [[244292, 252475]]},
  {'comp_id': 2900, 'seg_ids': [[347671, 348110]]},
  {'comp_id': 3025, 'seg_ids': [[9759, 109112], [378323, 381674]]},
  {'comp_id': 3313, 'seg_ids': [[354694, 354942]]},
  {'comp_id': 3358, 'seg_ids': [[112995, 113738]]}],
 'threshold': 0.5,
 'experiment': 'zebrafinch',
 'setup': '08-18_17-42_glum-cof__400k_11_micron_a',
 'network_configuration': '08-18_17-42_glum-cof__400k_11_micron_a',
 'merge_function': 'hist_quant_75',
 'run_type': '11_micron_roi_masked'}
"""



"""
ta = client['test_annotations']

In [98]: ta['zebrafinch.nodes'].find_one({'id': ta['nodes_zebrafinch_components_11_micron_roi_masked'].find_one({'component_id': 183})['id']}, {'_id': 0})
Out[98]: {'id': 316691, 'x': 44172, 'y': 49266, 'z': 51760, 'neuron_id': 310}


In [120]: b = client['08-18_17-42_glum-cof__400k_benchmark']

In [121]: b['nodes'].find_one()
Out[121]: 
{'_id': ObjectId('64e39266e1cdef95fc34c449'),
 'id': 19440000004,
 'center_z': 4018,
 'center_y': 7401,
 'center_x': 91006}

In [122]: b['edges_hist_quant_75'].find_one()
Out[122]: 
{'_id': ObjectId('64e39ef8e1cdef95fcfd77fc'),
 'u': 19440000004,
 'v': 19440000005,
 'merge_score': 0.853515625,
 'agglomerated': True}

In [123]: b['nodes'].find_one({'id': b['edges_hist_quant_75'].find_one()['u']})
Out[123]: 
{'_id': ObjectId('64e39266e1cdef95fc34c449'),
 'id': 19440000004,
 'center_z': 4018,
 'center_y': 7401,
 'center_x': 91006}


"""