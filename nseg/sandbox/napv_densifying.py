print('Deprecated, use densify_labels and napv instead.')

import napari
import numpy as np
import zarr

from funlib.segment.arrays import replace_values

# @numba.jit(nopython=True, parallel=True)
def densify_labels(lab: np.ndarray, dtype=np.uint64) -> tuple[np.ndarray, int]:
    old_ids = np.unique(lab)
    num_ids = old_ids.size
    new_ids = np.arange(num_ids, dtype=old_ids.dtype)
    replace_values(lab, old_ids, new_ids, inplace=True)
    # dense_lab = np.zeros_like(lab, dtype=dtype)
    # for o, n in zip(old_ids, new_ids):
    #     dense_lab[lab == o] = n
    return lab.astype(dtype), num_ids

# zarr_seg_path = '/cajal/scratch/projects/misc/mdraw/lsdex/v1/inference/09-11_22-19_obvious-__400k_benchmark/seg.zarr'
# zarr_seg_path = '/cajal/scratch/projects/misc/mdraw/lsdex/v1/inference/09-11_22-05_savory-b__400k_benchmark/seg.zarr'

zarr_seg_path = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/seg.zarr'  # 77417 IDs

fseg = zarr.open(zarr_seg_path, mode='r').volumes.segmentation


# z, y, x = 512, 1024, 1024
# dz, dy, dx = 512, 1024, 1024
#
# seg = fseg[z:z+dz, y:y+dy, x:x+dx]

seg = fseg[()]
print(f'Densifying IDs...')
seg, num_ids = densify_labels(seg, dtype=np.uint16)
print(f'Done. Found {num_ids} IDs.')

# seg = fseg[100:200, 100:200, 100:200]

# import IPython; IPython.embed(); raise SystemExit
# seg = np.load(seg_path)

viewer = napari.Viewer()
# viewer.add_image(raw, name='raw')
viewer.add_labels(
    seg,
    name='segmentation',
    scale=(2, 1, 1)
)


viewer.camera.center = (540, 600, 600)
viewer.camera.angles = (-3, 32, 148)
viewer.camera.zoom = 0.6



napari.run()
