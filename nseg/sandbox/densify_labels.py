from pathlib import Path

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


dtype = np.uint32

input_path = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/seg.zarr'
input_path = Path(input_path).expanduser()
output_path = input_path.with_stem(f'{input_path.stem}_densified')

inzarr = zarr.open(input_path, mode='r').volumes.segmentation

inarr = inzarr[()]  # Protect against overwriting

print(f'Densifying IDs...')
seg, num_ids = densify_labels(inarr, dtype=np.uint16)
print(f'Done. Found {num_ids} IDs.')

outzarr = zarr.open(output_path, mode='w')
outzarr.create_dataset('volumes/segmentation', data=seg, chunks=(64, 64, 64), dtype=dtype)

