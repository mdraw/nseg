import numpy as np
import zarr
import napari
from skimage.measure import label
from pathlib import Path

from funlib.segment.arrays import replace_values


def densify_labels(lab: np.ndarray, dtype=np.uint64) -> tuple[np.ndarray, int]:
    old_ids = np.unique(lab)
    num_ids = old_ids.size
    new_ids = np.arange(num_ids, dtype=old_ids.dtype)
    replace_values(lab, old_ids, new_ids, inplace=True)
    # dense_lab = np.zeros_like(lab, dtype=dtype)
    # for o, n in zip(old_ids, new_ids):
    #     dense_lab[lab == o] = n
    return lab.astype(dtype), num_ids


inf_root = Path('/u/mdraw/lsdex/v1/inference')

runs = [
    '09-27_20-33_rapid-si__400k_benchmark',
    '09-11_21-54_largo-tu__400k_benchmark',
    '08-19_01-34_absolute__400k_benchmark_b',
]

# fraw = '/cajal/scratch/projects/misc/mdraw/data/fullraw_fromknossos_j0126/j0126.zarr'
# zraw = zarr.open(f'{fraw}/volumes/raw', mode='r')

# fraw = '/cajal/scratch/projects/misc/mdraw/data/raw_full_j0126/j0126.zarr'
# zraw = zarr.open(f'{fraw}/raw', mode='r')

fraw = f'/cajal/scratch/projects/misc/mdraw/data/raw_j0126_11_micron.zarr'
zraw = zarr.open(f'{fraw}/volumes/raw', mode='r')

z = zraw.shape[0] // 2

# y, x = 256, 256
# dy, dx = 768, 768

y, x = 512, 512
dy, dx = 512, 512

raw = zraw[z, y:y + dy, x:x + dx]


viewer = napari.Viewer()

for run in runs:

    fpred = inf_root / run / 'affs.zarr'
    fseg = inf_root / run / 'seg.zarr'

    zseg = zarr.open(f'{fseg}/volumes/segmentation', mode='r')
    zaff = zarr.open(f'{fpred}/volumes/pred_affs', mode='r')

    seg = zseg[z, y:y+dy, x:x+dx]
    seg, num_ids = densify_labels(seg)
    aff = zaff[:, z, y:y+dy, x:x+dx]  # channel-first
    aff = np.moveaxis(aff, 0, -1)  # convert to channel-last format for visualization

    print(raw.shape, seg.shape)


    viewer.add_image(
        raw,
        # np.pad(raw, (0, 100), mode='constant', constant_values=255),  # pad with white to introduce grid margins
        scale=(9, 9),
        name=f'raw_{run}',
    )
    viewer.add_image(
        aff,  # channel-last
        scale=(9, 9),
        # channel_axis=0,
        name=f'affinities_{run}',
        rgb=True,
    )
    viewer.add_labels(
        seg,
        scale=(9, 9),
        name=f'segmentation_{run}',
        num_colors=np.unique(seg).size,
    )

# viewer.grid.enabled = True
# n_colums = 4
# viewer.grid.shape = (-1, n_colums)  # 3 columns

viewer.scale_bar.unit = 'nm'
viewer.scale_bar.visible = True


viewer.camera.zoom = 0.1113  # zoom value for 768x768 1:1 pixel mapping for screenshots


napari.run()
