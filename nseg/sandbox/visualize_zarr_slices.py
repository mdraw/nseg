import numpy as np
import zarr
import napari
from skimage.measure import label

fraw = '/home/m4/ma-local/raw_j0126_11_micron.zarr'
fpred = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/affs.zarr'
ffrag = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/frag.zarr'
fseg = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/seg_densified.zarr'


zraw = zarr.open(f'{fraw}/volumes/raw', mode='r')
zfra = zarr.open(f'{ffrag}/volumes/fragments', mode='r')
zseg = zarr.open(f'{fseg}/volumes/segmentation', mode='r')

zaff = zarr.open(f'{fpred}/volumes/pred_affs', mode='r')
zlsd = zarr.open(f'{fpred}/volumes/pred_lsds', mode='r')
zbou = zarr.open(f'{fpred}/volumes/pred_boundaries', mode='r')
zbdt = zarr.open(f'{fpred}/volumes/pred_boundary_distmap', mode='r')
zhar = zarr.open(f'{fpred}/volumes/pred_hardness', mode='r')


z = zraw.shape[0] // 2

y, x = 256, 256
dy, dx = 768, 768

raw = zraw[z, y:y+dy, x:x+dx]
fra = zfra[z, y:y+dy, x:x+dx]
seg = zseg[z, y:y+dy, x:x+dx]
bou = zbou[z, y:y+dy, x:x+dx]
bdt = zbdt[z, y:y+dy, x:x+dx]
har = zhar[z, y:y+dy, x:x+dx]

aff = zaff[:, z, y:y+dy, x:x+dx]  # channel-first
lsd = zlsd[:, z, y:y+dy, x:x+dx]  # channel-first
aff = np.moveaxis(aff, 0, -1)  # convert to channel-last format for visualization
lsd = np.moveaxis(lsd, 0, -1)  # convert to channel-last format for visualization

fra = label(fra, 0)

print(raw.shape, seg.shape)

viewer = napari.Viewer()


viewer.add_image(
    bou,
    scale=(9, 9),
    name='boundaries',
    colormap='viridis',
)
viewer.add_image(
    bdt,
    scale=(9, 9),
    name='boundary_distmap',
    colormap='viridis',
)
viewer.add_image(
    har,
    scale=(9, 9),
    name='hardness',
    colormap='viridis',
)
viewer.add_image(
    lsd[..., 9],
    scale=(9, 9),
    name='lsds10',
    colormap='viridis',
)
viewer.add_image(
    lsd[..., 6:9],
    scale=(9, 9),
    name='lsd789',
)
viewer.add_image(
    lsd[..., 3:6],
    scale=(9, 9),
    name='lsds456',
)
viewer.add_image(
    lsd[..., :3],
    scale=(9, 9),
    name='lsds123',
)
viewer.add_labels(
    seg,
    scale=(9, 9),
    name='segmentation',
    num_colors=np.unique(seg).size,
)
viewer.add_labels(
    fra,
    scale=(9, 9),
    name='fragments',
)
viewer.add_image(
    aff,  # channel-last
    scale=(9, 9),
    # channel_axis=0,
    name='affinities',
    rgb=True,
)
viewer.add_image(
    raw,
    # np.pad(raw, (0, 100), mode='constant', constant_values=255),  # pad with white to introduce grid margins
    scale=(9, 9),
    name='raw',
)

# viewer.grid.enabled = True
n_colums = 4
viewer.grid.shape = (-1, n_colums)  # 3 columns

viewer.scale_bar.unit = 'nm'
viewer.scale_bar.visible = True

napari.run()