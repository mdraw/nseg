import napari
import numpy as np
import zarr

from funlib.segment.arrays import replace_values

# zarr_seg_path = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/seg_densified.zarr'  # 77417 IDs
#
#
# fseg = zarr.open(zarr_seg_path, mode='r').volumes.segmentation
# seg = fseg[()]
#
# viewer = napari.Viewer()
# # viewer.add_image(raw, name='raw')
# viewer.add_labels(
#     seg,
#     name='segmentation',
#     scale=(2, 1, 1),
#     num_colors=100,
# )
#
# viewer.camera.center = (540, 600, 600)
# viewer.camera.angles = (-3, 32, 148)
# viewer.camera.zoom = 0.55
#
# napari.run()


# zarr_aff_path = '/home/m4/data/lsd/zfinch_11_micron_crop.zarr'
# faff = zarr.open(zarr_aff_path, mode='r').volumes.affs
# aff = faff[()]
#
# viewer = napari.Viewer()
# # viewer.add_image(raw, name='raw')
# viewer.add_image(
#     aff,
#     name='aff',
#     scale=(2, 1, 1),
# )
#
# viewer.camera.center = (540, 600, 600)
# viewer.camera.angles = (-3, 32, 148)
# viewer.camera.zoom = 0.55
#
# napari.run()


zarr_raw_path = '/home/m4/ma-local/raw_j0126_11_micron.zarr'
fraw = zarr.open(zarr_raw_path, mode='r').volumes.raw
raw = fraw[()]

viewer = napari.Viewer()
# viewer.add_image(raw, name='raw')
viewer.add_image(
    raw,
    name='raw',
    scale=(2, 1, 1),
)

viewer.camera.center = (540, 600, 600)
viewer.camera.angles = (-3, 32, 148)
viewer.camera.zoom = 0.55

napari.run()
