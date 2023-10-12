import napari
import numpy as np
import zarr


# zarr_seg_path = '/home/m4/ma-local/09-11_22-05_savory-b__400k_11_micron/seg_densified.zarr'  # 77417 IDs
zarr_seg_path = '/home/m4/ma-local/09-27_20-33_rapid-si__400k_11_micron/seg_densified.zarr'  # 52572 IDs

# select_id = None
select_id = 17369

fseg = zarr.open(zarr_seg_path, mode='r').volumes.segmentation
seg = fseg[()]

if select_id is not None:
    seg = seg == select_id

viewer = napari.Viewer()
# viewer.add_image(raw, name='raw')
viewer.add_labels(
    seg,
    name='segmentation',
    scale=(2, 1, 1),
    num_colors=100,
)

viewer.camera.center = (540, 600, 600)
viewer.camera.angles = (-3, 32, 148)
viewer.camera.zoom = 0.55

napari.run()
