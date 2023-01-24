import napari
import numpy as np
import zarr


fn = '/home/m4/dev/expmiclsd/validation_data.zarr'
z = zarr.open(fn, mode='r')

# import IPython ; IPython.embed(); raise SystemExit

viewer = napari.Viewer()
viewer.add_image(z.raw[0], name='raw')
viewer.add_labels(z.labels[0], name='lab')

napari.run()
# viewer = napari.view_image(np.moveaxis(h['image'][()], -1, 0), name='raw')
#viewer.add_labels(h['label'], name='label')
