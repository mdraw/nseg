import napari
import numpy as np

raw_path = '/home/m4/data/synth/from_h5_em_cfw_1.0_full_512_miki.npy'
lab_path = '/home/m4/data/synth/cc3d_labels_with_prob_512.npy'

raw = np.load(raw_path)[0, 0]
lab = np.load(lab_path)

viewer = napari.Viewer()
viewer.add_image(raw, name='raw')
viewer.add_labels(lab, name='lab')
napari.run()
