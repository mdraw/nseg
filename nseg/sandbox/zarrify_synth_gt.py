import numpy as np
import zarr

enable_split = True

raw_path = '/home/m4/data/synth/from_h5_em_cfw_1.0_full_512_miki.npy'
lab_path = '/home/m4/data/synth/cc3d_labels_with_prob_512.npy'
zpath = '/home/m4/data/synth/synth512.zarr'

zgroup_paths = [
    'volumes/raw',
    'volumes/labels/neuron_ids',
    'volumes/labels/labels_mask',
]

# Synthetic cubes are fully labeled -> zero offset, same res for all subvolumes
shared_attrs_dict = {'offset': [0, 0, 0], 'resolution': [20, 9, 9]}

raw_f32 = np.load(raw_path)
lab_u32 = np.load(lab_path)

raw_rescaled = (raw_f32 / 2. + 0.5) * 255  # [-1, 1] -> [0, 255]
raw_u8 = raw_rescaled.astype(np.uint8)
assert raw_u8.ndim == 5
raw_u8 = raw_u8[0, 0]  # -> 3D

lab_i64 = lab_u32.astype(np.int64)

# Synthetic cubes are fully labeled -> nothing to be masked out
lab_mask_u8 = np.ones_like(lab_i64, dtype=np.uint8)

chunk_shape = (256, 256, 256)

zstore = zarr.DirectoryStore(zpath)
zroot = zarr.group(store=zstore, overwrite=True)
# zroot.create_groups(*zgroup_paths)

zgroups = {
    'volumes/raw': raw_u8,
    'volumes/labels/neuron_ids': lab_i64,
    'volumes/labels/labels_mask': lab_mask_u8,
}

for key, arr in zgroups.items():
    zroot[key] = arr
    zroot[key].attrs.put(shared_attrs_dict)

print('Done')
