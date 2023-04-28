"""Convert synthetic GT from npy to nseg-compatible zarr"""

import numpy as np
import zarr
from pathlib import Path

split_z_idx = 384
chunk_shape = (128, 128, 128)

# Optionally split into 2 cubes (tr, val)
splits = [None]
# splits = [None, 'tr', 'val']

# raw_path = Path('/home/m4/data/synth/from_h5_em_cfw_1.0_full_512_miki.npy')
# lab_path = Path('/home/m4/data/synth/cc3d_labels_with_prob_512.npy')
# zarr_out_path = Path('/home/m4/data/synth/synth512.zarr')

raw_path = Path('/home/m4/data/synth/from_h5_em_cfw_1.0_full_512_miki.npy')
lab_path = Path('/home/m4/data/synth/cc3d_labels_with_prob_512.npy')
zarr_out_path = Path('/home/m4/data/synth/synth512.zarr')

for split in splits:

    if split is not None:
        zpath = zarr_out_path.with_stem(f'{zpath.stem}_{split}')
    else:
        zpath = zarr_out_path

    print(f'Creating {zpath}')

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


    zstore = zarr.DirectoryStore(zpath)
    zroot = zarr.group(store=zstore, overwrite=True)


    if split == 'tr':
        raw_u8 = raw_u8[:split_z_idx, :, :]
        lab_i64 = lab_i64[:split_z_idx, :, :]
        lab_mask_u8 = lab_mask_u8[:split_z_idx, :, :]

    elif split == 'val':
        raw_u8 = raw_u8[split_z_idx:, :, :]
        lab_i64 = lab_i64[split_z_idx:, :, :]
        lab_mask_u8 = lab_mask_u8[split_z_idx:, :, :]

    zgroups = {
        'volumes/raw': raw_u8,
        'volumes/labels/neuron_ids': lab_i64,
        'volumes/labels/labels_mask': lab_mask_u8,
    }

    for key, arr in zgroups.items():
        zroot[key] = arr
        zroot[key].attrs.put(shared_attrs_dict)
