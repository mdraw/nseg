"""Convert synthetic GT from npy to nseg-compatible zarr"""

import numpy as np
import zarr
import h5py
from pathlib import Path

# chunk_shape = (128, 128, 128)
chunk_shape = (256, 256, 256)

# Optionally split into 2 cubes (tr, val)
splits = [None]
# splits = [None, 'tr', 'val']
split_z_idx = 384

# raw_path = Path('/home/m4/data/synth/from_h5_em_cfw_1.0_full_512_miki.npy')
# lab_path = Path('/home/m4/data/synth/cc3d_labels_with_prob_512.npy')
# zarr_out_path = Path('/home/m4/data/synth/synth512.zarr')

# raw_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/from_h5_em_cfw_1.0_cut1000_miki.npy')
# lab_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/Franz_cut1000_ratio_6.0_with_prob.npy')
# zarr_out_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/synth1000.zarr')

raw_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth3000/from_h5_em_cfw_1.0_full_3000_ana_v2.npy')
lab_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth3000/labels_al_4fold_dilation_darkerlight_wborders_new.hdf5')
zarr_out_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth3000/synth3000.zarr')


for split in splits:

    if split is not None:
        zpath = zarr_out_path.with_stem(f'{zarr_out_path.stem}_{split}')
    else:
        zpath = zarr_out_path

    print(f'Creating {zpath}')

    # Synthetic cubes are fully labeled -> zero offset, same res for all subvolumes
    shared_attrs_dict = {'offset': [0, 0, 0], 'resolution': [20, 9, 9]}

    print(f'Loading {raw_path}')
    raw_f32 = np.load(raw_path)
    print(f'Loading {lab_path}')
    if lab_path.suffix == '.npy':
        lab = np.load(lab_path)
    elif lab_path.suffix in ['.hdf5', '.h5']:
        with h5py.File(lab_path, mode='r') as h5f:
            lab = h5f['label_values'][()]


    raw_rescaled = (raw_f32 / 2. + 0.5) * 255  # [-1, 1] -> [0, 255]
    del raw_f32
    raw_u8 = raw_rescaled.astype(np.uint8)
    del raw_rescaled
    assert raw_u8.ndim == 5
    raw_u8 = raw_u8[0, 0]  # -> 3D

    lab_i64 = lab.astype(np.int64)

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
