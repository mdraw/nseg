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
# split_z_idx = 384

## Old 512
# raw_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth512/from_h5_em_cfw_1.0_full_512_miki.npy')
# lab_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth512/cc3d_labels_with_prob_512.npy')
# zarr_out_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth512/synth512.zarr')

## Old 1000
# raw_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/from_h5_em_cfw_1.0_cut1000_miki.npy')
# lab_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/Franz_cut1000_ratio_6.0_with_prob.npy')
# zarr_out_path = Path('/cajal/scratch/projects/misc/mdraw/data/synth1000/synth1000.zarr')

## Old 3000
# raw_path = Path('/cajal/scratch/projects/misc/mdraw/data/_old_synth3000/from_h5_em_cfw_1.0_full_3000_ana_v2.npy')
# lab_path = Path('/cajal/scratch/projects/misc/mdraw/data/_old_synth3000/labels_al_4fold_dilation_darkerlight_wborders_new.hdf5')
# zarr_out_path = Path('/cajal/scratch/projects/misc/mdraw/data/_old_synth3000/synth3000.zarr')

## 3000 v2
# raw_path = Path('/cajal/nvmescratch/users/riegerfr/miki_seg/3k_65000n_v2_labels_al_4fold_dilation_darkerlight_wborders_new.npy')
# lab_path = Path('/cajal/scratch/users/anaml/3k_65000n_v2/labels_al_4fold_dilation_darkerlight_wborders_new.hdf5')
# zarr_out_path = Path('/cajal/nvmescratch/users/mdraw/data/synth3000v2/synth3000v2.zarr')  # SSD

## 3000 v3
raw_path = Path('/cajal/nvmescratch/users/riegerfr/miki_seg/3k_65000n_v3_test_labels_al_4fold_dilation_darkerlight_wborders_new_32b.npy')
lab_path = Path('/cajal/scratch/users/anaml/3k_65000n_v3_test/labels_al_4fold_dilation_darkerlight_wborders_new_32b.hdf5')
zarr_out_path = Path('/cajal/nvmescratch/users/mdraw/data/synth3000v3/synth3000v3.zarr')  # SSD



for split in splits:

    if split is not None:
        zpath = zarr_out_path.with_stem(f'{zarr_out_path.stem}_{split}')
    else:
        zpath = zarr_out_path


    # Synthetic cubes are fully labeled -> zero offset, same res for all subvolumes
    shared_attrs_dict = {'offset': [0, 0, 0], 'resolution': [20, 9, 9]}

    print(f'Loading {raw_path}')
    raw_f32 = np.load(raw_path)
    print(f'Loading {lab_path}')
    if lab_path.suffix == '.npy':
        lab = np.load(lab_path)
    elif lab_path.suffix in ['.hdf5', '.h5']:
        with h5py.File(lab_path, mode='r') as h5f:
            keys = list(h5f.keys())
            assert len(keys) == 1
            lab = h5f[keys[0]][()]


    raw_rescaled = (raw_f32 / 2. + 0.5) * 255  # [-1, 1] -> [0, 255]
    del raw_f32
    raw_u8 = raw_rescaled.astype(np.uint8)
    del raw_rescaled
    while raw_u8.ndim > 3:
        raw_u8 = raw_u8[0]  # Squeeze until it's actual 3D

    lab_u64 = lab.astype(np.uint64)

    # Synthetic cubes are fully labeled -> nothing to be masked out
    lab_mask_u8 = np.ones_like(lab_u64, dtype=np.uint8)

    print(f'Creating {zpath}')
    zstore = zarr.DirectoryStore(zpath)
    zroot = zarr.group(store=zstore, overwrite=True)


    if split == 'tr':
        raw_u8 = raw_u8[:split_z_idx, :, :]
        lab_u64 = lab_u64[:split_z_idx, :, :]
        lab_mask_u8 = lab_mask_u8[:split_z_idx, :, :]

    elif split == 'val':
        raw_u8 = raw_u8[split_z_idx:, :, :]
        lab_u64 = lab_u64[split_z_idx:, :, :]
        lab_mask_u8 = lab_mask_u8[split_z_idx:, :, :]

    zgroups = {
        'volumes/raw': raw_u8,
        'volumes/labels/neuron_ids': lab_u64,
        'volumes/labels/labels_mask': lab_mask_u8,
    }

    for key, arr in zgroups.items():
        zroot[key] = arr
        zroot[key].attrs.put(shared_attrs_dict)
