"""
Create nseg-compatible zarr files from HDF5 GT files

Source HDF5 files use xyz axis order, resulting zarr files use zyx
"""

# TODO: Copy info to zarr attributes

import logging
import h5py
import zarr
import numpy as np
from pathlib import Path

from nseg.conf import NConf, DictConfig, hydra


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    flat_h5_root = Path(cfg.data_prep.dest_flat_h5_root)  # "dest" from earlier rename script is now the new source
    dest_zarr_flat_root = Path(cfg.data_prep.dest_zarr_flat_root)
    if not cfg.data_prep.dry_run:
        # dest_zarr_flat_root.mkdir(exist_ok=False)  # Fail if dest already exists, to avoid making a mess
        dest_zarr_flat_root.mkdir(exist_ok=True)

    source_h5_paths = list(flat_h5_root.glob('*.h5'))
    assert len(source_h5_paths) > 0

    for source_h5_path in source_h5_paths:
        zarr_name = f'{source_h5_path.stem}.zarr'
        if cfg.data_prep.zarr_zip:
            zarr_name = f'{zarr_name}.zip'
        dest_zarr_path = dest_zarr_flat_root / zarr_name

        logging.info(f'Convert\n  src: {source_h5_path}\n dest: {dest_zarr_path}\n')

        # Load all data as numpy arrays
        with h5py.File(source_h5_path, mode='r') as h5f:
            _xyz_raw_u8: np.ndarray = h5f['em_raw'][()]
            _xyz_lab_u64: np.ndarray = h5f['labels'][()]

        # xyz -> zyx
        raw_u8 = _xyz_raw_u8.transpose(2, 1, 0)
        lab_u64 = _xyz_lab_u64.transpose(2, 1, 0)

        if cfg.data_prep.crop_raw_shape is not None:
            crop_shape = np.array(cfg.data_prep.crop_raw_shape)
            logging.info(f'Cropping raw from {raw_u8.shape} to {cfg.data_prep.crop_raw_shape}\n')
            # Perform center crop to crop_raw_shape
            current_shape = np.array(raw_u8.shape)
            lo = (current_shape // 2) - (crop_shape // 2)
            hi = (current_shape // 2) + (crop_shape // 2)
            raw_u8 = raw_u8[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        if cfg.data_prep.mask_shape_like_raw_shape:
            lab_mask_u8 = np.ones_like(raw_u8).astype(np.uint8)
        else:
            lab_mask_u8 = np.ones_like(lab_u64).astype(np.uint8)

        resolution = np.array(cfg.data_prep.resolution)
        raw_shape = np.array(raw_u8.shape)
        lab_shape = np.array(lab_u64.shape)
        # print(raw_shape, lab_shape)

        lab_vx_offset = raw_shape // 2 - lab_shape // 2
        lab_nm_offset = resolution * lab_vx_offset  # scale from voxels to nanometers

        raw_attrs_dict = {'resolution': resolution.tolist(), 'offset': [0, 0, 0]}
        lab_attrs_dict = {'resolution': resolution.tolist(), 'offset': lab_nm_offset.tolist()}
        if cfg.data_prep.mask_shape_like_raw_shape:
            lab_mask_attrs_dict = {'resolution': resolution.tolist(), 'offset': [0, 0, 0]}
        else:
            lab_mask_attrs_dict = {'resolution': resolution.tolist(), 'offset': lab_nm_offset.tolist()}

        if cfg.data_prep.dry_run:  # Use volatile memorystore instead of on-disk storage
            zstore = zarr.MemoryStore()
        else:
            if cfg.data_prep.zarr_zip:
                zstore = zarr.ZipStore(dest_zarr_path)
            else:
                zstore = zarr.DirectoryStore(dest_zarr_path)
        zroot = zarr.group(store=zstore, overwrite=True)

        # zraw = zarr.array(raw_u8, chunks=cfg.data_prep.chunk_shape)
        # zlab = zarr.array(lab_u64, chunks=cfg.data_prep.chunk_shape)
        # zlab_mask = zarr.array(lab_mask_u8, chunks=cfg.data_prep.chunk_shape)

        # Dict mapping zarr group key to tuple of (array, array.attrs)
        zgroups = {
            'volumes/raw': (raw_u8, raw_attrs_dict),
            'volumes/labels/neuron_ids': (lab_u64, lab_attrs_dict),
            'volumes/labels/labels_mask': (lab_mask_u8, lab_mask_attrs_dict),
        }

        for key, (arr, attrs_dict) in zgroups.items():
            zroot[key] = zarr.array(arr, chunks=cfg.data_prep.chunk_shape)
            zroot[key].attrs.put(attrs_dict)


if __name__ == '__main__':
    main()
