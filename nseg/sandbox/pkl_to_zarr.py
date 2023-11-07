import logging
import pickle
import h5py
from numcodecs import Zstd

from nseg.conf import NConf, DictConfig, hydra

import numpy as np
import zarr

from pathlib import Path


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    lab_pkl_path = Path(cfg.pkl_to_zarr.lab_input_path)
    raw_pkl_path = Path(cfg.pkl_to_zarr.raw_input_path)

    logging.info(f'Loading label data {lab_pkl_path}')
    with open(lab_pkl_path, 'rb') as f:
        seg = pickle.load(f)
    logging.info(f'Loading raw data from {raw_pkl_path}')
    raw = np.load(raw_pkl_path)

    raw = raw.squeeze((0, 1))  # Remove singleton dimensions
    raw *= 255  # Scale from [0, 1] to [0, 255]
    raw_u8 = raw.astype(np.uint8)

    lab_u64 = seg.astype(np.uint64)

    lab_mask_u8 = np.ones_like(lab_u64).astype(np.uint8)

    resolution = np.array(cfg.pkl_to_zarr.resolution)
    raw_shape = np.array(raw_u8.shape)
    lab_shape = np.array(lab_u64.shape)
    logging.info(f'{raw_shape=}, {lab_shape=}')

    lab_vx_offset = raw_shape // 2 - lab_shape // 2
    lab_nm_offset = resolution * lab_vx_offset  # scale from voxels to nanometers

    raw_attrs_dict = {'resolution': resolution.tolist(), 'offset': [0, 0, 0]}
    lab_attrs_dict = {'resolution': resolution.tolist(), 'offset': lab_nm_offset.tolist()}
    lab_mask_attrs_dict = {'resolution': resolution.tolist(), 'offset': lab_nm_offset.tolist()}

    dest_zarr_path = Path(cfg.pkl_to_zarr.dest_zarr_path)
    logging.info(f'Writing {dest_zarr_path}')

    if cfg.pkl_to_zarr.dry_run:  # Use volatile memorystore instead of on-disk storage
        zstore = zarr.MemoryStore()
    else:
        if cfg.pkl_to_zarr.zarr_zip:
            zstore = zarr.ZipStore(dest_zarr_path)
        else:
            zstore = zarr.DirectoryStore(dest_zarr_path)
    zroot = zarr.group(store=zstore, overwrite=True)

    # zraw = zarr.array(raw_u8, chunks=cfg.pkl_to_zarr.chunk_shape)
    # zlab = zarr.array(lab_u64, chunks=cfg.pkl_to_zarr.chunk_shape)
    # zlab_mask = zarr.array(lab_mask_u8, chunks=cfg.pkl_to_zarr.chunk_shape)

    # Dict mapping zarr group key to tuple of (array, array.attrs)
    zgroups = {
        'volumes/raw': (raw_u8, raw_attrs_dict),
        'volumes/labels/neuron_ids': (lab_u64, lab_attrs_dict),
        'volumes/labels/labels_mask': (lab_mask_u8, lab_mask_attrs_dict),
    }

    for key, (arr, attrs_dict) in zgroups.items():
        zroot[key] = zarr.array(arr, chunks=cfg.pkl_to_zarr.chunk_shape, compressor=Zstd(5))
        zroot[key].attrs.put(attrs_dict)


if __name__ == '__main__':
    main()
