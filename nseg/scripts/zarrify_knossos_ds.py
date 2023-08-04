"""
Create nseg-compatible zarr files from KNOSSOS datasets

Source KNOSSOS datasets use xyz axis order, resulting zarr files use zyx

Requires at least 700GB of RAM to run!
"""

import logging
import zarr
import numpy as np
from pathlib import Path
import gc

import knossos_utils

from nseg.conf import NConf, DictConfig, hydra


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    knossos_conf_path = Path(cfg.knossos_raw_to_zarr.knossos_conf_path)
    dest_zarr_path = Path(cfg.knossos_raw_to_zarr.dest_zarr_path)
    raw_key = cfg.knossos_raw_to_zarr.raw_key

    kd = knossos_utils.KnossosDataset(str(knossos_conf_path))

    kd_offset = (0, 0, 0)

    # kd_size = kd.boundary // 2  # half dataset for debugging
    kd_size = kd.boundary  # full dataset

    logging.info(f'Load data from {kd_offset=}, {kd_size=}...')
    raw_u8 = kd.load_raw(
        offset=kd_offset, size=kd_size, mag=1
    )  # Data is already zyx
    logging.info('Loaded.')

    # Make sure we don't have any deleted objects that are not yet gc'd
    for _ in range(3):
        gc.collect()

    logging.info(f'Convert\n  src: {knossos_conf_path}\n dest: {dest_zarr_path}\n')

    # Load knossos dataset
    resolution = list(cfg.knossos_raw_to_zarr.resolution)
    raw_attrs_dict = {'resolution': resolution, 'offset': [0, 0, 0]}

    zstore = zarr.DirectoryStore(dest_zarr_path)
    zroot = zarr.group(store=zstore, overwrite=True)

    zroot.create_dataset(raw_key, shape=raw_u8.shape, chunks=cfg.knossos_raw_to_zarr.chunk_shape, dtype=raw_u8.dtype)
    zroot[raw_key] = raw_u8
    logging.info(f'Finished writing to dest {dest_zarr_path}')
    logging.info(f'Add metadata {raw_attrs_dict} to data.')
    zroot[raw_key].attrs.put(raw_attrs_dict)


if __name__ == '__main__':
    main()
