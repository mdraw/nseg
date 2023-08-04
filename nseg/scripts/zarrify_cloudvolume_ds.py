"""
Create nseg-compatible zarr files from cloudvolume datasets

Source cloudvolume datasets use xyz axis order, resulting zarr files use zyx

Requires at least 700GB of RAM to run!
"""

import logging
import zarr
import numpy as np
from pathlib import Path

from cloudvolume import CloudVolume

from nseg.conf import NConf, DictConfig, hydra


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    cloudvolume_url = cfg.cloudvolume_raw_to_zarr.cloudvolume_url
    dest_zarr_path = Path(cfg.cloudvolume_raw_to_zarr.dest_zarr_path)
    raw_key = cfg.cloudvolume_raw_to_zarr.raw_key
    keep_original_layout = cfg.cloudvolume_raw_to_zarr.keep_original_layout

    raw_vol = CloudVolume(
        cloudvolume_url,
        bounded=True,
        progress=True
    )

    logging.info(f'Load data from cloudvolume {cloudvolume_url=}')
    if keep_original_layout:
        # Download data as xyzc
        raw_u8 = raw_vol[()]
    else:
        # Download data as xyzc, squeeze to xyz, swap axes to zyx
        raw_u8 = raw_vol[()].squeeze(-1).swapaxes(0, 2)
    logging.info('Loaded.')

    logging.info(f'Convert\n  src: {cloudvolume_url}\n dest: {dest_zarr_path}\n')

    # Load knossos dataset
    resolution = np.array(cfg.cloudvolume_raw_to_zarr.resolution)
    raw_attrs_dict = {'resolution': resolution.tolist(), 'offset': [0, 0, 0]}

    zstore = zarr.DirectoryStore(dest_zarr_path)
    zroot = zarr.group(store=zstore, overwrite=True)

    zroot.create_dataset(raw_key, shape=raw_u8.shape, chunks=cfg.cloudvolume_raw_to_zarr.chunk_shape, dtype=raw_u8.dtype)
    zroot[raw_key] = raw_u8
    logging.info(f'Finished writing to dest {dest_zarr_path}')
    logging.info(f'Add metadata {raw_attrs_dict} to data.')
    zroot[raw_key].attrs.put(raw_attrs_dict)


if __name__ == '__main__':
    main()
