"""Simplify GT naming, preserving (just) the cube ID

For example,
- source file: /wholebrain/fromhd/jkor/from_lustre/new j0126 segmentation gt/batch2/j0126 cubeSegmentor category-volume_gt__011-gpatzer-20151013-184417-final.h5
- dest file: /cajal/scratch/projects/misc/mdraw/data/j0126_h5_gt_flat_numbered/gt_11.h5

Does the same for info.txt files
"""

import logging
import hydra
from pathlib import Path

from nseg.conf import OmegaConf, DictConfig

def _noop(*args, **kwargs):
    """ No-op for dry run """
    pass


def _get_cube_id_str(orig_name: str) -> str:
    cube_id_str = orig_name[41:43]  # These 2 digits uniquely identify the cube
    # Expect 1 to 2 digit number, left-padded with zeros to 3 digits
    assert cube_id_str.isdigit() and orig_name[40] == '0'
    return cube_id_str


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{OmegaConf.to_yaml(cfg, resolve=True)}\n==')
    if cfg.data_prep.dry_run:
        copy2 = _noop
    else:
        from shutil import copy2

    source_h5_root = Path(cfg.data_prep.source_h5_root)
    dest_flat_h5_root = Path(cfg.data_prep.dest_flat_h5_root)

    source_h5_paths = list(source_h5_root.rglob('*.h5'))
    assert len(source_h5_paths) > 0

    for source_h5_path in source_h5_paths:
        cube_id_str = _get_cube_id_str(source_h5_path.name)
        dest_h5_name = f'gt_{cube_id_str}.h5'
        dest_h5_path = dest_flat_h5_root / dest_h5_name

        # logging.info(f'Copy\n  src: {source_h5_path}\n dest: {dest_h5_path}\n')
        copy2(source_h5_path, dest_h5_path)

        if cfg.data_prep.include_info:
            source_info_path = source_h5_path.with_name(f'{source_h5_path.stem}_info.txt')
            assert source_info_path.is_file()
            dest_info_name = f'gt_{cube_id_str}_info.txt'
            dest_info_path = dest_flat_h5_root / dest_info_name
            # logging.info(f'Copy\n  src: {source_h5_path}\n dest: {dest_h5_path}\n')
            copy2(source_info_path, dest_info_path)


if __name__ == '__main__':
    main()