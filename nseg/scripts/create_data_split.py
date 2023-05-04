"""Simplify GT naming, preserving (just) the cube ID

For example,
- source file: /wholebrain/fromhd/jkor/from_lustre/new j0126 segmentation gt/batch2/j0126 cubeSegmentor category-volume_gt__011-gpatzer-20151013-184417-final.h5
- dest file: /cajal/scratch/projects/misc/mdraw/data/j0126_h5_gt_flat_numbered/gt_11.h5

Does the same for info.txt files
"""

import logging
import itertools
from pathlib import Path

from nseg.conf import NConf, DictConfig, hydra

def _noop(*args, **kwargs):
    """ No-op for dry run """
    pass


def _get_zarrcube_id(fstem: str) -> int:
    cube_id_str = fstem[-2:]  # These 2 digits uniquely identify the cube
    assert cube_id_str.isdigit()
    return int(cube_id_str)


def _get_split_dirname(cube_id: int, split_lists: DictConfig) -> str:
    for list_name, cube_list in split_lists.items():
        if cube_id in cube_list:
            return str(list_name)


def validate_split_lists(split_lists: DictConfig) -> None:
    split_lists = NConf.to_object(split_lists)
    for a, b in itertools.combinations(split_lists.values(), 2):
        # Make sure nothing ends up in more than one split list
        assert set.isdisjoint(set(a), set(b))
    # Make sure that every cube is in one of the lists
    assert set.union(*[set(v) for v in split_lists.values()]) == set(range(29))


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    split_lists = cfg.data_prep.split_lists
    validate_split_lists(split_lists)
    zarr_flat_root = Path(cfg.data_prep.dest_zarr_flat_root)
    dest_zarr_split_root = Path(cfg.data_prep.dest_zarr_split_root)

    if cfg.data_prep.dry_run:
        copytree = _noop
    else:
        from shutil import copytree
        dest_zarr_split_root.mkdir(exist_ok=False)  # Fail if dest already exists, to avoid making a mess

    source_zarr_paths = list(zarr_flat_root.glob('*.zarr'))
    assert len(source_zarr_paths) > 0

    for source_zarr_path in source_zarr_paths:
        cube_id = _get_zarrcube_id(source_zarr_path.stem)
        split_dirname = _get_split_dirname(cube_id, split_lists)

        dest_zarr_path = dest_zarr_split_root / split_dirname / source_zarr_path.name

        logging.info(f'Copy\n  src: {source_zarr_path}\n dest: {dest_zarr_path}\n')
        copytree(source_zarr_path, dest_zarr_path)

if __name__ == '__main__':
    main()
