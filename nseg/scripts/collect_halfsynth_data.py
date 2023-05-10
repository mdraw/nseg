""" Copy and rename halfsynthetic j0126 data

Association example:

Original data:
- /cajal/scratch/from_wholebrain/fromhd/jkor/from_lustre/new j0126 segmentation gt/batch2/j0126 cubeSegmentor category-volume_gt__011-gpatzer-20151013-184417-final.h5
- /cajal/scratch/projects/misc/mdraw/data/j0126_h5_gt_flat_numbered/gt_11.h5

Halfsynth data:
- /cajal/nvmescratch/users/riegerfr/miki_seg/j0126_gt_v_16_j0126 cubeSegmentor category-volume_gt__011-gpatzer-20151013-184417-final.h5.npy

Destination (halfsynth with original labels, in common dataset format)
-> /cajal/scratch/projects/misc/mdraw/data/halfsynth_all/halfsynth_v16_gt_11.h5


After running, convert to zarr using:
    python -m nseg.scripts.zarrify_j0126 data_prep.crop_raw_shape=null data_prep.dest_flat_h5_root=/cajal/scratch/projects/misc/mdraw/data/halfsynth_h5_all data_prep.dest_zarr_flat_root=/cajal/scratch/projects/misc/mdraw/data/halfsynth_zarr_all


"""


# TODO: Check axis order


from shutil import copy2
import logging
from pathlib import Path
import h5py
import numpy as np


from nseg.conf import NConf, DictConfig, hydra

def _noop(*args, **kwargs):
    """ No-op for dry run """
    pass


def _get_cube_id_str(orig_name: str) -> str:
    cube_id_str = orig_name[41:43]  # These 2 digits uniquely identify the cube
    # Expect 1 to 2 digit number, left-padded with zeros to 3 digits
    assert cube_id_str.isdigit() and orig_name[40] == '0'
    return cube_id_str

def _get_halfsynth_name(original_name, version, prefix='j0126_gt_v_', postfix='.npy'):
    return f'{prefix}{version}_{original_name}{postfix}'


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.log_config:
        logging.info(f'{Path(__file__).stem} config:\n==\n{NConf.to_yaml(cfg, resolve=True)}\n==')

    if cfg.data_prep.dry_run:
        raise NotImplementedError

    source_h5_root = Path(cfg.data_prep.source_h5_root)
    dest_flat_h5_root = Path(cfg.data_prep.dest_flat_h5_root)
    halfsynth_name_prefix = cfg.data_prep.halfsynth_name_prefix
    source_halfsynth_h5_origin_root = Path(cfg.data_prep.source_halfsynth_h5_origin_root)
    dest_halfsynth_h5_flat_root = Path(cfg.data_prep.dest_halfsynth_h5_flat_root)

    dest_halfsynth_h5_flat_root.mkdir(exist_ok=False)  # Fail if dest already exists, to avoid making a mess


    source_h5_paths = list(source_h5_root.rglob('*.h5'))
    assert len(source_h5_paths) > 0

    versions = cfg.data_prep.halfsynth_versions

    for source_h5_path in source_h5_paths:
        cube_id_str = _get_cube_id_str(source_h5_path.name)

        if cfg.data_prep.halfsynth_use_only_training_cubes:
            if int(cube_id_str) not in cfg.data_prep.split_lists.training:
                continue

        for version in versions:
            halfsynth_source_name = _get_halfsynth_name(source_h5_path.name, version=version)
            halfsynth_source_path = source_halfsynth_h5_origin_root / halfsynth_source_name

            halfsynth_dest_name = f'{halfsynth_name_prefix}_v{version}_{cube_id_str}.h5'
            dest_h5_path = dest_halfsynth_h5_flat_root / halfsynth_dest_name

            logging.info(f'Copy\n  raw src: {halfsynth_source_path}\n dest: {dest_h5_path}\n')
            copy2(source_h5_path, dest_h5_path)
            logging.info(f'Replacing raw data of dest: {dest_h5_path} with {halfsynth_source_path} contents')
            synth_raw_f32 = np.load(halfsynth_source_path)
            synth_raw_rescaled = (synth_raw_f32 / 2. + 0.5) * 255  # [-1, 1] -> [0, 255]
            synth_raw_u8 = synth_raw_rescaled.astype(np.uint8)
            while synth_raw_u8.ndim > 3:
                synth_raw_u8 = synth_raw_u8[0]  # Squeeze until it's actual 3D
            with h5py.File(dest_h5_path, mode='r+') as f:
                del f['em_raw']
                f['em_raw'] = synth_raw_u8


if __name__ == '__main__':
    main()

