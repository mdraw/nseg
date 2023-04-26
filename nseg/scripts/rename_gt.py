"""Simplify GT naming, preserving (just) the cube ID

For example,
- source file: /wholebrain/fromhd/jkor/from_lustre/new j0126 segmentation gt/batch2/j0126 cubeSegmentor category-volume_gt__011-gpatzer-20151013-184417-final.h5
- target file: /cajal/scratch/projects/misc/mdraw/data/j0126_h5_gt_flat_numbered/gt_11.h5

Does the same for info.txt files
"""


from pathlib import Path
from shutil import copy2

include_info = True
dry_run = True

if dry_run:
    copy2 = lambda x, y: print(f'src: {x}\ndst: {y}\n')

# source_h5_path = Path('/cajal/scratch/projects/misc/mdraw/data/flat_h5_j0126_gt_from_wb/')

source_h5_root = Path('/wholebrain/fromhd/jkor/from_lustre/new j0126 segmentation gt')

target_h5_root = Path('/cajal/scratch/projects/misc/mdraw/data/j0126_h5_gt_flat_numbered')

source_h5_paths = list(source_h5_root.rglob('*.h5'))
assert len(source_h5_paths) > 0


for source_h5_path in source_h5_paths:
    cube_id_str = source_h5_path.name[41:43]
    assert cube_id_str.isdigit() and source_h5_path.name[40] == '0'
    target_h5_name = f'gt_{cube_id_str}.h5'
    target_h5_path = target_h5_root / target_h5_name

    copy2(source_h5_path, target_h5_path)

    if include_info:
        source_info_path = source_h5_path.with_name(f'{source_h5_path.stem}_info.txt')
        assert source_info_path.is_file()
        target_info_name = f'gt_{cube_id_str}_info.txt'
        target_info_path = target_h5_root / target_info_name
        copy2(source_info_path, target_info_path)
