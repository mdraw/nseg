import os
import numpy as np
import zarr
from pathlib import Path

# zarr_root = Path('~/lsdex/data/zebrafinch_msplit/').expanduser()
zarr_root = Path('/home/m4/data/zebrafinch_msplit_z/')


# samples = [glob.glob(os.path.join(f"funke/{v}/training", "*.zarr")) for v in volumes]
zarr_paths = list(zarr_root.rglob('*.zarr'))
assert len(zarr_paths) > 0


for zarr_path in zarr_paths:
    ds = zarr.DirectoryStore(zarr_path)
    zip_path = zarr_path.with_suffix('.zarr.zip')
    zs = zarr.ZipStore(zip_path, mode='w')
    zarr.copy_store(ds, zs)
