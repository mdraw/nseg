# Based on https://github.com/funkelab/lsd_nm_experiments/blob/b3af5ddcfa6d6b7f1f266a1326ed309b49decced/01_data/create_masks.py

# Just for zebrafinch

import os
import numpy as np
import zarr
from pathlib import Path



zarr_root = Path('~/lsdex/data/zebrafinch_msplit/').expanduser()

# samples = [glob.glob(os.path.join(f"funke/{v}/training", "*.zarr")) for v in volumes]
samples = list(zarr_root.rglob('*.zarr'))
assert len(samples) > 0

labels_name = 'volumes/labels/neuron_ids'
labels_mask_name = 'volumes/labels/labels_mask'

for sample in samples:
    f = zarr.open(sample, 'a')

    labels = f[labels_name][:]

    labels_mask = np.ones_like(labels).astype(np.uint8)

    f[labels_mask_name] = labels_mask
    f[labels_mask_name].attrs['offset'] = f[labels_name].attrs['offset']
    f[labels_mask_name].attrs['resolution'] = f[labels_name].attrs['resolution']
