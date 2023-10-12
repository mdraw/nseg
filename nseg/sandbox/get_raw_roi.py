"""Get raw data for a given ROI and save as zarr."""

import numpy as np
import zarr
import daisy

# roi = '11_micron'
roi = 'benchmark'

zraw = daisy.open_ds(f'./data/zf_{roi}_roi.json', 'volumes/raw', mode='r')
output_path = f'/cajal/scratch/projects/misc/mdraw/data/raw_j0126_{roi}.zarr'

print('Loading')
raw = zraw.to_ndarray()
print('Creating', output_path)
outzarr = zarr.open(output_path, mode='w')
print('Writing')
outzarr.create_dataset('volumes/raw', data=raw, chunks=(128, 128, 128), dtype=raw.dtype)
