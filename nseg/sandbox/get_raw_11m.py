import numpy as np
import zarr
import daisy



zraw = daisy.open_ds('./data/zf_11_micron_roi.json', 'volumes/raw', mode='r')

raw = zraw.to_ndarray()

output_path = '/cajal/scratch/projects/misc/mdraw/lsdex/v1/data/raw_j0126_11_micron.zarr'

outzarr = zarr.open(output_path, mode='w')
outzarr.create_dataset('volumes/raw', data=raw, chunks=(256, 256, 256), dtype=raw.dtype)
