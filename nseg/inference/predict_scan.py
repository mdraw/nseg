# Based on https://github.com/funkelab/lsd_nm_experiments/blob/b3af5ddcfa6d6b7f1f266a1326ed309b49decced/02_train/zebrafinch/mtlsd/predict_scan.py


from __future__ import print_function


import json
import logging
import numpy as np
import os
import sys
import zarr
import gunpowder as gp

from nseg.gp_predict import Predict



# voxels
input_shape = gp.Coordinate([84, 268, 268])
output_shape = gp.Coordinate([84, 268, 268])
# output_shape = gp.Coordinate([44, 228, 228])

# nm
voxel_size = gp.Coordinate((20, 9, 9))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


from torch import nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aff = nn.Conv3d(1, 3, 1)
        self.lsd = nn.Conv3d(1, 10, 1)

    def forward(self, input):
        out = self.lsd(input), self.aff(input)
        return out


def predict(checkpoint_path, raw_file, raw_dataset, out_file, out_datasets):
    raw = gp.ArrayKey("RAW")
    affs = gp.ArrayKey("AFFS")
    lsds = gp.ArrayKey("LSDS")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(affs, output_size)
    scan_request.add(lsds, output_size)

    context = (input_size - output_size) / 2

    source = gp.ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
        },
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")

    for ds_name, data in out_datasets.items():

        ds = f.create_dataset(
            ds_name,
            shape=[data["out_dims"]] + list(total_output_roi.get_shape() / voxel_size),
            dtype=data["out_dtype"],
        )

        ds.attrs["resolution"] = voxel_size
        ds.attrs["offset"] = total_output_roi.get_offset()

    pipeline = source

    pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)

    pipeline += gp.Stack(1)

    # pipeline += gp.IntensityScaleShift(raw, 2, -1)

    model = DummyModel()
    checkpoint_path = None

    pipeline += Predict(
        model=model,
        checkpoint=checkpoint_path,
        inputs={
            'input': raw
        },
        outputs={
            0: lsds,
            1: affs,
            # 2: pred_hardness
        }
    )

    pipeline += gp.IntensityScaleShift(affs, 255, 0)
    pipeline += gp.IntensityScaleShift(lsds, 255, 0)

    pipeline += gp.ZarrWrite(
        dataset_names={
            affs: "affs",
            lsds: "lsds",
        },
        output_filename=out_file,
    )

    pipeline += gp.Scan(scan_request)

    predict_request = gp.BatchRequest()

    # predict_request.add(raw, total_input_roi.get_shape())
    # predict_request.add(affs, total_output_roi.get_shape())
    # predict_request.add(lsds, total_output_roi.get_shape())

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    checkpoint_path = None

    raw_file = '/cajal/scratch/projects/misc/mdraw/data/fullraw_fromknossos_j0126/j0126.zarr'
    raw_dataset = "volumes/raw"
    out_file = "/cajal/scratch/projects/misc/mdraw/lsd-results/inference/test2_prediction.zarr"

    out_datasets = {
        'affs': {"out_dims": 3, "out_dtype": "uint8"},
        'lsds': {"out_dims": 10, "out_dtype": "uint8"},
    }

    predict(checkpoint_path, raw_file, raw_dataset, out_file, out_datasets)
