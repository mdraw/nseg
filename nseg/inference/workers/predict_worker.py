# Based on https://github.com/funkelab/lsd_nm_experiments/blob/b3af5ddcfa6d6b7f1f266a1326ed309b49decced/02_train/zebrafinch/mtlsd/predict.py

from __future__ import print_function
import json
import logging
import numpy as np
import os
import sys
import pymongo

import torch


import gunpowder as gp

from nseg.gp_predict import Predict
from nseg.gp_boundaries import ArgMax

setup_dir = os.path.dirname(os.path.realpath(__file__))


DUMMY = False

from torch import nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aff = nn.Conv3d(1, 3, 1)
        self.lsd = nn.Conv3d(1, 10, 1)
        self.boundaries = nn.Conv3d(1, 2, 1)
        self.hardness = nn.Conv3d(1, 1, 1)

    def forward(self, input):
        out = self.lsd(input), self.aff(input), self.boundaries(input), self.hardness(input)
        return out



def block_done_callback(db_host, db_name, worker_config, block, start, duration):
    print("Recording block-done for %s" % (block,))

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection = db["blocks_predicted"]

    document = dict(worker_config)
    document.update(
        {
            "block_id": block.block_id,
            "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
            "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
            "start": start,
            "duration": duration,
        }
    )

    collection.insert(document)

    print("Recorded block-done for %s" % (block,))


def predict(
    model_path,
    raw_file,
    raw_dataset,
    out_file,
    db_host,
    db_name,
    worker_config,
    input_size,
    output_size,
):
    raw = gp.ArrayKey("RAW")
    lsds = gp.ArrayKey("LSDS")
    affs = gp.ArrayKey("AFFS")
    boundaries = gp.ArrayKey('BOUNDARIES')
    hardness = gp.ArrayKey('HARDNESS')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, output_size)
    chunk_request.add(affs, output_size)
    chunk_request.add(boundaries, output_size)
    chunk_request.add(hardness, output_size)

    pipeline = gp.ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
        },
    )

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Pad(raw, size=None)

    pipeline += gp.Normalize(raw)

    # pipeline += gp.Stack(1)
    # pipeline += gp.IntensityScaleShift(raw, 2, -1)

    if model_path.lower() == 'dummy':
        model = DummyModel().eval()
    else:
        model = torch.load(model_path)

    # TODO: Clear up model_path vs. checkpoint path use - support both? See segment_mtlsd.py
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
            2: boundaries,
            3: hardness,
        }
    )


    # TODO: Output channel 1 probmap instead?
    pipeline += ArgMax(boundaries)

    pipeline += gp.Squeeze([
        raw,
        lsds,
        affs,
        boundaries,
        hardness,
    ])

    pipeline += gp.Squeeze([
        raw,
    #     lsds,
    #     affs,
        boundaries,
    #     hardness,
    ])

    # pipeline += gp.IntensityScaleShift(raw, 255, 0)
    pipeline += gp.IntensityScaleShift(affs, 255, 0)
    pipeline += gp.IntensityScaleShift(lsds, 255, 0)
    pipeline += gp.IntensityScaleShift(boundaries, 255, 0)


    pipeline += gp.ZarrWrite(
        dataset_names={
            # raw: 'raw',
            affs: 'volumes/affs',
            lsds: 'volumes/lsds',
            boundaries: 'volumes/boundaries',
            hardness: 'volumes/hardness',
        },
        output_filename=out_file,
    )
    pipeline += gp.PrintProfilingStats(every=10)

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: "read_roi",
            affs: "write_roi",
            lsds: "write_roi",
            boundaries: "write_roi",
            hardness: "write_roi",
        },
        num_workers=worker_config["num_cache_workers"],
        block_done_callback=lambda b, s, d: block_done_callback(
            db_host, db_name, worker_config, b, s, d
        ),
    )

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        run_config = json.load(f)

    predict(**run_config)
