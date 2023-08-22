# Based on https://github.com/funkelab/lsd_nm_experiments/blob/b3af5ddcfa6d6b7f1f266a1326ed309b49decced/02_train/zebrafinch/mtlsd/predict.py

from __future__ import print_function
import json
import logging
import os
import sys
import pymongo
import psutil

import torch


import gunpowder as gp

from nseg.gpx.gp_predict import Predict
from nseg.gpx.gp_boundaries import SoftMax, Take

setup_dir = os.path.dirname(os.path.realpath(__file__))


# TODO: Use logging instead of print

torch.backends.cudnn.benchmark = True

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
    output_cfg,
):
    output_names = list(output_cfg.keys())

    raw = gp.ArrayKey("RAW")
    output_arrkeys = {k: gp.ArrayKey(k.upper()) for k in output_names}

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)

    for arrkey in output_arrkeys.values():
        chunk_request.add(arrkey, output_size)

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

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    if model_path.lower() == 'dummy':
        model = DummyModel().eval()
    else:
        model = torch.load(model_path).eval()

    # TODO: Clear up model_path vs. checkpoint path use - support both? See segment_mtlsd.py
    checkpoint_path = None

    # _predict_outputs = {output_cfg[n]['idx']: output_arrkeys[n] for n in output_names}
    _predict_outputs = {n: output_arrkeys[n] for n in output_names}

    pipeline += Predict(
        model=model,
        checkpoint=checkpoint_path,
        inputs={
            'input': raw
        },
        outputs=_predict_outputs,
    )

    if 'pred_boundaries' in output_names:
        boundary_arrkey = output_arrkeys['pred_boundaries']
        # pipeline += ArgMax(boundaries)
        pipeline += SoftMax(boundary_arrkey)
        pipeline += Take(boundary_arrkey, 1, 1)  # Take channel 1

    outputs_to_squeeze = [
        output_arrkeys[k]
        for k, v in output_cfg.items()
        if v['squeeze']
    ]

    pipeline += gp.Squeeze([
        raw,
        *outputs_to_squeeze
    ])

    pipeline += gp.Squeeze([raw])

    for k, v in output_cfg.items():
        if v.get('scale', 1) != 1:
            pipeline += gp.IntensityScaleShift(output_arrkeys[k], v['scale'], 0)

    out_dataset_names = {
        v: f'volumes/{k}' for k, v in output_arrkeys.items()
    }
    # out_dataset_names[raw] = 'volumes/raw'

    pipeline += gp.ZarrWrite(
        dataset_names=out_dataset_names,
        output_filename=out_file,
    )
    # pipeline += gp.PrintProfilingStats(every=10)

    roi_map = {output_arrkeys[k]: "write_roi" for k in output_names}
    roi_map[raw] = "read_roi"

    pipeline += gp.DaisyRequestBlocks(
        chunk_request,
        roi_map=roi_map,
        num_workers=worker_config["num_cache_workers"],
        block_done_callback=lambda b, s, d: block_done_callback(
            db_host, db_name, worker_config, b, s, d
        ),
    )

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    print("Prediction finished")
    print(f'Prediction finished.')
    print(f'Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB')
    print(f'Peak RAM usage: {psutil.Process().memory_info().rss / 1024**3:.2f} GiB')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        run_config = json.load(f)

    predict(**run_config)
