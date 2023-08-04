# Based on https://github.com/funkelab/lsd/blob/b6aee2fd0c87bc70a52ea77e85f24cc48bc4f437/lsd/tutorial/scripts/01_predict_blockwise.py

from pathlib import Path
from typing import Any
import daisy
import datetime
import hashlib
import json
import logging
import numpy as np
import os
import pymongo
import sys
import time

from nseg.conf import NConf, DictConfig, hydra, unwind_dict


# logging.basicConfig(level=logging.INFO)

def predict_blockwise(
        experiment,
        setup,
        model_path,
        raw_file,
        raw_dataset,
        out_file,
        num_workers,
        db_host,
        db_name,
        output_cfg,
        net_input_shape,
        net_offset,
        voxel_size,
        pybin,
        slurm_options,
        _hydra_run_dir,
        **_  # Gobble all other kwargs
):

    '''

    Run prediction in parallel blocks. Within blocks, predict in chunks.

    Args:

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        raw_file (``string``):

            Path to raw file (zarr/n5) - can also be a json container
            specifying a crop, where offset and size are in world units:

                {
                    "container": "path/to/raw",
                    "offset": [z, y, x],
                    "size": [z, y, x]
                }

        raw_dataset (``string``):

            Raw dataset to use (e.g 'volumes/raw'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/raw/s0')

        out_base (``string``):

            Path to base directory where zarr/n5 should be stored. The out_file
            will be built from this directory, setup, file name

            **Note:

                out_dataset no longer needed as input, build out_dataset from config
                outputs dictionary generated in mknet.py (config.json for
                example)

        file_name (``string``):

            Name of output zarr/n5

        num_workers (``int``):

            How many blocks to run in parallel.

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (for logging successful blocks in
            check function and DaisyRequestBlocks node inside worker predict
            script).

        queue (``string``):

            Name of gpu queue to run inference on (i.e gpu_rtx, gpu_tesla, etc)

    '''

    initial_timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    network_dir = os.path.join(experiment, setup)

    raw_file = os.path.abspath(raw_file)
    # out_file = os.path.abspath(os.path.join(out_base, setup, file_name))

    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)

    logging.info(f'Source shape: {source.shape}')
    logging.info(f'Source roi: {source.roi}')
    logging.info(f'Source voxel size: {source.voxel_size}')

    # voxels
    net_input_shape = daisy.Coordinate(net_input_shape)
    net_offset = daisy.Coordinate(net_offset)
    net_output_shape = net_input_shape - net_offset

    # nm
    voxel_size = daisy.Coordinate(voxel_size)
    net_input_size = net_input_shape * voxel_size
    net_output_size = net_output_shape * voxel_size

    context = (net_input_size - net_output_size)/2

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    logging.info('Preparing output dataset...')


    # get output file(s) meta data from config.json, prepare dataset(s)
    for output_name, val in output_cfg.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        out_dataset = f'volumes/{output_name}'
        print(out_dataset)

        ds = daisy.prepare_ds(
            filename=out_file,
            ds_name=out_dataset,
            total_roi=output_roi,
            voxel_size=source.voxel_size,
            dtype=out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'zstd', 'level': 5}
        )

    logging.info('Starting block-wise processing...')

    # for logging successful blocks (see check_block function). if anything
    # fails, blocks which completed will be skipped when re-running

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    if 'blocks_predicted' not in db.list_collection_names():
        blocks_predicted = db['blocks_predicted']
        blocks_predicted.create_index(
            keys=[('block_id', pymongo.ASCENDING)],
            name='block_id'
        )
    else:
        blocks_predicted = db['blocks_predicted']

    # process block-wise
    succeeded = daisy.run_blockwise(
        total_roi=input_roi,
        read_roi=block_read_roi,
        write_roi=block_write_roi,
        process_function=lambda: predict_worker(
            network_dir=network_dir,
            model_path=model_path,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            db_host=db_host,
            db_name=db_name,
            pybin=pybin,
            net_input_size=net_input_size,
            net_output_size=net_output_size,
            initial_timestamp=initial_timestamp,
            slurm_options=slurm_options,
            output_cfg=output_cfg,
            _hydra_run_dir=_hydra_run_dir,
        ),
        check_function=lambda b: check_block(blocks_predicted, b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='overhang'
    )

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

def predict_worker(
        network_dir,
        model_path,
        raw_file,
        raw_dataset,
        out_file,
        db_host,
        db_name,
        pybin,
        net_input_size,
        net_output_size,
        initial_timestamp,
        slurm_options,
        output_cfg,
        _hydra_run_dir,
):

    timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    # get the relevant worker script to distribute
    worker_script = Path(__file__).parent / 'workers' / 'predict_worker.py'

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']

    worker_config = {
        'num_cpus': 4,
        'num_cache_workers': 0
    }

    config = {
        'model_path': model_path,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'out_file': out_file,
        'db_host': db_host,
        'db_name': db_name,
        'worker_config': worker_config,
        'input_size': net_input_size,
        'output_size': net_output_size,
        'output_cfg': output_cfg,
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    # get worker id
    worker_id = daisy.Context.from_env().worker_id

    output_dir = os.path.join(_hydra_run_dir, 'predict_blockwise', network_dir, initial_timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # pipe output
    config_file = os.path.join(output_dir, '%d.config'%config_hash)


    log_out = os.path.join(output_dir, f'{timestamp}_predict_blockwise_{worker_id}.out')
    log_err = os.path.join(output_dir, f'{timestamp}_predict_blockwise_{worker_id}.err')
    _pyb_log_out = os.path.join(output_dir, f'{timestamp}_log_{worker_id}')
    _srun_log_out = os.path.join(output_dir, f'{timestamp}_srun_log_{worker_id}')

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    # create worker command
    command = [
        'srun',
        *slurm_options,
        '-o', f'{_srun_log_out}',
    ]

    command += [
        f'{pybin} -u {worker_script} {config_file} &> {_pyb_log_out}'
    ]

    logging.info(f'Worker command: {command}')

    # call command
    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.info('Predict worker finished')

def check_block(blocks_predicted, block):

    done = blocks_predicted.count({'block_id': block.block_id}) >= 1

    return done

@hydra.main(version_base='1.3', config_path='../conf/inference', config_name='inference_config')
def main(cfg: DictConfig) -> None:

    start = time.time()

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i1_predict'])

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    dict_cfg['_hydra_run_dir'] = _hydra_run_dir
    logging.info(f'Hydra run dir: {_hydra_run_dir}')
    logging.info(f'Config: {dict_cfg}')

    predict_blockwise(**dict_cfg)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to predict: {seconds}')


if __name__ == "__main__":
    main()
