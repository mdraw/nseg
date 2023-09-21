import datetime
import daisy
import hashlib
import json
import logging
import lsd
import numpy as np
import os
import pymongo
import sys
import time

from pathlib import Path

from nseg.conf import NConf, DictConfig, hydra, unwind_dict


logging.basicConfig(level=logging.INFO)

def agglomerate(
        experiment,
        setup,
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        block_size,
        context,
        db_host,
        db_name,
        num_workers,
        merge_function,
        pybin,
        slurm_options,
        _hydra_run_dir,
        **_  # Gobble all other kwargs
):

    '''

    Agglomerate in parallel blocks. Requires that affinities and supervoxels
    have been generated.

    Args:

        * following three params just used to build out file directory *

        experiment (``string``):

            Name of the experiment (fib25, hemi, zfinch, ...).

        setup (``string``):

            Name of the setup to predict (setup01, setup02, ...).

        affs_file (``string``):

            Path to file (zarr/n5) where predictions are stored.

        affs_dataset (``string``):

            Predictions dataset to use (e.g 'volumes/affs').

        fragments_file (``string``):

            Path to file (zarr/n5) where fragments (supervoxels) are stored.

        fragments_dataset (``string``):

            Name of fragments (supervoxels) dataset (e.g 'volumes/fragments').

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction in world units.

        db_host (``string``):

            Name of MongoDB client.

        db_name (``string``):

            Name of MongoDB database to use (for logging successful blocks in
            check function and reading nodes from + writing edges to the region
            adjacency graph).

        num_workers (``int``):

            How many blocks to run in parallel.

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary in worker script
            (workers/agglomerate_worker.py).

    '''

    initial_timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


    logging.info(f"Reading affs from {affs_file}")
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    blocks_agglomerated = 'blocks_agglomerated_' + merge_function

    if blocks_agglomerated not in db.list_collection_names():
        blocks_agglomerated = db[blocks_agglomerated]
        blocks_agglomerated.create_index(
                [('block_id', pymongo.ASCENDING)],
                name='block_id')
    else:
        blocks_agglomerated = db[blocks_agglomerated]

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            db_host=db_host,
            db_name=db_name,
            merge_function=merge_function,
            initial_timestamp=initial_timestamp,
            pybin=pybin,
            slurm_options=slurm_options,
            _hydra_run_dir=_hydra_run_dir,
        ),
        check_function=lambda b: check_block(
            blocks_agglomerated,
            b
        ),
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink'
    )

def start_worker(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        db_name,
        merge_function,
        initial_timestamp,
        pybin,
        slurm_options,
        _hydra_run_dir,
):

    worker_id = daisy.Context.from_env().worker_id

    logging.info(f"worker {worker_id} started...")

    # TODO: Rename 'agglomerate_blockwise' to 'i3_agglomerate_blockwise' when experiments are done
    output_dir = os.path.join(_hydra_run_dir, 'agglomerate_blockwise', merge_function, initial_timestamp)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    log_out = os.path.join(output_dir, f'{timestamp}_agglomerate_blockwise_{worker_id}.out')
    log_err = os.path.join(output_dir, f'{timestamp}_agglomerate_blockwise_{worker_id}.err')

    config = {
            'affs_file': affs_file,
            'affs_dataset': affs_dataset,
            'fragments_file': fragments_file,
            'fragments_dataset': fragments_dataset,
            'db_host': db_host,
            'db_name': db_name,
            'merge_function': merge_function
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker_script = Path(__file__).parent / 'workers' / 'agglomerate_worker.py'

    _pyb_log_out = os.path.join(output_dir, f'{timestamp}_log_{worker_id}')



    base_command = [
        'srun',
        *slurm_options,
        '-o', f'{log_out}',
        f'{pybin} {worker_script} {config_file} &> {_pyb_log_out}'
    ]

    logging.info(f'Base command: {base_command}')

    daisy.call(base_command, log_out=log_out, log_err=log_err)


def check_block(blocks_agglomerated, block):
    # Optimized query that does not require a full scan of the collection
    done = blocks_agglomerated.find_one({'block_id': block.block_id}) is not None
    # assert done == blocks_agglomerated.count({'block_id': block.block_id}) >= 1  # Check against original query
    return done


@hydra.main(version_base='1.3', config_path='../conf/', config_name='inference_config')
def main(cfg: DictConfig) -> None:

    start = time.time()

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i3_agglomerate'])

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    dict_cfg['_hydra_run_dir'] = _hydra_run_dir
    logging.info(f'Hydra run dir: {_hydra_run_dir}')
    dict_cfg['_hydra_run_dir'] = _hydra_run_dir
    logging.info(f'Config: {dict_cfg}')

    agglomerate(**dict_cfg)

    end = time.time()

    seconds = end - start
    logging.info(f'Total time to agglomerate: {seconds}')


if __name__ == "__main__":
    main()
