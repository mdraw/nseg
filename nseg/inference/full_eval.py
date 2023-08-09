from pathlib import Path
from typing import Any, Sequence
from datetime import timedelta
import json
import gc
import logging
import os
import sys
import time
import submitit

from nseg.conf import NConf, DictConfig, hydra, unwind_dict

from nseg.inference import (
    i1_predict,
    i2_extract_fragments,
    i3_agglomerate,
    i4_find_segments,
    i6_extract_segmentation,
    i5_evaluate_annotations
)


def get_logging_kwargs(log_file_path: Path) -> dict:
    return dict(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, mode='w'),
        ],
        force=True,
    )
    # logging.info(f'Task logs are written to: {log_file_path}')


def run_i456(dict_cfg: dict, hydra_run_dir) -> None:
    jobs_to_run = dict_cfg['meta']['jobs_to_run']
    logging.info(f'\nRunning jobs {jobs_to_run}\n')
    assert jobs_to_run and set(jobs_to_run).issubset({'i1', 'i2', 'i3', 'i4', 'i5', 'i6'})

    if 'i4' in jobs_to_run:
        gc.collect(); gc.collect(); gc.collect()  # Avoid OOM caused by lazy GC
        t0 = time.time()
        logging.basicConfig(**get_logging_kwargs(hydra_run_dir / 'i4_find_segments.log'))
        logging.info('Running i4_find_segments')
        i4_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i4_find_segments'])
        i4_find_segments.find_segments(**i4_dict_cfg)
        logging.info(f'i4_find_segments took {timedelta(seconds=time.time() - t0)}')

    if 'i5' in jobs_to_run:
        gc.collect(); gc.collect(); gc.collect()  # Avoid OOM caused by lazy GC
        t0 = time.time()
        logging.basicConfig(**get_logging_kwargs(hydra_run_dir / 'i5_evaluate_annotations.log'))
        logging.info('Running i5_evaluate_annotations')
        i5_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i5_evaluate_annotations'])
        evaluate = i5_evaluate_annotations.EvaluateAnnotations(**i5_dict_cfg)
        evaluate.evaluate()
        logging.info(f'i5_evaluate_annotations took {timedelta(seconds=time.time() - t0)}')

    if 'i6' in jobs_to_run:
        gc.collect(); gc.collect(); gc.collect()  # Avoid OOM caused by lazy GC
        t0 = time.time()
        logging.basicConfig(**get_logging_kwargs(hydra_run_dir / 'i6_extract_segmentation.log'))
        logging.info('Running i6_extract_segmentation')
        i6_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_extract_segmentation'])
        i6_extract_segmentation.extract_segmentation(**i6_dict_cfg)
        logging.info(f'i6_extract_segmentation took {timedelta(seconds=time.time() - t0)}')


@hydra.main(version_base='1.3', config_path='../conf/', config_name='inference_config')
def main(cfg: DictConfig) -> None:
    start = time.time()

    jobs_to_run = list(cfg.meta.jobs_to_run)

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    hydra_run_dir = Path(hydra.core.hydra_config.HydraConfig.get()['run']['dir'])
    dict_cfg['_hydra_run_dir'] = hydra_run_dir
    logging.info(f'Hydra run dir: {hydra_run_dir}')
    logging.info(f'Config: {dict_cfg}')

    # i1..i3 already distribute jobs to cluster nodes on their own so they should be run on a login node.

    if 'i1' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i1_predict')
        i1_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i1_predict'])
        i1_predict.predict_blockwise(**i1_dict_cfg)
        logging.info(f'i1_predict took {timedelta(seconds=time.time() - t0)}')

    if 'i2' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i2_extract_fragments')
        i2_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i2_extract_fragments'])
        i2_extract_fragments.extract_fragments(**i2_dict_cfg)
        logging.info(f'i2_extract_fragments took {timedelta(seconds=time.time() - t0)}')

    if 'i3' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i3_agglomerate')
        i3_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i3_agglomerate'])
        i3_agglomerate.agglomerate(**i3_dict_cfg)
        logging.info(f'i3_agglomerate took {timedelta(seconds=time.time() - t0)}')

    # i4..i6 are run locally, so they should be started on a dedicated compute node.

    if cfg.meta.run_i456_locally:
        run_i456(dict_cfg)
    else:
        executor = submitit.AutoExecutor(folder=hydra_run_dir / 'submitit')
        submitit_opts = NConf.to_container(cfg.meta.submitit_i456_options, resolve=True)
        executor.update_parameters(stderr_to_stdout=True, **submitit_opts)
        job = executor.submit(run_i456, dict_cfg, hydra_run_dir=hydra_run_dir)
        logging.info(f'Submitted tasks i456 to separate compute node. Job ID: {job.job_id}')
        logging.info(f'Terminal outputs are delayed until the job is finished. See real-time logs in {hydra_run_dir}/')
        job.wait()
        logging.info('Completed i456')
        logging.info(f'i456 stderr:\n\n{job.stderr()}')
        logging.info(f'i456 stdout:\n\n{job.stdout()}')

    seconds = (time.time() - start)
    logging.info(f'Finished all tasks ({jobs_to_run}) in {timedelta(seconds=seconds)}')
    logging.info(f'Results can be found under {cfg.output_dir}/')


if __name__ == "__main__":
    main()
