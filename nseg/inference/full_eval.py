from pathlib import Path
from typing import Any, Sequence
import datetime
import json
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
    i5_extract_segmentation_from_lut,
    i6_evaluate_annotations
)


def run_i456(dict_cfg: dict, log_file=None) -> None:
    jobs_to_run = dict_cfg['meta']['jobs_to_run']

    if log_file is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                # logging.StreamHandler(sys.stderr),
                logging.FileHandler(log_file, mode='a'),
            ]
        )
    logging.info(f'\nRunning jobs {jobs_to_run}\n')
    assert jobs_to_run and set(jobs_to_run).issubset({'i1', 'i2', 'i3', 'i4', 'i5', 'i6'})

    if 'i4' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i4_find_segments')
        i4_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i4_find_segments'])
        i4_find_segments.find_segments(**i4_dict_cfg)
        logging.info(f'i4_find_segments took {(time.time() - t0) / 3600:.2f} h')

    if 'i5' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i5_extract_segmentation_from_lut')
        i5_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i5_extract_segmentation'])
        i5_extract_segmentation_from_lut.extract_segmentation(**i5_dict_cfg)
        logging.info(f'i5_extract_segmentation_from_lut took {(time.time() - t0) / 3600:.2f} h')

    if 'i6' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i6_evaluate_annotations')
        i6_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_evaluate_annotations'])
        evaluate = i6_evaluate_annotations.EvaluateAnnotations(**i6_dict_cfg)
        evaluate.evaluate()
        logging.info(f'i6_evaluate_annotations took {(time.time() - t0) / 3600:.2f} h')


@hydra.main(version_base='1.3', config_path='../conf/inference', config_name='inference_config')
def main(cfg: DictConfig) -> None:
    start = time.time()

    jobs_to_run = list(cfg.meta.jobs_to_run)

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    dict_cfg['_hydra_run_dir'] = _hydra_run_dir
    logging.info(f'Hydra run dir: {_hydra_run_dir}')
    logging.info(f'Config: {dict_cfg}')

    # i1..i3 already distribute jobs to cluster nodes on their own so they should be run on a login node.

    if 'i1' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i1_predict')
        i1_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i1_predict'])
        i1_predict.predict_blockwise(**i1_dict_cfg)
        logging.info(f'i1_predict took {(time.time() - t0) / 3600:.2f} h')

    if 'i2' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i2_extract_fragments')
        i2_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i2_extract_fragments'])
        i2_extract_fragments.extract_fragments(**i2_dict_cfg)
        logging.info(f'i2_extract_fragments took {(time.time() - t0) / 3600:.2f} h')

    if 'i3' in jobs_to_run:
        t0 = time.time()
        logging.info('Running i3_agglomerate')
        i3_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i3_agglomerate'])
        i3_agglomerate.agglomerate(**i3_dict_cfg)
        logging.info(f'i3_agglomerate took {(time.time() - t0) / 3600:.2f} h')

    # i4..i6 are run locally, so they should be started on a dedicated compute node.

    if cfg.meta.run_i456_locally:
        run_i456(dict_cfg)
    else:
        _i456_log_file = f'{_hydra_run_dir}/i456.log'
        executor = submitit.AutoExecutor(folder=Path(_hydra_run_dir) / 'submitit')
        submitit_opts = NConf.to_container(cfg.meta.submitit_i456_options, resolve=True)
        executor.update_parameters(stderr_to_stdout=True, **submitit_opts)
        job = executor.submit(run_i456, dict_cfg, log_file=_i456_log_file)
        logging.info(f'Submitted tasks i456 to separate compute node. Job ID: {job.job_id}')
        logging.info(f'Task logs are written to: {_i456_log_file}')
        logging.info(f'(Terminal outputs are delayed until the job is finished.)')
        job.wait()
        logging.info('Completed i456')
        logging.info(f'i456 stderr:\n\n{job.stderr()}')
        logging.info(f'i456 stdout:\n\n{job.stdout()}')

    # if 'i4' in jobs_to_run:
    #     logging.info('Running i4_find_segments')
    #     i4_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i4_find_segments'])
    #     i4_find_segments.find_segments(**i4_dict_cfg)
    #
    # if 'i5' in jobs_to_run:
    #     logging.info('Running i5_extract_segmentation_from_lut')
    #     i5_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i5_extract_segmentation_from_lut'])
    #     i5_extract_segmentation_from_lut.extract_segmentation(**i5_dict_cfg)
    #
    # if 'i6' in jobs_to_run:
    #     logging.info('Running i6_evaluate_annotations')
    #     i6_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_evaluate_annotations'])
    #     evaluate = i6_evaluate_annotations.EvaluateAnnotations(**i6_dict_cfg)
    #     evaluate.evaluate()

    hours = (time.time() - start) / 3600
    logging.info(f'Total time: {hours:.2f} h')


if __name__ == "__main__":
    main()
