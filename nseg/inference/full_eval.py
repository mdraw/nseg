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
    i5_evaluate_annotations,
    i6_extract_segmentation,
    iutils,
)


def get_logging_kwargs(log_file_path: Path) -> dict:
    return dict(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, mode='a'),
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

    # Collect best thresholds w.r.t. voi and erl on val and test
    best_thresh_results = {'voi': {}, 'erl': {}}
    thresh_results = {'voi': {}, 'erl': {}}


    if 'i5' in jobs_to_run:
        anno_names = dict_cfg['meta']['evaluate_on']  # ['val', 'test']
        for anno_name in anno_names:
            gc.collect(); gc.collect(); gc.collect()  # Avoid OOM caused by lazy GC
            t0 = time.time()
            i5_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i5_evaluate_annotations'])

            # Set annotations_db and scores_db to val or test, depending on anno_name
            annotations_db_name = i5_dict_cfg['annotations_db_names'][anno_name]
            scores_db_name = i5_dict_cfg['scores_db_names'][anno_name]
            i5_dict_cfg['annotations_db_name'] = annotations_db_name
            i5_dict_cfg['scores_db_name'] = scores_db_name

            logging.basicConfig(**get_logging_kwargs(hydra_run_dir / f'i5_evaluate_annotations_{anno_name}.log'))
            logging.info(f'Running i5_evaluate_annotations ({anno_name}')
            evaluate = i5_evaluate_annotations.EvaluateAnnotations(**i5_dict_cfg)
            evaluate.evaluate()

            best_thresh_results['voi'][anno_name] = evaluate._best_voi_threshold
            best_thresh_results['erl'][anno_name] = evaluate._best_erl_threshold
            thresh_results['voi'][anno_name] = evaluate._thresh_vois
            thresh_results['erl'][anno_name] = evaluate._thresh_erls

            logging.info(f'i5_evaluate_annotations ({anno_name}) took {timedelta(seconds=time.time() - t0)}')

        logging.info(f'Best threshold configurations:\n{json.dumps(best_thresh_results, indent=4)}\n')
        if 'val' in anno_names and 'test' in anno_names:
            test_voi_on_best_val_threshold = thresh_results['voi']['test'][
                best_thresh_results['voi']['val']
            ]
            test_erl_on_best_val_threshold = thresh_results['erl']['test'][
                best_thresh_results['erl']['val']
            ]

            best_thresh_results['voi']['test_best_val_threshold'] = test_voi_on_best_val_threshold
            best_thresh_results['erl']['test_best_val_threshold'] = test_erl_on_best_val_threshold

            logging.info(f'Test VOI on best val threshold: {test_voi_on_best_val_threshold}')
            logging.info(f'Test ERL on best val threshold: {test_erl_on_best_val_threshold}')

            logging.info(f'Storing best threshold results in db')
            iutils.store_document(
                doc=best_thresh_results,
                collection_name=dict_cfg['common']['setup'],
                db_name='best_thresh_results',
                db_host=dict_cfg['common']['db_host'],
            )

            # Convert all keys of nested dict to strings for db compat
            thresh_results_doc = json.loads(json.dumps(thresh_results))

            iutils.store_document(
                doc=thresh_results_doc,
                collection_name=dict_cfg['common']['setup'],
                db_name='thresh_results',
                db_host=dict_cfg['common']['db_host'],
            )

    if 'i6' in jobs_to_run:
        gc.collect(); gc.collect(); gc.collect()  # Avoid OOM caused by lazy GC
        t0 = time.time()
        logging.basicConfig(**get_logging_kwargs(hydra_run_dir / 'i6_extract_segmentation.log'))
        logging.info('Running i6_extract_segmentation')
        i6_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_extract_segmentation'])

        # TODO: Get best thresholds from database
        # TODO: Fail gracefully if best thresholds are not collected above
        if i6_dict_cfg['threshold'] == 'best_voi':
            i6_dict_cfg['threshold'] = best_thresholds['voi']['val']
        elif i6_dict_cfg['threshold'] == 'best_erl':
            i6_dict_cfg['threshold'] = best_thresholds['erl']['val']

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
        run_i456(dict_cfg, hydra_run_dir=hydra_run_dir)
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
