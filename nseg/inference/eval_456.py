import logging
import time

from nseg.conf import NConf, DictConfig, hydra, unwind_dict

from nseg.inference import (
    i4_find_segments,
    i6_extract_segmentation,
    i5_evaluate_annotations
)


@hydra.main(version_base='1.3', config_path='../conf/inference', config_name='inference_config')
def main(cfg: DictConfig) -> None:

    start = time.time()

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    logging.info(f'Hydra run dir: {_hydra_run_dir}')
    logging.info(f'Config: {dict_cfg}')

    i4_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i4_find_segments'])
    i4_find_segments.find_segments(**i4_dict_cfg)

    i5_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_extract_segmentation'])
    i6_extract_segmentation.extract_segmentation(**i5_dict_cfg)

    i6_dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i5_evaluate_annotations'])
    evaluate = i5_evaluate_annotations.EvaluateAnnotations(**i6_dict_cfg)
    evaluate.evaluate()

    seconds = time.time() - start
    hours = seconds / 3600
    logging.info(f'Total time: {hours} h')


if __name__ == "__main__":
    main()
