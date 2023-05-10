# Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_mtlsd.ipynb

import os
import logging
import time
from pathlib import Path
from typing import Any, Callable
import skimage

import matplotlib
matplotlib.use('AGG')

import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np
import torch
from lsd.train.gp import AddLocalShapeDescriptor
from torch.utils import tensorboard
from tqdm import tqdm

import wandb

# from params import input_size, output_size, voxel_size
from nseg.segment_mtlsd import center_crop, eval_cubes, get_mean_report, get_per_cube_metrics, spatial_center_crop_nd
from nseg.shared import create_lut, get_mtlsdmodel, build_mtlsdmodel, WeightedMSELoss
from nseg.gp_train import Train
from nseg.gp_sources import ZarrSource
from nseg.conf import NConf, DictConfig, hydra



def _get_include_fn(exts) -> Callable[[list[str]], bool]:
    def include_fn(fn):
        return any(str(fn).endswith(ext) for ext in exts)
    return include_fn


def densify_labels(lab: np.ndarray, dtype=np.uint8) -> np.ndarray:
    old_ids = np.unique(lab)
    num_ids = old_ids.size
    new_ids = range(num_ids)
    dense_lab = np.zeros_like(lab, dtype=dtype)
    for o, n in zip(old_ids, new_ids):
        dense_lab[lab == o] = n
    return dense_lab, num_ids


# TODO: Squeeze singleton C dim?
def get_zslice(vol, z_plane=None, squeeze=True, as_wandb=False, enable_rgb_labels=False):
    """
    Expects C, D, H, W or D, H, W volume and returns a 2D image slice compatible with `wandb.Image()`

    """
    # if vol.ndim == 5:  # N, C, D, H, W
    #     vol = vol[0]  # Strip batch dim -> [C,] D, H, W
    if z_plane is None:
        z_plane = vol.shape[-3] // 2  # Get D // 2
    slc = vol[..., z_plane, :, :]  # -> [C,] H, W
    if squeeze and slc.shape[0] == 1:
        slc = np.squeeze(slc, 0)  #  Strip singleton C dim
    assert slc.ndim in [2, 3]
    if slc.ndim == 3:  # C, H, W
        slc = np.moveaxis(slc, 0, -1)
    # Else: H, W, nothing to do
    if enable_rgb_labels:
        lab = skimage.measure.label(slc)
        rgb = skimage.color.label2rgb(lab, bg_label=0)
        slc = rgb
    if as_wandb:
        return wandb.Image(slc)
    return slc


def prefixkeys(dictionary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Returns a dict with prefixed key names but same values."""
    prefdict = {}
    for key, value in dictionary.items():
        prefdict[f'{prefix}{key}'] = value
    return prefdict


def get_run_time_str(start_time: float) -> str:
    delta_seconds = int(time.time() - start_time)
    m, s = divmod(delta_seconds, 60)
    h, m = divmod(m, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


def train(cfg: DictConfig) -> None:
    _training_start_time = time.time()
    tr_root = Path(cfg.dataset.tr_root)
    tr_files = [str(fp) for fp in tr_root.glob('*.zarr')]

    val_root = cfg.dataset.val_root
    if val_root is None:
        val_files = []
    else:
        val_root = Path(val_root)
        val_files = [str(fp) for fp in val_root.glob('*.zarr')]

    # Get standard Python dict representation of NConf / omegaconf cfg (for wandb cfg logging)
    _cfg_dict = NConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Find the current hydra run dir
    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    # Set general save_path (also for non-hydra-related outputs) to hydra run dir so we have everything in one place
    save_path = Path(_hydra_run_dir)
    logging.info(f'save_path: {save_path}')
    num_workers = cfg.training.num_workers
    if num_workers == 'auto':
        num_workers = len(os.sched_getaffinity(0))  # Get cores available cpu cores for this (SLURM) job
        logging.info(f'num_workers: automatically using all {num_workers} available cpu cores')

    # Silence log spam of "requesting complete mask..." and "allocating mask integral array..." messages
    logging.getLogger('gunpowder.nodes.random_location').setLevel(logging.WARNING)

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    if cfg.dataset.enable_mask:
        labels_mask = gp.ArrayKey('GT_LABELS_MASK')
        gt_affs_mask = gp.ArrayKey('GT_AFFINITIES_MASK')
    else:
        labels_mask = None
        gt_affs_mask = None

    voxel_size = gp.Coordinate(cfg.dataset.voxel_size)
    # Prefer ev_inp_shape if specified, use regular inp_shape otherwise
    input_shape = gp.Coordinate(cfg.model.backbone.inp_shape)
    input_size = input_shape * voxel_size
    offset = gp.Coordinate(cfg.model.backbone.offset)
    output_shape = input_shape - offset
    output_size = output_shape * voxel_size

    # model = get_mtlsdmodel()
    model = build_mtlsdmodel(cfg.model)
    example_input = torch.randn(
        1,  # cfg.training.batch_size,
        1,  # cfg.model.backbone.init_kwargs.in_channels,
        *cfg.model.backbone.inp_shape,
    )

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=cfg.training.lr, params=model.parameters())

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    if cfg.dataset.enable_mask:
        request.add(labels_mask, output_size)
        request.add(gt_affs_mask, output_size)

    source_data_dict = {
        raw: cfg.dataset.raw_name,
        labels: cfg.dataset.gt_name,
    }
    source_array_specs = {
        raw: gp.ArraySpec(interpolatable=True),
        labels: gp.ArraySpec(interpolatable=False),
    }

    if cfg.dataset.labels_padding is not None:
        labels_padding = gp.Coordinate(cfg.dataset.labels_padding)
    if cfg.dataset.enable_mask:
        source_data_dict[labels_mask] = cfg.dataset.mask_name
        source_array_specs[labels_mask] = gp.ArraySpec(interpolatable=False)

    sources = []

    for tr_file in tr_files:
        src = ZarrSource(
            tr_file,
            source_data_dict,
            source_array_specs,
            in_memory=cfg.dataset.in_memory,
        )
        src += gp.Normalize(raw)
        src += gp.Pad(raw, None)
        if cfg.dataset.labels_padding is not None:
            src += gp.Pad(labels, labels_padding)
        if cfg.dataset.enable_mask and cfg.dataset.labels_padding is not None:
            src += gp.Pad(labels_mask, labels_padding)
        if cfg.dataset.enable_mask:
            # TODO: min_masked=0.5 causes freezing/extreme slowdown. 0.3 or 0.4 work fine. TODO: get ratio from cfg.dataset
            src += gp.RandomLocation(min_masked=0.3, mask=labels_mask)
        else:
            src += gp.RandomLocation()
        sources.append(src)

    assert len(sources) > 0
    sources = tuple(sources)

    pipeline = sources

    pipeline += gp.RandomProvider()

    # pipeline += gp.ElasticAugment(
    #     control_point_spacing=[4, 4, 10],
    #     jitter_sigma=[0, 2, 2],
    #     # rotation_interval=[0, np.pi / 2.0],
    #     rotation_interval=[-np.pi / 16.0, np.pi / 16.0],  # Smaller rotation interval so we don't need as much space for rotation
    #     prob_slip=0.05,
    #     prob_shift=0.05,
    #     max_misalign=10,
    #     subsample=2,
    # )

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])  # TODO: rot90

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        z_section_wise=True
    )

    pipeline += gp.GrowBoundary(
        labels,
        mask=labels_mask,
        steps=1,
        only_xy=True
    )

    # TODO: Find formula for valid combinations of sigma, downsample, input/output shapes
    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,
        **cfg.labels.lsd
    )

    neighborhood = cfg.labels.aff.nhood

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        labels_mask=labels_mask,
        affinities_mask=gt_affs_mask,
        dtype=np.float32
    )

    # if cfg.dataset.enable_mask:
    pipeline += gp.BalanceLabels(
        gt_affs,
        affs_weights,
        gt_affs_mask,
    )

    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Stack(cfg.training.batch_size)

    if num_workers > 0:
        pipeline += gp.PreCache(cache_size=40, num_workers=num_workers)

    if cfg.training.enable_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # pipeline += gp.IntensityScaleShift(raw, 2,-1)  # Rescale for training

    save_every = cfg.training.save_every
    pipeline += Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights
        },
        # log_dir = "./logs/"
        save_every=save_every,  # todo: increase,
        checkpoint_basename=str(save_path / 'model'),
        resume=False,
        enable_amp=cfg.training.enable_amp,
        enable_dynamo=cfg.training.enable_dynamo,
        save_jit=cfg.training.save_jit,
        example_input=example_input,
        cfg=cfg,
    )

    # pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)  # Rescale for snapshot outputs


    # pipeline += gp.PrintProfilingStats(every=10)

    # Write tensorboard logs into common parent folder for all trainings for easier comparison
    # tb = tensorboard.SummaryWriter(save_path.parent / "logs" / save_path.name)
    tb = tensorboard.SummaryWriter(save_path / "logs")

    wandb.init(config=_cfg_dict, **cfg.wandb.init_cfg)
    # Root directory where recursive code file discovery should start
    _code_root = Path(__file__).parent
    wandb.run.log_code(root=_code_root, include_fn=_get_include_fn(cfg.wandb.code_include_fn_exts))
    # Define summary metrics
    wandb.define_metric('training/scalars/loss', summary='min')
    wandb.define_metric('validation/scalars/loss', summary='min')
    wandb.define_metric('validation/scalars/voi', summary='min')


    with gp.build(pipeline):
        progress = tqdm(range(cfg.training.iterations), dynamic_ncols=True)
        for i in progress:
            _step_start_time = time.time()
            batch = pipeline.request_batch(request)
            # print('Batch sucessfull')
            start = request[labels].roi.get_begin() / voxel_size
            end = request[labels].roi.get_end() / voxel_size
            if (i + 1) % 10 or (i + 1) % save_every == 0:
                tb.add_scalar("loss", batch.loss, batch.iteration)
                wandb.log({'training/scalars/loss': batch.loss}, step=batch.iteration)
            if (i + 1) % save_every == 0:
                logging.info(
                    f'Evaluating at step {batch.iteration}, '
                    f'run time {get_run_time_str(_training_start_time)}, '
                    f'loss {batch.loss:.4f}...'
                )
                # raw_cropped = batch[raw].data[:, :, start[0]:end[0], start[1]:end[1]]
                # raw_sh = raw_cropped.shape
                # raw_slice = raw_cropped[0, 0, rsh[2] // 2]

                raw_data = batch[raw].data
                # halfz_in = raw_data.shape[2] // 2
                # full_raw_slice = raw_data[0, 0, halfz_in]
                full_raw_slice = get_zslice(raw_data[0])

                lab_data = batch[labels].data

                # halfz_out = lab_data.shape[1] // 2
                # lab_slice = lab_data[0, halfz_out]
                lab_slice = get_zslice(lab_data[0])

                # Center-crop raw slice to lab slice shape for overlay
                raw_slice, _ = spatial_center_crop_nd(full_raw_slice, lab_slice, ndim_spatial=2)

                if cfg.wandb.vis.enable_binary_labels:
                    lab_slice = lab_slice > 0
                    class_labels = {
                        0: 'bg',
                        1: 'fg',
                    }
                else:
                    lab_slice, num_ids = densify_labels(lab_slice)
                    class_labels = {i: f'c{i}' for i in range(num_ids)}


                gt_seg_overlay_img = wandb.Image(
                    raw_slice,
                    masks={
                        'gt_seg': {
                            'mask_data': lab_slice,
                            'class_labels': class_labels
                        },
                })

                gt_affs_slice = get_zslice(batch[gt_affs].data[0])
                gt_affs_img = wandb.Image(gt_affs_slice)

                pred_affs_slice = get_zslice(batch[pred_affs].data[0])
                pred_affs_img = wandb.Image(pred_affs_slice)

                pred_lsds3_slice = get_zslice(batch[pred_lsds].data[0][:3])
                pred_lsds3_img = wandb.Image(pred_lsds3_slice)

                wandb.log(
                    {
                        'training/images/gt_seg_overlay': gt_seg_overlay_img,
                        'training/images/gt_affs': gt_affs_img,
                        'training/images/pred_affs': pred_affs_img,
                        'training/images/pred_lsds3': pred_lsds3_img,
                    },
                    step=batch.iteration,
                    commit=False
                )

                if len(val_files) > 0:

                    checkpoint_path = save_path / f'model_checkpoint_{batch.iteration}.pth'
                    cube_eval_results = eval_cubes(cfg=cfg, checkpoint_path=checkpoint_path, enable_zarr_results=cfg.enable_zarr_results)

                    rand_voi_reports = {name: cube_eval_results[name].report for name in cube_eval_results.keys()}
                    # print(rand_voi_reports)
                    mean_report = get_mean_report(rand_voi_reports)
                    mean_report = prefixkeys(mean_report, prefix='validation/scalars/')
                    wandb.log(mean_report, step=batch.iteration, commit=False)

                    cevr = next(iter(cube_eval_results.values()))

                    val_raw_img = get_zslice(cevr.arrays['cropped_raw'], as_wandb=True)
                    val_pred_seg_img = get_zslice(cevr.arrays['cropped_pred_seg'], as_wandb=True, enable_rgb_labels=True)
                    val_pred_frag_img = get_zslice(cevr.arrays['cropped_pred_frag'], as_wandb=True, enable_rgb_labels=True)
                    val_pred_affs_img = get_zslice(cevr.arrays['cropped_pred_affs'], as_wandb=True)
                    val_pred_lsds3_img = get_zslice(cevr.arrays['cropped_pred_lsds'][:3], as_wandb=True)
                    val_gt_seg_img = get_zslice(cevr.arrays['gt_seg'], as_wandb=True, enable_rgb_labels=True)
                    val_gt_affs_img = get_zslice(cevr.arrays['gt_affs'], as_wandb=True)
                    val_gt_lsds3_img = get_zslice(cevr.arrays['gt_lsds'][:3], as_wandb=True)

                    wandb.log(
                        {
                            'validation/images/raw': val_raw_img,
                            'validation/images/pred_seg': val_pred_seg_img,
                            'validation/images/pred_frag': val_pred_frag_img,
                            'validation/images/pred_affs': val_pred_affs_img,
                            'validation/images/pred_lsds3': val_pred_lsds3_img,
                            'validation/images/gt_seg': val_gt_seg_img,
                            'validation/images/gt_affs': val_gt_affs_img,
                            'validation/images/gt_lsds3': val_gt_lsds3_img,
                        },
                        step=batch.iteration,
                        commit=False
                    )

                    # wandb.log(rand_voi_reports, commit=True)
                    if cfg.wandb.enable_per_cube_metrics:
                        per_cube_vois = get_per_cube_metrics(rand_voi_reports, metric_name='voi')
                        per_cube_vois = prefixkeys(per_cube_vois, prefix='validation/scalars/')
                        wandb.log(per_cube_vois, commit=False, step=batch.iteration)

                        per_cube_losses = get_per_cube_metrics(rand_voi_reports, metric_name='loss')
                        per_cube_losses = prefixkeys(per_cube_losses, prefix='validation/scalars/')
                        wandb.log(per_cube_losses, commit=False, step=batch.iteration)

            progress.set_description(f'Step {batch.iteration}, loss {batch.loss:.4f}')

            # Remember that this wand.log() call is the only one that uses commit=True, so don't just remove it.
            wandb.log(
                {'stats/speed_its_per_sec': 1 / (time.time() - _step_start_time)},
                step=batch.iteration,
                commit=True
            )


@hydra.main(version_base='1.3', config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
