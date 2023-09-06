# Based on https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_mtlsd.ipynb

import multiprocessing
multiprocessing.set_start_method('spawn', True)
# multiprocessing.set_start_method('forkserver', True)

# import torch
# torch.multiprocessing.set_start_method('spawn', True)

import os
from unittest.mock import MagicMock


# Make sure we're not multithreading because we already use one process per CPU core in training
#  See https://stackoverflow.com/a/53224849
for lib in ['OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'VECLIB']:
    os.environ[f'{lib}_NUM_THREADS'] = '1'


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
# wandb = MagicMock()

# from params import input_size, output_size, voxel_size
from nseg.segment_mtlsd import eval_cubes, get_mean_report, get_per_cube_metrics, spatial_center_crop_nd
from nseg.shared import compute_loss_maps, build_mtlsdmodel, import_symbol
from nseg.gpx.gp_train import Train
from nseg.gpx.gp_sources import ZarrSource
from nseg.gpx.gp_boundaries import AddBoundaryLabels
from nseg.gpx.gp_distmaps import AddDistMap
from nseg.gpx.gp_tensor import Cast, ToTorch, ToNumpy
from nseg.conf import NConf, DictConfig, hydra
from omegaconf.errors import ConfigAttributeError



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


def get_mpl_imshow_fig(img):
    fig, ax = plt.subplots(figsize=(8, 6))
    aximg = ax.imshow(img)
    fig.colorbar(aximg)
    return fig


def train(cfg: DictConfig) -> None:
    _training_start_time = time.time()
    tr_root = Path(cfg.dataset.tr_root)
    tr_files = sorted(list(tr_root.glob('*.zarr')))

    # Figure out what outputs we want to train (only outputs with nonzero loss term weights are considered)
    # Note that aff(inity) is always included as it's necessary for segmentation
    try:
        loss_term_weights = cfg.loss.init_kwargs.loss_term_weights
        nonzero_loss_term_weights = {
            k: v for k, v in loss_term_weights.items() if v != 0 and v is not None
        }
        nonzero_loss_term_names = list(nonzero_loss_term_weights.keys())
    except ConfigAttributeError:
        # Missing loss term config -> fall back to lsd and aff (original LSD loss)
        nonzero_loss_term_names = ['lsd', 'aff']

    lsd_enabled = 'lsd' in nonzero_loss_term_names
    boundaries_enabled = 'bce' in nonzero_loss_term_names or 'bcd' in nonzero_loss_term_names
    boundary_distmaps_enabled = 'bdt' in nonzero_loss_term_names
    hardness_enabled = 'hardness' in nonzero_loss_term_names

    float_dtype = np.float32

    val_root = cfg.dataset.val_root
    if val_root is None:
        val_files = []
    else:
        val_root = Path(val_root)
        val_files = sorted(list(val_root.glob('*.zarr')))

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
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    gt_boundaries = gp.ArrayKey('GT_BOUNDARIES')
    pred_boundaries = gp.ArrayKey('PRED_BOUNDARIES')

    gt_boundary_distmap = gp.ArrayKey('GT_BOUNDARY_DISTMAP')
    pred_boundary_distmap = gp.ArrayKey('PRED_BOUNDARY_DISTMAP')

    pred_hardness = gp.ArrayKey('PRED_HARDNESS')

    if cfg.dataset.enable_mask:
        gt_labels_mask = gp.ArrayKey('GT_LABELS_MASK')
        gt_affs_mask = gp.ArrayKey('GT_AFFINITIES_MASK')
    else:
        gt_labels_mask = None
        gt_affs_mask = None

    voxel_size = gp.Coordinate(cfg.dataset.voxel_size)
    # Prefer ev_inp_shape if specified, use regular inp_shape otherwise
    input_shape = gp.Coordinate(cfg.model.backbone.inp_shape)
    input_size = input_shape * voxel_size
    offset = gp.Coordinate(cfg.model.backbone.offset)
    output_shape = input_shape - offset
    output_size = output_shape * voxel_size

    model_cfg = cfg.model
    model = build_mtlsdmodel(model_cfg)
    example_input = torch.randn(
        1,  # cfg.training.batch_size,
        1,  # cfg.model.backbone.init_kwargs.in_channels,
        *cfg.model.backbone.inp_shape,
    )

    # loss = HardnessEnhancedLoss(cfg.loss.init_kwargs)

    loss_class = import_symbol(cfg.loss.loss_class)
    loss_init_kwargs = cfg.loss.get('init_kwargs', {})
    loss = loss_class(**loss_init_kwargs)

    optimizer = torch.optim.Adam(
        lr=cfg.training.lr,
        betas=tuple(cfg.training.adam_betas),
        params=model.parameters()
    )

    trainer_inputs = {'input': raw}
    trainer_outputs = {
        'pred_affs': pred_affs,
    }
    trainer_loss_inputs = {
        'pred_affs': pred_affs,
        'gt_affs': gt_affs,
        'affs_weights': affs_weights,
        'gt_labels_mask': gt_labels_mask,
    }

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    if lsd_enabled:
        request.add(gt_lsds, output_size)
        request.add(lsds_weights, output_size)
        request.add(pred_lsds, output_size)
        trainer_outputs['pred_lsds'] = trainer_loss_inputs['pred_lsds'] = pred_lsds
        trainer_loss_inputs['gt_lsds'] = gt_lsds
        trainer_loss_inputs['lsds_weights'] = lsds_weights

    if boundaries_enabled:
        request.add(gt_boundaries, output_size)
        request.add(pred_boundaries, output_size)
        trainer_outputs['pred_boundaries'] = trainer_loss_inputs['pred_boundaries'] = pred_boundaries
        trainer_loss_inputs['gt_boundaries'] = gt_boundaries

    if boundary_distmaps_enabled:
        request.add(gt_boundary_distmap, output_size)
        request.add(pred_boundary_distmap, output_size)
        trainer_outputs['pred_boundary_distmap'] = trainer_loss_inputs['pred_boundary_distmap'] = pred_boundary_distmap
        trainer_loss_inputs['gt_boundary_distmap'] = gt_boundary_distmap

    if hardness_enabled:
        request.add(pred_hardness, output_size)
        trainer_outputs['pred_hardness'] = trainer_loss_inputs['pred_hardness'] = pred_hardness

    if cfg.dataset.enable_mask:
        request.add(gt_labels_mask, output_size)
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
        source_data_dict[gt_labels_mask] = cfg.dataset.mask_name
        source_array_specs[gt_labels_mask] = gp.ArraySpec(interpolatable=False)

    sources = []
    sampling_weights_list = []

    for tr_file in tr_files:
        src = ZarrSource(
            str(tr_file),
            source_data_dict,
            source_array_specs,
            in_memory=cfg.dataset.in_memory,
        )
        src += gp.Normalize(raw, dtype=np.float32)
        # src += gp.Normalize(raw, dtype=float_dtype)
        src += gp.Pad(raw, None)
        if cfg.dataset.labels_padding is not None:
            src += gp.Pad(labels, labels_padding)
        if cfg.dataset.enable_mask and cfg.dataset.labels_padding is not None:
            src += gp.Pad(gt_labels_mask, labels_padding)
        if cfg.dataset.enable_mask:
            # TODO: min_masked=0.5 causes freezing/extreme slowdown. 0.3 or 0.4 work fine. TODO: get ratio from cfg.dataset
            src += gp.RandomLocation(min_masked=0.5, mask=gt_labels_mask)
        else:
            src += gp.RandomLocation()
        sources.append(src)
        sampling_weights = cfg.dataset.get('sampling_weights')
        if sampling_weights is not None:
            sw = sampling_weights.get(tr_file.name, 1)
            sampling_weights_list.append(sw)
            if sw != 1:
                logging.info(f'Using sampling weight {sw} for {tr_file.name}')


    assert len(sources) > 0
    sources = tuple(sources)

    pipeline = sources

    sampling_weights_list = sampling_weights_list if len(sampling_weights_list) > 0 else None
    pipeline += gp.RandomProvider(probabilities=sampling_weights_list)

    if cfg.augmentations.enable_elastic:
        # TODO: Is control_point_spacing accidentally xyz instead of zyx?
        pipeline += gp.ElasticAugment(
            control_point_spacing=(4, 4, 10),
            # control_point_spacing=(10, 4, 4),
            jitter_sigma=(0, 2, 2),
            rotation_interval=[0, np.pi / 2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8,
        )

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
        mask=gt_labels_mask,
        steps=1,
        only_xy=True
    )

    if boundaries_enabled:
        pipeline += AddBoundaryLabels(
            instance_labels=labels,
            boundary_labels=gt_boundaries,
            dtype=np.int64,
            # dtype=np.uint8,  # TODO: torch._C._nn.cross_entropy_loss fails on uint8 targets
        )

    if boundary_distmaps_enabled:
        pipeline += AddDistMap(
            instance_labels=labels,
            distmap=gt_boundary_distmap,
            vector_enabled=False,
            inverted=True,
            signed=True,
            scale=50,
            dtype=float_dtype,
        )

    if lsd_enabled:
        # TODO: Find formula for valid combinations of sigma, downsample, input/output shapes
        pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            lsds_mask=lsds_weights,
            dtype=float_dtype,
            **cfg.labels.lsd
        )

    neighborhood = cfg.labels.aff.nhood

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        labels_mask=gt_labels_mask,
        affinities_mask=gt_affs_mask,
        dtype=float_dtype
    )

    # TODO: Replace by global label balancing.
    pipeline += gp.BalanceLabels(
        gt_affs,
        affs_weights,
        gt_affs_mask,
    )

    # pipeline += Cast(labels, dtype=np.uint8)

    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Stack(cfg.training.batch_size)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)  # Rescale for training


    # pipeline += ToTorch()

    if num_workers > 0:
        pipeline += gp.PreCache(cache_size=40, num_workers=num_workers)

    if cfg.training.enable_cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


    save_every = cfg.training.save_every
    trainer = Train(
        model,
        loss,
        optimizer,
        inputs=trainer_inputs,
        outputs=trainer_outputs,
        loss_inputs=trainer_loss_inputs,
        # log_dir = "./logs/"
        save_every=save_every,
        checkpoint_basename=str(save_path / 'model'),
        resume=False,
        enable_amp=cfg.training.enable_amp,
        enable_dynamo=cfg.training.enable_dynamo,
        save_jit=cfg.training.save_jit,
        example_input=example_input,
        cfg=cfg,
    )

    pipeline += trainer

    # pipeline += ToNumpy()

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)  # Rescale for snapshot outputs

    pipeline += gp.PrintProfilingStats(every=cfg.training.save_every)

    # Write tensorboard logs into common parent folder for all trainings for easier comparison
    # tb = tensorboard.SummaryWriter(save_path.parent / "logs" / save_path.name)
    # tb = tensorboard.SummaryWriter(save_path / "logs")

    wandb.init(config=_cfg_dict, **cfg.wandb.init_cfg)
    # Root directory where recursive code file discovery should start
    _code_root = Path(__file__).parent
    wandb.run.log_code(root=_code_root, include_fn=_get_include_fn(cfg.wandb.code_include_fn_exts))
    # Define summary metrics
    wandb.define_metric('training/scalars/loss', summary='min')
    wandb.define_metric('validation/scalars/loss', summary='min')
    wandb.define_metric('validation/scalars/voi', summary='min')


    with gp.build(pipeline):
        progress = tqdm(range(trainer.iteration, cfg.training.iterations), dynamic_ncols=True)
        for i in progress:
            _step_start_time = time.time()
            batch = pipeline.request_batch(request)
            # print('Batch sucessfull')
            start = request[labels].roi.get_begin() / voxel_size
            end = request[labels].roi.get_end() / voxel_size
            # Determine if/how much we should log this iteration
            _full_eval_now = (i + 1) % save_every == 0 or cfg.training.first_eval_at_step == i + 1
            _loss_log_now = _full_eval_now or (i + 1) % 100 == 0
            if _loss_log_now:
                # tb.add_scalar("loss", batch.loss, batch.iteration)
                wandb.log({'training/scalars/loss': batch.loss}, step=batch.iteration)
            if _full_eval_now:
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

                tr_imgs_wandb = {
                    'training/images/gt_seg_overlay': gt_seg_overlay_img,
                    'training/images/gt_affs': gt_affs_img,
                    'training/images/pred_affs': pred_affs_img,
                }

                if lsd_enabled:
                    gt_lsds3_slice = get_zslice(batch[gt_lsds].data[0][:3])
                    gt_lsds3_img = wandb.Image(gt_lsds3_slice)
                    pred_lsds3_slice = get_zslice(batch[pred_lsds].data[0][:3])
                    pred_lsds3_img = wandb.Image(pred_lsds3_slice)
                    tr_imgs_wandb.update({
                        'training/images/gt_lsds3': gt_lsds3_img,
                        'training/images/pred_lsds3': pred_lsds3_img,
                    })

                if hardness_enabled:
                    pred_hardness_slice = get_zslice(batch[pred_hardness].data[0])
                    pred_hardness_fig = get_mpl_imshow_fig(pred_hardness_slice)
                    tr_imgs_wandb.update({
                        'training/images/pred_hardness': pred_hardness_fig,
                    })

                if boundaries_enabled:
                    with torch.inference_mode():
                        _th_pred_boundaries = torch.as_tensor(batch[pred_boundaries].data[0], dtype=torch.float32)
                        _np_sm_pred_boundaries = torch.softmax(_th_pred_boundaries, dim=0).numpy().astype(np.float32)[1]
                    pred_boundaries_slice = get_zslice(_np_sm_pred_boundaries)
                    pred_boundaries_fig = get_mpl_imshow_fig(pred_boundaries_slice)
                    # pred_boundaries_fig = wandb.Image(pred_boundaries_slice)
                    tr_imgs_wandb.update({
                        'training/images/pred_boundaries': pred_boundaries_fig,
                    })

                if boundary_distmaps_enabled:
                    pred_boundary_distmap_slice = get_zslice(batch[pred_boundary_distmap].data[0])
                    pred_boundary_distmap_fig = get_mpl_imshow_fig(pred_boundary_distmap_slice)
                    tr_imgs_wandb.update({
                        'training/images/boundary_distmap': pred_boundary_distmap_fig,
                    })

                if hasattr(loss, 'compute_seg_loss_maps') and hasattr(loss, '_combine_seg_loss_maps'):
                    loss_maps = compute_loss_maps(
                        loss_module=loss,
                        batch=batch,
                        loss_inputs_gpkeys=trainer.loss_inputs,
                        device=trainer.device
                    )
                    loss_maps = prefixkeys(loss_maps, 'training/images/loss_maps/')

                    loss_maps_z_figs = {
                        k: get_mpl_imshow_fig(get_zslice(v[0]))
                        for k, v in loss_maps.items()
                    }
                    tr_imgs_wandb.update(loss_maps_z_figs)

                wandb.log(
                    tr_imgs_wandb,
                    step=batch.iteration,
                    commit=False
                )

                checkpoint_path = save_path / f'model_checkpoint_{batch.iteration}.pth'
                if not checkpoint_path.exists():
                    # Manually create checkpoint if it doesn't exist (can happen on `first_eval_at_step` trigger)
                    trainer._save_model(save_state_dict=True)

                if len(val_files) > 0:

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
                    val_gt_affs_img = get_zslice(cevr.arrays['gt_affs'], as_wandb=True)
                    val_gt_seg_img = get_zslice(cevr.arrays['gt_seg'], as_wandb=True, enable_rgb_labels=True)

                    val_imgs_wandb = {
                        'validation/images/raw': val_raw_img,
                        'validation/images/pred_seg': val_pred_seg_img,
                        'validation/images/pred_frag': val_pred_frag_img,
                        'validation/images/pred_affs': val_pred_affs_img,
                        'validation/images/gt_seg': val_gt_seg_img,
                        'validation/images/gt_affs': val_gt_affs_img,
                    }

                    if lsd_enabled:
                        val_pred_lsds3_img = get_zslice(cevr.arrays['cropped_pred_lsds'][:3], as_wandb=True)
                        val_gt_lsds3_img = get_zslice(cevr.arrays['gt_lsds'][:3], as_wandb=True)
                        val_imgs_wandb.update({
                            'validation/images/pred_lsds3': val_pred_lsds3_img,
                            'validation/images/gt_lsds3': val_gt_lsds3_img,
                        })

                    wandb.log(
                        val_imgs_wandb,
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
