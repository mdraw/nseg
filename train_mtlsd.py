# https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_mtlsd.ipynb
# conda install python=3.10 napari boost cython pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# pip install gunpowder matplotlib scikit-image scipy zarr tensorboard git+https://github.com/funkelab/funlib.evaluate git+https://github.com/funkelab/funlib.learn.torch.git git+https://github.com/funkey/waterz.git git+https://github.com/funkelab/lsd.git
# pip install hydra-core omegaconf

# TODO: Add ref to own gp fork

import os
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use('AGG')

import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np
import torch
from gunpowder.torch import Train
from lsd.train.gp import AddLocalShapeDescriptor
from torch.utils import tensorboard
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf, DictConfig

import wandb

from params import input_size, output_size, batch_size, voxel_size
from segment_mtlsd import center_crop, eval_cubes, get_mean_report, get_per_cube_metrics, spatial_center_crop_nd
from shared import create_lut, get_mtlsdmodel


lsd_channels = {
    'offset (y)': 0,
    'offset (x)': 1,
    'offset (z)': 2,
    'orient (y)': 3,
    'orient (x)': 4,
    'orient (z)': 5,
    'yx change': 6,  # todo: order correct?
    'yz change': 7,
    'xz change': 8,
    'voxel count': 9
}

aff_channels = {
    'affs_0': 0,  # todo: fix names
    'affs_1': 1,
    'affs_2': 2,
    # 'affs_3': 3,
    # 'affs_4': 4,
    # 'affs_5': 5,
}



def imshow(
        tb, it,
        raw=None,
        ground_truth=None,
        target=None,
        prediction=None,
        h=None,
        shader='jet',
        subplot=True,
        channel=0,
        target_name='target',
        prediction_name='prediction'):
    raw = raw[:, :, :, raw.shape[-1] // 2] if raw is not None else None
    ground_truth = ground_truth[:, :, :, ground_truth.shape[-1] // 2] if ground_truth is not None else None
    target = target[:, :, :, :, target.shape[-1] // 2] if target is not None else None
    prediction = prediction[:, :, :, :, prediction.shape[-1] // 2] if prediction is not None else None

    rows = 0

    if raw is not None:
        rows += 1
        cols = raw.shape[0] if len(raw.shape) > 2 else 1
    if ground_truth is not None:
        rows += 1
        cols = ground_truth.shape[0] if len(ground_truth.shape) > 2 else 1
    if target is not None:
        rows += 1
        cols = target.shape[0] if len(target.shape) > 2 else 1
    if prediction is not None:
        rows += 1
        cols = prediction.shape[0] if len(prediction.shape) > 2 else 1

    if subplot:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 4),
            sharex=True,
            sharey=True,
            squeeze=False)

    if h is not None:
        fig.subplots_adjust(hspace=h)

    def wrapper(data, row, name="raw"):

        if subplot:
            if len(data.shape) == 2:
                if name == 'raw':
                    axes[0][0].imshow(data, cmap='gray')
                    axes[0][0].set_title(name)
                else:
                    axes[row][0].imshow(create_lut(data))
                    axes[row][0].set_title(name)

            elif len(data.shape) == 3:
                for i, im in enumerate(data):
                    if name == 'raw':
                        axes[0][i].imshow(im, cmap='gray')
                        axes[0][i].set_title(name)
                    else:
                        axes[row][i].imshow(create_lut(im))
                        axes[row][i].set_title(name)

            else:
                for i, im in enumerate(data):
                    axes[row][i].imshow(im[channel], cmap=shader)
                    axes[row][i].set_title(name + str(channel))


        else:
            if name == 'raw':
                plt.imshow(data, cmap='gray')
            if name == 'labels':
                plt.imshow(data, alpha=0.5)

    row = 0
    if raw is not None:
        wrapper(raw, row=row)
        row += 1
    if ground_truth is not None:
        wrapper(ground_truth, row=row, name='labels')
        row += 1
    if target is not None:
        wrapper(target, row=row, name=target_name)
        row += 1
    if prediction is not None:
        wrapper(prediction, row=row, name=prediction_name)
        row += 1
    # for label in axes.xaxis.get_tick_labels()[1::2]:
    #    print(len(label.get_text()))
    tb.add_figure(axes[0][0].title.get_text(), fig, it)
    return plt
    # plt.show()


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights,
    ):

        # calc each loss and combine
        loss1 = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        loss2 = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss1 + loss2

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
def get_zslice(vol, z_plane=None, squeeze=True, as_wandb=False):
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
    if as_wandb:
        return wandb.Image(slc)
    return slc


def prefixkeys(dictionary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Returns a dict with prefixed key names but same values."""
    prefdict = {}
    for key, value in dictionary.items():
        prefdict[f'{prefix}{key}'] = value
    return prefdict


def train(cfg: DictConfig) -> None:
    tr_root = Path(cfg.tr_root)
    val_root = Path(cfg.val_root)
    tr_files = [str(fp) for fp in tr_root.glob('*.zarr')]
    val_files = [str(fp) for fp in val_root.glob('*.zarr')]

    # Get standard Python dict representation of omegaconf cfg (for wandb cfg logging)
    _cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Find the current hydra run dir
    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    # Set general save_path (also for non-hydra-related outputs) to hydra run dir so we have everything in one place
    save_path = Path(_hydra_run_dir)
    logging.info(f'save_path: {save_path}')
    num_workers = cfg.training.num_workers
    if num_workers == 'auto':
        num_workers = len(os.sched_getaffinity(0))  # Get cores available cpu cores for this (SLURM) job
        logging.info(f'num_workers: automatically using all {num_workers} available cpu cores')

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    model = get_mtlsdmodel()

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

    ## TODO: Padding?
    # labels_padding = gp.Coordinate((350,550,550))
    sources = tuple(
        gp.ZarrSource(
            tr_file,
            {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids'
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False)
            }) +
        gp.Normalize(raw) +
        # gp.Squeeze([raw], axis=0) +
        # gp.Pad(raw, None) +
        # gp.Pad(labels, labels_padding) +
        gp.RandomLocation()
        for tr_file in tr_files
    )

    # raw:      (h, w)
    # labels:   (h, w)

    pipeline = sources

    pipeline += gp.RandomProvider()

    # Note before re-enabling: SimpleAugement can fail! Why? Maybe axes are wrong and expect xyz?
    # pipeline += gp.SimpleAugment(transpose_only=[0, 1])  # todo: also rotate?

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)
    pipeline += gp.GrowBoundary(labels)

    # TODO: Find formula for valid combinations of sigma, downsample, input/output shapes
    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,
        sigma=80,  # 80,  # todo: tune --> zf: 120, see https://github.com/funkelab/lsd/issues/9#issuecomment-1065299067
        downsample=2  # todo: tune
    )

    neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype=np.float32)

    pipeline += gp.BalanceLabels(  # todo: needed?
        gt_affs,
        affs_weights)

    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=num_workers)

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
    )

    # Write tensorboard logs into common parent folder for all trainings for easier comparison
    # tb = tensorboard.SummaryWriter(save_path.parent / "logs" / save_path.name)
    tb = tensorboard.SummaryWriter(save_path / "logs")

    wandb.init(config=_cfg_dict, **cfg.wandb.init_cfg)
    # Root directory where recursive code file discovery should start
    _code_root = Path(__file__).parent
    wandb.run.log_code(root=_code_root, include_fn=_get_include_fn(cfg.wandb.code_include_fn_exts))

    with gp.build(pipeline):
        progress = tqdm(range(cfg.training.iterations), dynamic_ncols=True)
        for i in progress:
            batch = pipeline.request_batch(request)
            # print('Batch sucessfull')
            start = request[labels].roi.get_begin() / voxel_size
            end = request[labels].roi.get_end() / voxel_size
            if (i + 1) % 10 or (i + 1) % save_every == 0:
                tb.add_scalar("loss", batch.loss, batch.iteration)
                wandb.log({'training/scalars/loss': batch.loss}, step=batch.iteration)
            if (i + 1) % save_every == 0:
                logging.info('logging batch visualizations')

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
                    step=batch.iteration
                )
                # for c in range(3):
                #     img = wandb.Image(gt_affs_slice[c])
                # gt_affs_img = get_img(batch[gt_affs].data)


                # import IPython ; IPython.embed(); raise SystemExit


                for c in range(batch[raw].data.shape[1]):
                    imshow(tb=tb, it=batch.iteration,
                           raw=np.squeeze(batch[raw].data[:, :, start[0]:end[0], start[1]:end[1]]), channel=c)

                imshow(tb=tb, it=batch.iteration, ground_truth=batch[labels].data)

                if lsd_channels:
                    for n, c in lsd_channels.items():

                        if cfg.training.show_gt:
                            imshow(tb=tb, it=batch.iteration, target=batch[gt_lsds].data, target_name='gt ' + n,
                                   channel=c)
                        if cfg.training.show_pred:
                            imshow(tb=tb, it=batch.iteration, prediction=batch[pred_lsds].data,
                                   prediction_name='pred ' + n, channel=c)

                if aff_channels:
                    for n, c in aff_channels.items():

                        if cfg.training.show_gt:
                            imshow(tb=tb, it=batch.iteration, target=batch[gt_affs].data, target_name='gt ' + n,
                                   channel=c)
                        if cfg.training.show_pred:
                            imshow(tb=tb, it=batch.iteration, target=batch[pred_affs].data, target_name='pred ' + n,
                                   channel=c)

                # fig, voi_split, voi_merge = eval_cube(save_path / f'model_checkpoint_{batch.iteration}', show_in_napari=cfg.training.show_in_napari)
                cube_eval_results = eval_cubes(
                    cube_root=val_root,
                    checkpoint=save_path / f'model_checkpoint_{batch.iteration}',
                    result_zarr_root=save_path
                    show_in_napari=cfg.training.show_in_napari
                )
                rand_voi_reports = {name: cube_eval_results[name].report for name in cube_eval_results.keys()}
                # print(rand_voi_reports)
                mean_report = get_mean_report(rand_voi_reports)
                mean_report = prefixkeys(mean_report, prefix='validation/scalars/')
                wandb.log(mean_report, step=batch.iteration)

                cevr = next(iter(cube_eval_results.values()))

                val_raw_img = get_zslice(cevr.raw, as_wandb=True)
                val_gt_seg_img = get_zslice(cevr.gt_seg, as_wandb=True)
                val_pred_seg_img = get_zslice(cevr.pred_seg, as_wandb=True)
                val_pred_frag_img = get_zslice(cevr.pred_frag, as_wandb=True)
                val_pred_affs_img = get_zslice(cevr.pred_affs, as_wandb=True)
                val_pred_lsds3_img = get_zslice(cevr.pred_lsds[:3], as_wandb=True)

                # TODO: Looks like frag and seg are being binarized during logging but we don't want that!

                wandb.log(
                    {
                        'validation/images/raw': val_raw_img,
                        'validation/images/gt_seg': val_gt_seg_img,
                        'validation/images/pred_seg': val_pred_seg_img,
                        'validation/images/pred_frag': val_pred_frag_img,
                        'validation/images/pred_affs': val_pred_affs_img,
                        'validation/images/pred_lsds3': val_pred_lsds3_img,
                    },
                    step=batch.iteration
                )

                # wandb.log(rand_voi_reports, commit=True)
                if cfg.wandb.enable_per_cube_metrics:
                    per_cube_vois = get_per_cube_metrics(rand_voi_reports, metric_name='voi')
                    per_cube_vois = prefixkeys(per_cube_vois, prefix='validation/scalars/')
                    wandb.log(per_cube_vois, commit=True, step=batch.iteration)
                # tb.add_figure("eval", fig, batch.iteration)
                # tb.add_scalar("voi_split", voi_split, batch.iteration)
                # tb.add_scalar("voi_merge", voi_merge, batch.iteration)
                # tb.add_scalar("voi", voi_split + voi_merge, batch.iteration)

                tb.flush()
            progress.set_description(f'Training iteration {i}')
            pass
    # todo: save weights?


@hydra.main(version_base='1.3', config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
