# https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/train_mtlsd.ipynb
# conda install python=3.10 napari boost cython pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# pip install gunpowder matplotlib scikit-image scipy zarr tensorboard git+https://github.com/funkelab/funlib.evaluate git+https://github.com/funkelab/funlib.learn.torch.git git+https://github.com/funkey/waterz.git git+https://github.com/funkelab/lsd.git
# pip install hydra-core omegaconf

# TODO: Add ref to own gp fork

import datetime
import logging
from pathlib import Path

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
from segment_mtlsd import eval_cube, get_mean_report, get_per_cube_vois
from shared import create_lut, get_mtlsdmodel

# logging.basicConfig(level=logging.INFO)


# @title utility function to view labels

# matplotlib uses a default shader
# we need to recolor as unique objects


# @title utility  function to download / save data as zarr


# @title utility function to view a batch

# matplotlib.pyplot wrapper to view data
# default shape should be 2 - 2d data

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


# mtlsd model - designed to use lsds as an auxiliary learning task for improving affinities
# raw --> lsds / affs

# wrap model in a class. need two out heads, one for lsds, one for affs


# combine the lsds and affs losses

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


# def train(  # todo: validate?
#         tr_files,
#         iterations,
#         show_every,
#         show_gt=True,
#         show_pred=False,
#         lsd_channels=None,
#         aff_channels=None,
#         show_in_napari=False,
#         save_path=Path('.')
# ):
def train(cfg: DictConfig) -> None:

    tr_root = Path(cfg.tr_root)
    val_root = Path(cfg.val_root)
    tr_files = [str(fp) for fp in tr_root.glob('*.zarr')]
    val_files = [str(fp) for fp in val_root.glob('*.zarr')]

    # print(tr_root)
    # print(tr_files)

    timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    save_path = Path(_hydra_run_dir)
    # save_path = Path('/cajal/scratch/projects/misc/mdraw/lsd-results/training') / f'tr-{timestamp}'
    # save_path.mkdir(parents=True)
    logging.info(f'save_path: {save_path}')


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

    # pipeline += gp.SimpleAugment(transpose_only=[0, 1])  # todo: also rotate?

    pipeline += gp.IntensityAugment(  # todo: channel wise!
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)
    # todo: randomly reorder channels? (except for synapse markers homer/basoon)
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

    # neighborhood = [  # todo: order?
    #         #    [0, -1],
    #         #    [-1, 0]
    #         [0, 0, -1],
    #         [0, -1, 0],
    #         [-1, 0, 0],
    #         # [-1, 0, 0],
    #         # [1, 0, 0],
    #         # [0, -1, 0],
    #         # [0, 1, 0],
    #         # [0, 0, -1],
    #         # [0, 0, 1],
    # ]


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

    pipeline += gp.PreCache(num_workers=10)
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

    wandb.init(
        # set the wandb project where this run will be logged
        project="mlsd",

        # track hyperparameters and run metadata
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )

    with gp.build(pipeline):
        progress = tqdm(range(cfg.training.iterations), dynamic_ncols=True)
        for i in progress:
            batch = pipeline.request_batch(request)
            # print('Batch sucessfull')
            start = request[labels].roi.get_begin() / voxel_size
            end = request[labels].roi.get_end() / voxel_size
            if (i + 1) % 10 or (i + 1) % save_every == 0:
                tb.add_scalar("loss", batch.loss, batch.iteration)
                wandb.log({"loss": batch.loss})
            if (i + 1) % save_every == 0:
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
                rand_voi_reports = eval_cube(save_path / f'model_checkpoint_{batch.iteration}', show_in_napari=cfg.training.show_in_napari)
                # print(rand_voi_reports)
                mean_report = get_mean_report(rand_voi_reports)
                wandb.log(mean_report, commit=False)
                # wandb.log(rand_voi_reports, commit=True)
                per_cube_vois = get_per_cube_vois(rand_voi_reports)
                wandb.log(per_cube_vois, commit=True)
                # tb.add_figure("eval", fig, batch.iteration)
                # tb.add_scalar("voi_split", voi_split, batch.iteration)
                # tb.add_scalar("voi_merge", voi_merge, batch.iteration)
                # tb.add_scalar("voi", voi_split + voi_merge, batch.iteration)

                tb.flush()
            progress.set_description(f'Training iteration {i}')
            pass
    # todo: save weights?


# view a batch of ground truth lsds/affs, no need to show predicted lsds/affs yet

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

# just view first y affs
# aff_channels = {'affs': 0}

# train(
#    iterations=1,
#    show_every=1,
#    lsd_channels=lsd_channels,
#    aff_channels=aff_channels)

# lets just view the mean offset y channels
# train for ~1k iterations, view every 100th batch
# show the prediction as well as the ground truth

# lsd_channels = {'offset (y)': 0}
aff_channels = {
    'affs_0': 0,  # todo: fix names
    'affs_1': 1,
    'affs_2': 2,
    # 'affs_3': 3,
    # 'affs_4': 4,
    # 'affs_5': 5,
}

# assert torch.cuda.is_available()

@hydra.main(version_base='1.3', config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # tr_root = Path('/cajal/scratch/projects/misc/mdraw/data/zebrafinch_msplit/training/').expanduser()
    # val_root = Path('/cajal/scratch/projects/misc/mdraw/data/zebrafinch_msplit/validation/').expanduser()
    # tr_files = [str(fp) for fp in tr_root.glob('*.zarr')]
    # val_files = [str(fp) for fp in val_root.glob('*.zarr')]

    # timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    # save_path = Path('/cajal/scratch/projects/misc/mdraw/lsd-results/training') / f'tr-{timestamp}'
    # save_path.mkdir(parents=True)

    # train(
    #     tr_files=tr_files,
    #     # iterations=100001,
    #     iterations=101,
    #     show_every=20,
    #     show_pred=True,
    #     lsd_channels=lsd_channels,
    #     aff_channels=aff_channels,
    #     show_in_napari=False,
    #     save_path=save_path
    # )
    train(cfg)


if __name__ == "__main__":
    main()
