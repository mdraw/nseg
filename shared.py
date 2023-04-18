import numpy as np
import torch
from torch import nn
# from funlib.learn.torch.models import UNet, ConvPass
import importlib

from funlib_unet import UNet
# from elektronn3.models.unet import UNet as EU


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


def create_lut(labels):
    max_label = np.max(labels)

    # Generate random RGB colors for all labels
    lut = np.random.randint(
        low=0,
        high=255,
        size=(int(max_label + 1), 3),
        dtype=np.uint8)

    # Add an alpha=255 channel to all colors -> RGBA values
    lut = np.append(
        lut,
        np.zeros(
            (int(max_label + 1), 1),
            dtype=np.uint8) + 255,
        axis=1)

    # Set background color to (0,0,0,0)
    lut[0] = 0
    # Get colors from label LUT
    colored_labels = lut[labels]

    return colored_labels


# Deprecated
def get_mtlsdmodel(padding='valid'):  # todo: also use advanced architectures
    in_channels = 1
    num_fmaps = 12
    fmap_inc_factor = 5
    ds_fact = [(2, 2, 2), (2, 2, 2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    ksu = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)
    constant_upsample = True
    model = MtlsdModel(
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        ds_fact,
        ksd,
        ksu,
        constant_upsample,
        padding=padding)
    return model


def get_funlib_unet(
        in_channels=1,
        num_fmaps=14,
        fmap_inc_factor=5,
        ds_fact=((2, 2, 2), (2, 2, 2)),
        constant_upsample = True,
        padding='valid',
):
    """Construct a funlib U-Net model as used in the LSD paper. Defaults to the paper's architecture config."""
    from funlib_unet import UNet

    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    ksu = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)

    return UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        constant_upsample=constant_upsample,
        padding=padding
    )


# Deprecated
class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            constant_upsample,
            padding):
        super().__init__()

        # create unet
        # self.unet = UNet(
        #     in_channels=in_channels,
        #     num_fmaps=num_fmaps,
        #     fmap_inc_factor=fmap_inc_factor,
        #     downsample_factors=downsample_factors,
        #     kernel_size_down=kernel_size_down,
        #     kernel_size_up=kernel_size_up,
        #     constant_upsample=constant_upsample,
        #     padding=padding)

        from elektronn3.models.unet import UNet as EU
        self.unet = EU(
            in_channels=in_channels,
            out_channels=num_fmaps,
            dim=3,
            conv_mode='same',
            n_blocks=4,
            padding=padding,
        )
        # self.unet = torch.nn.Conv3d(in_channels, num_fmaps, 1)

        # create lsd and affs heads
        self.lsd_head = ConvPass(num_fmaps, 10  # 6
                                 , [[1, 1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, 3  # 6  # 2 #todo: only L1 grid neighbors?
                                 , [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):
        # pass raw through unet
        z = self.unet(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)
        # print(input.shape, z.shape, lsds.shape, affs.shape)

        return lsds, affs


# TODO: cfg
def get_swin_unetr_v2(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=14,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name='instance',
        drop_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample='merging',
        # use_v2=True,
):
    from monai.networks.nets.swin_unetr import SwinUNETR

    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=depths,
        num_heads=num_heads,
        feature_size=feature_size,
        norm_name=norm_name,
        drop_rate=drop_rate,
        normalize=normalize,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
        downsample=downsample,
        # use_v2=use_v2,
    )
    return model


def import_symbol(import_path: str) -> type:
    """Return symbol (class, function, ...) given an import path string"""
    if '.' not in import_path:
        raise ValueError('import path with at least one "." expected.')
    mod_path, _, cls_name = import_path.rpartition('.')
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls


def build_mtlsdmodel(model_cfg):
    backbone_class = import_symbol(model_cfg.backbone.model_class)
    init_kwargs = model_cfg.backbone.get('init_kwargs', {})
    backbone = backbone_class(**init_kwargs)

    lsd_head = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 10, 1),
        nn.Sigmoid()
    )

    aff_head = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 3, 1),
        nn.Sigmoid()
    )

    model = GeneralMtlsdModel(
        backbone=backbone, lsd_head=lsd_head, aff_head=aff_head
    )
    return model


class GeneralMtlsdModel(torch.nn.Module):

    def __init__(
            self,
            backbone: torch.nn.Module,
            lsd_head: torch.nn.Module,
            aff_head: torch.nn.Module,
        ):
        super().__init__()

        self.backbone = backbone
        self.lsd_head = lsd_head
        self.aff_head = aff_head
        self.register_module('backbone', backbone)
        self.register_module('lsd_head', lsd_head)
        self.register_module('aff_head', aff_head)

    def forward(self, input):
        # pass raw through unet
        z = self.backbone(input)

        # pass output through heads
        lsds = self.lsd_head(z)
        affs = self.aff_head(z)
        # print(input.shape, z.shape, lsds.shape, affs.shape)

        return lsds, affs

