import numpy as np
import torch
from torch import nn
import importlib

from elektronn3.models import unet as e3unet

from nseg import funlib_unet



class HardnessEnhancedLoss(torch.nn.Module):
    def __init__(
            self,
            enable_hardness_weighting=True,
            hardness_loss_formula: str = 'original_hl',
            # hardness_c=0.1,
            hardness_alpha=0.01,
            enable_mean_reduction=True
    ):
        super().__init__()

        self.enable_hardness_weighting = enable_hardness_weighting
        self.hardness_loss_formula = hardness_loss_formula
        self.hardness_alpha = hardness_alpha
        self.enable_mean_reduction = enable_mean_reduction

    @staticmethod
    def _scaled_mse(pred, target, weights):
        scaled = weights * (pred - target) ** 2
        if len(torch.nonzero(scaled)) != 0:
            scaled = torch.masked_select(scaled, torch.gt(weights, 0))
        return scaled

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights,
            hardness_prediction,
    ):
        if self.enable_hardness_weighting:
            lsds_weights = lsds_weights * hardness_prediction
            affs_weights = affs_weights * hardness_prediction

        lsd_loss = self._scaled_mse(lsds_prediction, lsds_target, lsds_weights)
        aff_loss = self._scaled_mse(affs_prediction, affs_target, affs_weights)

        seg_loss = lsd_loss + aff_loss
        seg_loss_copy_detached = seg_loss.clone().detach()

        if self.hardness_loss_formula == 'original_hl':
            # Mind the leading minus
            hardness_loss = - self.hardness_alpha * hardness_prediction * seg_loss_copy_detached
        elif self.hardness_loss_formula == 'mse':
            # Leading + because we want to minimize the MSE expression
            hardness_loss = + self.hardness_alpha * (hardness_prediction - seg_loss_copy_detached) ** 2
        else:
            raise ValueError(f'{self.hardness_loss_formula=} unkown.')

        total_loss = lsd_loss + aff_loss + hardness_loss
        if self.enable_mean_reduction:
            total_loss = torch.mean(total_loss)

        return total_loss


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
        num_fmaps_out=None,
        fmap_inc_factor=5,
        num_ds=2,
        constant_upsample = True,
        padding='valid',
        enable_batch_norm=False,
        **kwargs
):
    """Construct a funlib U-Net model as used in the LSD paper. Defaults to the paper's architecture config."""

    ds_fact=((2, 2, 2), (2, 2, 2))
    ds_fact = tuple([(2, 2, 2)] * num_ds)
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    ksu = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)

    return funlib_unet.UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        num_fmaps_out=num_fmaps_out,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        constant_upsample=constant_upsample,
        padding=padding,
        enable_batch_norm=enable_batch_norm,
        **kwargs
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

        self.unet = e3unet.UNet(
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


class SimpleHead(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gradient_detached=True,
            activation_class=nn.Sigmoid
    ):
        self.gradient_detached = gradient_detached
        self.conv = nn.Conv3d(in_channels, out_channels, 1)
        self.activation = activation_class()

    def forward(self, inp):
        if self.gradient_detached:
            inp = inp.detach()
        out = self.activation(self.conv(inp))
        return out


def get_decoder(decoder_variant, in_channels, out_channels):
    if decoder_variant == 'e3unet_s':
        return e3unet.UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=2,
            normalization='none',
            conv_mode='same'
        )
    elif decoder_variant == 'minimal':
        return nn.Conv3d(in_channels, out_channels, 1)
    elif decoder_variant == 'convs':
        # num_layers = int(decoder_variant.split('_')[-1])
        hidden_channels = 32
        return nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1), nn.ReLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(),
            nn.Conv3d(hidden_channels, out_channels, 3, padding=1)
        )
    else:
        raise ValueError(f'Invalid choice {decoder_variant=}')


@torch.vmap
def _vmap_norm(x: torch.Tensor) -> torch.Tensor:
    """Divides each batch element (along dim 0) by its individual sum across all non-batch dimensions."""
    return x / torch.sum(x)


# class HardnessHead(nn.Module):
#     """Hardness head forward pass of https://arxiv.org/abs/2305.08462

#     # Following https://github.com/Menoly-xin/Hardness-Level-Learning/blob/df48aa76c034ad61d4486b894581ec67822666ea/mmsegmentation/mmseg/models/segmentors/hl_encoder_decoder.py#L158-L164
#     """
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             gradient_detached=True,
#             # activation_class=nn.Sigmoid,
#             decoder_variant='e3unet_s',
#             hardness_c=0.1,
#             enable_original_batch_sum=False,
#     ):
#         self.gradient_detached = gradient_detached
#         self.hardness_c = hardness_c
#         self.decoder = get_decoder(decoder_variant, in_channels, out_channels)
#         # self.decoder = nn.Conv3d(in_channels, out_channels, 1)
#         # self.activation = activation_class()
#         self.enable_original_batch_sum = enable_original_batch_sum


#     def forward(self, inp):
#         if self.gradient_detached:
#             inp = inp.detach()
#         # Raw hardness level prediction  (D)
#         out = self.decoder(inp)
#         # Calculate numerator of H by applying sigmoid and adding constant c
#         out = torch.sigmoid(out) + self.hardness_c
#         # Normalize to sum of 1
#         # TODO: Investigate if the official implementation is wrong here:
#         #  IMO we shouldn't divide by the sum of the full batch here but do the
#         #  sum-normalization for each batch element separately.
#         if self.enable_original_batch_sum:
#             out /= out.sum()  # Official implementation
#         else:
#             # Apply sum scaling per batch index
#             out = _vmap_norm(out)
#         return out


def finalize_hardness(
        pre_hardness: torch.Tensor,
        hardness_c: float = 0.1,
        enable_original_batch_sum: bool = True,
) -> torch.Tensor:
    """
    Get normalized hardness map H from "pre-hardness" D https://arxiv.org/abs/2305.08462

    # Following https://github.com/Menoly-xin/Hardness-Level-Learning/blob/df48aa76c034ad61d4486b894581ec67822666ea/mmsegmentation/mmseg/models/segmentors/hl_encoder_decoder.py#L158-L164
    """

    # Raw hardness level prediction (D)
    # Calculate numerator of H by applying sigmoid and adding constant c
    out = torch.sigmoid(pre_hardness) + hardness_c
    # Normalize to sum of 1
    # TODO: Investigate if the official implementation is wrong here:
    #  IMO we shouldn't divide by the sum of the full batch here but do the
    #  sum-normalization for each batch element separately.
    if enable_original_batch_sum:
        out /= out.sum()  # Official implementation
    else:
        # Apply sum scaling per batch index
        out = _vmap_norm(out)
    return out

def build_mtlsdmodel(model_cfg):
    backbone_class = import_symbol(model_cfg.backbone.model_class)
    bb_init_kwargs = model_cfg.backbone.get('init_kwargs', {})
    backbone = backbone_class(**bb_init_kwargs)

    # hardness_head_class = import_symbol(model_cfg.hardness_head.model_class)
    # hh_init_kwargs = model_cfg.hardness_head.get('init_kwargs', {})
    # hardness_head = hardness_head_class(**hh_init_kwargs)

    lsd_fc = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 10, 1),
        nn.Sigmoid()
    )

    aff_fc = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 3, 1),
        nn.Sigmoid()
    )

    hardness_fc = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 1, 1),
        nn.Sigmoid()
    )

    # model = GeneralMtlsdModel(
    model = HardnessEnhancedMtlsdModel(
        backbone=backbone, lsd_fc=lsd_fc, aff_fc=aff_fc, hardness_fc=hardness_fc
    )
    return model


# TODO: Experiment with turning the lsd_head and aff_head into full proper decoder heads
# TODO: Think of better naming of heads - distinguish between big heads (decoders) and small heads (single conv layers)
class HardnessEnhancedMtlsdModel(torch.nn.Module):

    def __init__(
            self,
            backbone: torch.nn.Module,
            lsd_fc: torch.nn.Module,
            aff_fc: torch.nn.Module,
            hardness_fc: torch.nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.lsd_fc = lsd_fc
        self.aff_fc = aff_fc
        self.hardness_fc = hardness_fc
        self.register_module('backbone', backbone)
        self.register_module('lsd_fc', lsd_fc)
        self.register_module('aff_fc', aff_fc)
        self.register_module('hardness_fc', hardness_fc)

    def forward(self, input):
        # pass raw through unet
        outputs = self.backbone(input)
        assert len(outputs) == 2  # Hardcode for now - TODO: Re-enable support for other head output counts
        z, pre_hardness = outputs

        # pass output through fcs
        lsds = self.lsd_fc(z)
        affs = self.aff_fc(z)
        pre_hardness = self.hardness_fc(pre_hardness)

        hardness = finalize_hardness(pre_hardness)

        return lsds, affs, hardness

# class GeneralMtlsdModel(torch.nn.Module):

#     def __init__(
#             self,
#             backbone: torch.nn.Module,
#             lsd_head: torch.nn.Module,
#             aff_head: torch.nn.Module,
#             hardness_head: torch.nn.Module,
#     ):
#         super().__init__()

#         self.backbone = backbone
#         self.lsd_head = lsd_head
#         self.aff_head = aff_head
#         self.hardness_head = hardness_head
#         self.register_module('backbone', backbone)
#         self.register_module('lsd_head', lsd_head)
#         self.register_module('aff_head', aff_head)
#         self.register_module('hardness_head', hardness_head)

#     def forward(self, input):
#         # pass raw through unet
#         outputs = self.backbone(input)
#         z, pre_hardness = outputs

#         # pass output through heads
#         lsds = self.lsd_head(z)
#         affs = self.aff_head(z)

#         # z = z.detach() if DETACH else z
#         hardness = self.hardness_head(z)  # gradient detaching happens in head if configured

#         return lsds, affs, hardness
