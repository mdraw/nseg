from typing import Optional, Sequence, Any
import numpy as np
import torch
from torch import nn
import importlib
import gunpowder as gp

from elektronn3.models import unet as e3unet

from nseg import funlib_unet
from nseg.conf import NConf, DictConfig


# _default_loss_term_weights = {
#     'aff': 1.0,
#     'lsd': 1.0,
#     'bce': 0,
#     'hardness': 0.01,  # alpha value in https://arxiv.org/abs/2305.08462
# }


def as_torch_dict(
        arr_dict: dict[str, np.ndarray | torch.Tensor],
        device: torch.device | str = 'cpu',
        dtypes: Optional[dict[str, torch.dtype]] = None,
) -> dict[str, torch.Tensor]:
    dtypes = {} if dtypes is None else dtypes
    torch_dict = {
        k: torch.as_tensor(arr, device=device, dtype=dtypes.get(k))
        for k, arr in arr_dict.items()
    }
    return torch_dict


def as_np_dict(
        arr_dict: dict[str, torch.Tensor] | dict[str, np.ndarray],
        dtypes: Optional[dict[str, np.dtype]] = None,
) -> dict[str, np.ndarray]:
    dtypes = {} if dtypes is None else dtypes
    np_dict = {
        k: v.detach().cpu().numpy().astype(dtypes.get(k))
        if isinstance(v, torch.Tensor)
        else np.asarray(v, dtype=dtypes.get(k))
        for k, v in arr_dict.items()
    }
    return np_dict


class HardnessEnhancedLoss(torch.nn.Module):
    """Calculate segmentation losses (aff, lsd, ...) and hardness losses
    similar to https://arxiv.org/abs/2305.08462.
    """
    def __init__(
            self,
            loss_term_weights: dict[str, float],
            enable_hardness_weighting: bool = True,
            hardness_loss_formula: str = 'ohl',
    ):
        super().__init__()

        self.enable_hardness_weighting = enable_hardness_weighting
        self.hardness_loss_formula = hardness_loss_formula
        self.loss_term_weights = loss_term_weights
        if not self.loss_term_weights:
            raise ValueError('Non-empty loss_term_weights are required.')
        # self.crossentropy = nn.CrossEntropyLoss(
        #     reduction='none'
        # )


    @staticmethod
    def _scaled_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        scaled = weights * (pred - target) ** 2
        # # Reduce channel dim 1 to its mean but keep singleton channel dim. N,C,D,H,W -> N,1,D,H,W
        # scaled_mean = torch.mean(scaled, 1, keepdim=True)
        # # Reduce channel dim 1 to its mean. N,C,D,H,W -> N,D,H,W
        scaled_mean = torch.mean(scaled, 1)
        return scaled_mean

    @staticmethod
    def _compute_bce_map(pred_boundaries: torch.Tensor, gt_boundaries: torch.Tensor, bce_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce_loss_map = torch.nn.functional.cross_entropy(
            input=pred_boundaries,
            target=gt_boundaries,
            weight=bce_weights,
            reduction='none',
        )
        # # Reduce channel dim 1 to its mean but keep singleton channel dim. N,C,D,H,W -> N,1,D,H,W
        # bce_loss_map = torch.mean(bce_loss_map, dim=1, keepdim=True)
        # bce_loss_map.unsqueeze_(1)
        return bce_loss_map

    @staticmethod
    @torch.no_grad()
    def _compute_mask(*weights, mode='or') -> torch.Tensor:
        """Combine weights into a mask by sequential logical "and" or "or" of a all mask weight channels.
        Voxels that have a weight of <= 0 in any (`mode='or'`) / all (`mode='and'`) channels are masked out."""
        if mode == 'and':
            mask_init = torch.ones_like
            bin_op = torch.logical_and
        elif mode == 'or':
            mask_init = torch.zeros_like
            bin_op = torch.logical_or
        else:
            raise ValueError(f'Invalid bin_op mode {mode}')

        _w1 = weights[0][:, 0]
        mask = mask_init(_w1, dtype=torch.bool)
        for weight in weights:
            for c in range(weight.shape[1]):
                mask = bin_op(mask, weight[:, c] > 0)

        return mask

    def _apply_mask(self, loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if torch.any(mask):
            # Only keep elements that are not masked out, flatten to 1D
            masked_loss = torch.masked_select(loss_map, mask)
        else:
            # mask is 0 everywhere -> masked_select would return a 0-element tensor,
            # so we just skip this and use masked_loss instead. -> Mean loss is 0.
            masked_loss = loss_map * 0.
            print('all masked out')
        return masked_loss

    def _compute_hardness_loss_map(
            self, pred_hardness: Optional[torch.Tensor], seg_loss_map: torch.Tensor
    ) -> torch.Tensor:
        if pred_hardness is None:
            return torch.zeros_like(seg_loss_map)
        # Detach from graph so the hardness gradient does not backpropagate to the
        #  backbone and the segmentation head (we don't want to risk encouraging the
        #  model to trade off segmentation quality for better hardness prediction).
        seg_loss_map_detached = seg_loss_map.clone().detach()

        if self.hardness_loss_formula == 'ohl':
            # Mind the leading minus
            hardness_loss = - pred_hardness * seg_loss_map_detached
        elif self.hardness_loss_formula == 'mse':
            # Leading + because we want to minimize the MSE expression
            hardness_loss = + (pred_hardness - seg_loss_map_detached) ** 2
        else:
            raise ValueError(f'{self.hardness_loss_formula=} unkown.')
        hardness_loss *= self.loss_term_weights['hardness']
        return hardness_loss

    def _apply_hardness_weighting(
            self, loss_map: torch.Tensor, hardness_prediction: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.enable_hardness_weighting and hardness_prediction is not None:
            # hardness_scaling = hardness_prediction.squeeze(1)
            # print(hardness_prediction.shape, hardness_scaling.shape)
            loss_map = hardness_prediction * loss_map
            # print(loss_map.shape)
        return loss_map

    def compute_seg_loss_maps(
            self,
            pred_lsds: torch.Tensor,
            gt_lsds: torch.Tensor,
            lsds_weights: torch.Tensor,
            pred_affs: torch.Tensor,
            gt_affs: torch.Tensor,
            affs_weights: torch.Tensor,
            pred_hardness: Optional[torch.Tensor],
            pred_boundaries: torch.Tensor,
            gt_boundaries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seg_loss_maps = {}
        if self.loss_term_weights.get('lsd', 0) != 0:
            lsd_loss_map = self._scaled_mse(pred_lsds, gt_lsds, lsds_weights)
            lsd_loss_map = self._apply_hardness_weighting(lsd_loss_map, pred_hardness)
            seg_loss_maps['lsd'] = lsd_loss_map
        if self.loss_term_weights.get('aff', 0) != 0:
            aff_loss_map = self._scaled_mse(pred_affs, gt_affs, affs_weights)
            aff_loss_map = self._apply_hardness_weighting(aff_loss_map, pred_hardness)
            seg_loss_maps['aff'] = aff_loss_map
        if self.loss_term_weights.get('bce', 0) != 0:
            bce_loss_map = self._compute_bce_map(pred_boundaries, gt_boundaries)#, boundary_weights)
            bce_loss_map = self._apply_hardness_weighting(bce_loss_map, pred_hardness)
            seg_loss_maps['bce'] = bce_loss_map

        # for k, v in seg_loss_maps.items():
        #     print(f'{k}: {v.shape}')
        return seg_loss_maps

    def _combine_seg_loss_maps(self, seg_loss_maps: dict[str, torch.Tensor], exclude: Optional[Sequence[str]] = None) -> torch.Tensor:
        combined_seg_loss_map = 0.
        for loss_name, loss_map in seg_loss_maps.items():
            if exclude is None or loss_name not in exclude:
                weight = self.loss_term_weights[loss_name]
                combined_seg_loss_map += weight * loss_map
        return combined_seg_loss_map


    def forward(
            self,
            pred_lsds: torch.Tensor,
            gt_lsds: torch.Tensor,
            lsds_weights: torch.Tensor,
            pred_affs: torch.Tensor,
            gt_affs: torch.Tensor,
            affs_weights: torch.Tensor,
            pred_hardness: Optional[torch.Tensor],
            pred_boundaries: torch.Tensor,
            gt_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        seg_loss_maps = self.compute_seg_loss_maps(
            pred_lsds=pred_lsds,
            gt_lsds=gt_lsds,
            lsds_weights=lsds_weights,
            pred_affs=pred_affs,
            gt_affs=gt_affs,
            affs_weights=affs_weights,
            pred_hardness=pred_hardness,
            pred_boundaries=pred_boundaries,
            gt_boundaries=gt_boundaries,
        )
        total_seg_loss_map = self._combine_seg_loss_maps(seg_loss_maps)

        # Disregard LSD because of masking issues
        total_hardness_relevant_seg_loss_map = self._combine_seg_loss_maps(seg_loss_maps, exclude=['lsd'])

        # hardness_loss_map = self._compute_hardness_loss_map(pred_hardness, total_seg_loss_map)
        hardness_loss_map = self._compute_hardness_loss_map(pred_hardness, total_hardness_relevant_seg_loss_map)

        enable_oob_masking = False
        if enable_oob_masking:
            total_loss_map = total_seg_loss_map + hardness_loss_map
            mask = self._compute_mask(lsds_weights, affs_weights)
            masked_total_loss = self._apply_mask(total_loss_map, mask)
            total_loss_scalar = torch.mean(masked_total_loss)
        else:
            total_loss_map = total_seg_loss_map + hardness_loss_map
            total_loss_scalar = torch.mean(total_loss_map)
        return total_loss_scalar


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


@torch.inference_mode()
def compute_loss_maps(
        loss_module: HardnessEnhancedLoss,
        batch: gp.batch.Batch,
        loss_inputs_gpkeys: dict[Any, gp.ArrayKey],
        device: torch.device | str = 'cpu',
) -> dict[str, np.ndarray]:
    # loss_input_arraydict = loss_input_arraydict.values()
    arr_keys = {akey.identifier.lower(): akey for akey in loss_inputs_gpkeys.values()}
    arr_dict = {
        name: batch[arr_key].data for name, arr_key in arr_keys.items()
    }
    loss_inputs = as_torch_dict(arr_dict=arr_dict, device=device)

    seg_loss_maps = loss_module.compute_seg_loss_maps(
        **loss_inputs
    )
    total_seg_loss_map = loss_module._combine_seg_loss_maps(seg_loss_maps)
    pred_hardness = loss_inputs['pred_hardness']
    hardness_loss_map = loss_module._compute_hardness_loss_map(pred_hardness, total_seg_loss_map)

    loss_maps = {**seg_loss_maps, 'total_seg_loss': total_seg_loss_map, 'hardness': hardness_loss_map}
    np_loss_maps = as_np_dict(loss_maps)

    return np_loss_maps


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


# TODO: Don't use constant upsampling
def get_funlib_unet(
        in_channels=1,
        num_fmaps=14,
        num_fmaps_out=None,
        fmap_inc_factor=5,
        ds_fact=((2, 2, 2), (2, 2, 2)),
        # ds_fact=((1, 3, 3), (1, 3, 3), (3, 3, 3)),
        constant_upsample=True,
        padding='valid',
        enable_batch_norm=False,
        **kwargs
):
    """Construct a funlib U-Net model as used in the LSD paper. Defaults to the paper's architecture config."""

    ds_fact = [tuple(f) for f in ds_fact]  # Tuples (not lists) are needed for down/up-sampling
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


@torch.vmap
def _vmap_sum_norm(x: torch.Tensor) -> torch.Tensor:
    """Divides each batch element (along dim 0) by its individual sum across all non-batch dimensions."""
    return x / torch.sum(x)


@torch.vmap
def _vmap_sum_numel_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.sum(x) / x.numel())


@torch.vmap
def _vmap_sum_10k_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.sum(x) / 10_000.)


def finalize_hardness(
        pre_hardness: torch.Tensor,
        hardness_c: float = 0.1,
        normalization_mode = 'sum_numel',
) -> torch.Tensor:
    """
    Get normalized hardness map H from "pre-hardness" D https://arxiv.org/abs/2305.08462

    # Following https://github.com/Menoly-xin/Hardness-Level-Learning/blob/df48aa76c034ad61d4486b894581ec67822666ea/mmsegmentation/mmseg/models/segmentors/hl_encoder_decoder.py#L158-L164
    """

    # Raw hardness level prediction (D)
    # Calculate numerator of H by applying sigmoid and adding constant c
    out = torch.sigmoid(pre_hardness) + hardness_c
    if normalization_mode == 'none':
        pass
    elif normalization_mode == 'original_batchsum':
        # Normalize to sum of 1
        # The official implementation is apparently wrong here:
        #  IMO we shouldn't divide by the sum of the full batch here but do the
        #  sum-normalization for each batch element separately.
        out /= out.sum()  # Official implementation
    elif normalization_mode == 'batchsum':
        # Apply sum scaling per batch index
        out = _vmap_sum_norm(out)
    elif normalization_mode == 'sum_numel':
        out = _vmap_sum_numel_norm(out)
    elif normalization_mode == 'sum_1M':
        out = _vmap_sum_10k_norm(out)

    # Squeeze singleton channel dimension -> N, D, H, W
    out.squeeze_(1)

    return out

def build_mtlsdmodel(model_cfg):
    backbone_class = import_symbol(model_cfg.backbone.model_class)
    bb_init_kwargs = model_cfg.backbone.get('init_kwargs', {})
    if isinstance(bb_init_kwargs, DictConfig):
        bb_init_kwargs = NConf.to_container(bb_init_kwargs, resolve=True, throw_on_missing=True)
    backbone = backbone_class(**bb_init_kwargs)

    finalize_hardness_kwargs = model_cfg.get('finalize_hardness_kwargs', {})

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
        # nn.Sigmoid()  # sigmoid is already applied in finalize_hardness afterwards!
    )

    boundary_fc = nn.Sequential(
        nn.Conv3d(model_cfg.backbone.num_fmaps, 2, 1),
        # nn.Softmax()  # softmax is integrated in nn.CrossEntropyLoss
    )

    # model = GeneralMtlsdModel(
    model = HardnessEnhancedMtlsdModel(
        backbone=backbone,
        lsd_fc=lsd_fc,
        aff_fc=aff_fc,
        boundary_fc=boundary_fc,
        hardness_fc=hardness_fc,
        finalize_hardness_kwargs=finalize_hardness_kwargs,
    )
    return model


# TODO: Experiment with turning the lsd_head and aff_head into full proper decoder heads
class HardnessEnhancedMtlsdModel(torch.nn.Module):

    def __init__(
            self,
            backbone: torch.nn.Module,
            lsd_fc: torch.nn.Module,
            aff_fc: torch.nn.Module,
            boundary_fc: torch.nn.Module,
            hardness_fc: torch.nn.Module,
            finalize_hardness_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.lsd_fc = lsd_fc
        self.aff_fc = aff_fc
        self.boundary_fc = boundary_fc
        self.hardness_fc = hardness_fc
        self.finalize_hardness_kwargs = {} if finalize_hardness_kwargs is None else finalize_hardness_kwargs
        self.register_module('backbone', backbone)
        self.register_module('lsd_fc', lsd_fc)
        self.register_module('aff_fc', aff_fc)
        self.register_module('boundary_fc', boundary_fc)
        self.register_module('hardness_fc', hardness_fc)

    def forward(self, input):
        # pass raw through unet
        outputs = self.backbone(input)
        assert len(outputs) == 2  # Hardcode for now - TODO: Re-enable support for other head output counts
        z, pre_hardness = outputs

        # pass output through fcs
        lsds = self.lsd_fc(z)
        affs = self.aff_fc(z)
        boundaries = self.boundary_fc(z)
        pre_hardness = self.hardness_fc(pre_hardness)
        hardness = finalize_hardness(pre_hardness, **self.finalize_hardness_kwargs)

        model_outputs = {
            'pred_lsds': lsds,
            'pred_affs': affs,
            'pred_boundaries': boundaries,
            'pred_hardness': hardness,
        }

        return model_outputs

        # return lsds, affs, boundaries, hardness

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
