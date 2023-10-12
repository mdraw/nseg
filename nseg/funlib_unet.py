# Based on https://github.com/funkelab/funlib.learn.torch/blob/636813e60d4/funlib/learn/torch/models/unet.py
# Changes:
# - Remove 4D convolution support
# - (WIP) Add types for torchscript compiler

# Experimental flags
USE_E3_CROPPING = False
USE_EXTRACTOR = False

import math
from typing import Optional, Sequence
import torch
import torch.nn as nn


class ConvPass(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            activation,
            padding='valid',
            enable_batch_norm=False):

        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []

        for kernel_size in kernel_sizes:

            self.dims = len(kernel_size)

            conv = {
                2: torch.nn.Conv2d,
                3: torch.nn.Conv3d,
            }[self.dims]

            if padding == 'same':
                pad = tuple(k//2 for k in kernel_size)
            else:
                pad = 0

            try:
                layers.append(
                    conv(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=pad))
                if enable_batch_norm:
                    bn = {
                        2: torch.nn.BatchNorm2d,
                        3: torch.nn.BatchNorm3d,
                    }[self.dims]
                    layers.append(bn(out_channels))
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):

        return self.conv_pass(x)


class Downsample(torch.nn.Module):

    def __init__(
            self,
            downsample_factor,
            ceil_mode=True
    ):

        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(
            downsample_factor,
            stride=downsample_factor,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):

        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d" % (
                        x.size(),
                        self.downsample_factor,
                        self.dims - d))

        return self.down(x)


class Upsample(torch.nn.Module):

    def __init__(
            self,
            scale_factor,
            mode='transposed_conv',
            in_channels=None,
            out_channels=None,
            crop_factor=None,
            next_conv_kernel_sizes=None,
            enable_pre_cropping=False):

        super(Upsample, self).__init__()

        if USE_E3_CROPPING:
            print('Warning, experimental changes in cropping active!')
        if USE_EXTRACTOR:
            print('Warning, experimental MedNext feature extractor active!')

        assert (crop_factor is None) == (next_conv_kernel_sizes is None), \
            "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes
        self.enable_pre_cropping = enable_pre_cropping

        self.dims = len(scale_factor)

        if mode == 'transposed_conv':

            up = {
                2: torch.nn.ConvTranspose2d,
                3: torch.nn.ConvTranspose3d
            }[self.dims]

            self.up = up(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor)

        else:

            self.up = torch.nn.Upsample(
                scale_factor=scale_factor,
                mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes: list[list[int]]):
        '''Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        '''

        shape = x.size()
        spatial_shape = shape[-self.dims:]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes)
            for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c)/f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n*f + c
            for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            assert all((
                    (t > c) for t, c in zip(
                        target_spatial_shape,
                        convolution_crop))
                ), \
                "Feature map with shape %s is too small to ensure " \
                "translation equivariance with factor %s and following " \
                "convolutions %s" % (
                    shape,
                    factor,
                    kernel_sizes)

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):

        g_up = self.up(g_out)

        if USE_E3_CROPPING:
            from elektronn3.models.resunet import autocrop

            f_cropped, g_cropped = autocrop(f_left, g_up)
            return torch.cat([f_cropped, g_up], dim=1)
        else:
            if self.enable_pre_cropping and self.next_conv_kernel_sizes is not None:
                g_cropped = self.crop_to_factor(
                    g_up,
                    self.crop_factor,
                    self.next_conv_kernel_sizes)
            else:
                g_cropped = g_up

            f_cropped = self.crop(f_left, g_cropped.size()[-self.dims:])
            return torch.cat([f_cropped, g_cropped], dim=1)




class UNet(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            activation='ReLU',
            fov=(1, 1, 1),
            voxel_size=(1, 1, 1),
            num_fmaps_out=None,
            num_heads=1,
            constant_upsample=False,
            padding='valid',
            ceil_mode=False,
            enable_batch_norm=False,
            enable_pre_cropping=False,
            active_head_ids: Optional[Sequence[int]] = None,
            detached_head_ids: Optional[Sequence[int]] = None,
    ):
        '''Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        This U-Net performs only "valid" convolutions, i.e., sizes of the
        feature maps decrease after each convolution. It will perfrom 4D
        convolutions as long as ``length`` is greater than 1. As soon as
        ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            in_channels:

                The number of input channels.

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            fov (optional):

                Initial field of view in physical units

            voxel_size (optional):

                Size of a voxel in the input data, in physical units

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            padding (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).

            active_head_ids:

                If `None`, compute and return all head outputs in `forward()`. If this is a
                Sequence[int], it is used as the index list of the head outputs that are computed and returned.
                Has no effect if num_heads == 1.
                Useful if not all heads are always required (as in https://arxiv.org/abs/2305.08462).

            detached_head_ids:

                If `None`, compute all gradients stay attached in `forward()`. If this is a
                Sequence[int], it is used as the index list of the head outputs
                that are detached from the encoder pathway.
                Intended for the hardness learning head of https://arxiv.org/abs/2305.08462.
        '''

        super(UNet, self).__init__()

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps
        self.enable_batch_norm = enable_batch_norm
        self.enable_pre_cropping = enable_pre_cropping

        self.active_head_ids = range(self.num_heads) if active_head_ids is None else active_head_ids
        self.detached_head_ids = [] if detached_head_ids is None else detached_head_ids


        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3, 3, 3), (3, 3, 3)]]*self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3, 3, 3), (3, 3, 3)]]*(self.num_levels - 1)

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(
                    f*ff
                    for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        if USE_EXTRACTOR:
            # MedNext-based feature extractor that outputs features of the original spatial shape,
            #  which the UNet is trained on instead of directly ingesting raw data.
            #  The in_channels parameter of the UNet should then be set to the feature dimensionality
            #  of the extractor instead of 1.
            from nseg.mednext import MedNeXt
            self.mednext = MedNeXt(
                in_channels=1,
                n_channels=in_channels,
                n_classes=in_channels,
                exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
                # exp_r = 2,
                kernel_size=3,  # Can test kernel_size
                deep_supervision=False,  # Can be used to test deep supervision
                do_res=False,  # Can be used to individually test residual connection
                do_res_up_down=False,
                # block_counts = [2,2,2,2,2,2,2,2,2],
                block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                checkpoint_style=None,
            )


        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvPass(
                in_channels
                if level == 0
                else num_fmaps*fmap_inc_factor**(level - 1),
                num_fmaps*fmap_inc_factor**level,
                kernel_size_down[level],
                activation=activation,
                padding=padding,
                enable_batch_norm=enable_batch_norm)
            for level in range(self.num_levels)
        ])
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList([
            Downsample(downsample_factors[level], ceil_mode=ceil_mode)
            for level in range(self.num_levels - 1)
        ])

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList([
            nn.ModuleList([
                Upsample(
                    downsample_factors[level],
                    mode='nearest' if constant_upsample else 'transposed_conv',
                    in_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    out_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    crop_factor=crop_factors[level],
                    next_conv_kernel_sizes=kernel_size_up[level],
                    enable_pre_cropping=self.enable_pre_cropping)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(
                    num_fmaps*fmap_inc_factor**level +
                    num_fmaps*fmap_inc_factor**(level + 1),
                    num_fmaps*fmap_inc_factor**level
                    if num_fmaps_out is None or level != 0
                    else num_fmaps_out,
                    kernel_size_up[level],
                    activation=activation,
                    padding=padding,
                    enable_batch_norm=enable_batch_norm)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            fs_out = [f_left] * len(self.active_head_ids)

        else:

            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            fs_right = [None] * len(self.active_head_ids)
            fs_out = [None] * len(self.active_head_ids)
            for h in self.active_head_ids:
                if h in self.detached_head_ids:
                    f_left = f_left.detach()  # TODO: Should we clone here in addition to detaching?
                # up, concat, and crop
                fs_right[h] = self.r_up[h][i](f_left, gs_out[h])
                # convolve
                fs_out[h] = self.r_conv[h][i](fs_right[h])

        return fs_out

    def forward(self, x):

        if USE_EXTRACTOR:
            x = self.mednext(x)

        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        if self.active_head_ids is not None and len(self.active_head_ids) == 1:
            return y[self.active_head_ids[0]]

        return y

    # def rec_encode(self, level, f_in):

    #     # index of level in layer arrays
    #     i = self.num_levels - level - 1

    #     # convolve
    #     f_left = self.l_conv[i](f_in)

    #     # end of recursion
    #     if level == 0:

    #         gs_out = [f_left]*self.num_heads

    #     else:

    #         # down
    #         g_in = self.l_down[i](f_left)

    #         # nested levels
    #         gs_out = self.rec_encode(level - 1, g_in)

    #     return gs_out

    # def encode(self, x):
    #     y = self.rec_encode(self.num_levels - 1, x)
    #     if self.num_heads == 1:
    #         return y[0]

    #     return y
