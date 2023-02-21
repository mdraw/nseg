import numpy as np
import torch
from funlib.learn.torch.models import UNet, ConvPass

# from elektronn3.models.unet import UNet as EU

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
        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            constant_upsample=constant_upsample,
            padding=padding)

        from elektronn3.models.unet import UNet as EU
        # self.unet = EU(
        #     in_channels=in_channels,
        #     out_channels=num_fmaps,
        #     dim=3,
        #     conv_mode='same',
        #     n_blocks=4,
        # )
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
