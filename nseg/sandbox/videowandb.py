import wandb
import numpy as np
import zarr
import os
from pathlib import Path

from typing import Dict, Optional
from wandb import util
from wandb.sdk.data_types._private import MEDIA_TMP
from wandb.sdk.lib import runid

class CVideo(wandb.Video):
    def encode(self) -> None:
        mpy = util.get_module(
            "moviepy.editor",
            required='wandb.Video requires moviepy and imageio when passing raw data.  Install with "pip install moviepy imageio"',
        )
        tensor = self._prepare_video(self.data)
        _, self._height, self._width, self._channels = tensor.shape

        # encode sequence of images into gif string
        clip = mpy.ImageSequenceClip(list(tensor), fps=self._fps)

        filename = os.path.join(
            MEDIA_TMP.name, runid.generate_id() + "." + self._format
        )
        kwargs = {"logger": None}
        clip.write_videofile(
            filename,
            codec='libx264',
            # ffmpeg_params=['-crf', '45'],
            **kwargs
        )
        self._set_file(filename, is_tmp=True)


def get_data(dchw_rgb=True, crop_raw=True):
    raw_path = Path('~/data/zebrafinch_msplit/validation/gt_z2834-2984_y5311-5461_x5077-5227.zarr').expanduser()
    data = zarr.open(str(raw_path), 'r')
    raw = np.array(data.volumes.raw)
    if crop_raw:
        raw = raw[100:250, 200:350, 200:350]
    lab = np.array(data.volumes.labels.neuron_ids)
    if dchw_rgb:
        raw = raw[:, None]
        raw = np.concatenate((raw, raw, raw), 1)
        lab = lab[:, None]
        # TODO: rgbify by color lut
        lab = np.concatenate((lab, lab, lab), 1)
    return raw, lab


def main():
    raw, lab = get_data()

    wandb.init(
        project="vtest",
        notes="Video logging experiment",
    )
    rawvid = CVideo(raw, fps=4, format='mp4')
    labvid = CVideo(lab, fps=4, format='mp4')
    wandb.log({'raw': rawvid, 'lab': labvid})

if __name__ == '__main__':
    main()