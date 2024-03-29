import daisy
import json
import logging
import numpy as np
import os
import sys
import time
import zarr
from funlib.segment.arrays import replace_values

from nseg.conf import NConf, DictConfig, hydra, unwind_dict


def extract_segmentation(
        fragments_file,
        fragments_dataset,
        edges_collection,
        threshold,
        block_size,
        seg_file,
        seg_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
        run_type=None,
        **_ # Gobble all other kwargs
):

    '''

    Args:

        fragments_file (``string``):

            Path to file (zarr/n5) containing fragments (supervoxels).

        fragments_dataset (``string``):

            Name of fragments dataset (e.g `volumes/fragments`)

        edges_collection (``string``):

            The name of the MongoDB database edges collection to use.

        threshold (``float``):

            The threshold to use for generating a segmentation.

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        seg_file (``string``):

            Path to file (zarr/n5) to write segmentation to.

        seg_dataset (``string``):

            Name of segmentation dataset (e.g `volumes/segmentation`).

        num_workers (``int``):

            How many workers to use when reading the region adjacency graph
            blockwise.

        roi_offset (array-like of ``int``, optional):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``, optional):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

        run_type (``string``, optional):

            Can be used to direct luts into directory (e.g testing, validation,
            etc).

    '''

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    total_roi = fragments.roi
    if roi_offset is not None:
        assert roi_shape is not None, "If roi_offset is set, roi_shape " \
                                      "also needs to be provided"
        total_roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

    read_roi = daisy.Roi((0,)*3, daisy.Coordinate(block_size))
    # read_roi = total_roi
    write_roi = read_roi

    logging.info("Preparing segmentation dataset...")
    segmentation = daisy.prepare_ds(
        seg_file,
        seg_dataset,
        total_roi,
        voxel_size=fragments.voxel_size,
        dtype=np.uint64,
        write_roi=write_roi,
        compressor={'id': 'zstd', 'level': 5},
    )

    lut_filename = f'seg_{edges_collection}_{int(threshold*100)}'

    lut_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    if run_type:
        lut_dir = os.path.join(lut_dir, run_type)
        logging.info(f"Run type set, using luts from {run_type} data")

    # lut = os.path.join(lut_dir, lut_filename + '.npz')
    lut_path = os.path.join(lut_dir, lut_filename + '.zarr')

    assert os.path.exists(lut_path), f"{lut_path} does not exist"

    logging.info("Reading fragment-segment LUT...")

    # lut = np.load(lut_path)['fragment_segment_lut']
    lut = zarr.open(lut_path, 'r')
    lut = np.array(lut)  # Load LUT into memory at once

    logging.info(f"Found {len(lut[0])} fragments in LUT")

    num_segments = len(np.unique(lut[1]))
    logging.info(f"Relabelling fragments to {num_segments} segments")

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        lambda b: segment_in_block(
            b,
            fragments_file,
            segmentation,
            fragments,
            lut),
        fit='shrink',
        num_workers=num_workers)

def segment_in_block(
        block,
        fragments_file,
        segmentation,
        fragments,
        lut):

    logging.debug("Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.write_roi)

    # # replace values, write to empty array
    # relabelled = np.zeros_like(fragments)
    relabelled = replace_values(fragments, lut[0], lut[1], inplace=True)

    segmentation[block.write_roi] = relabelled

    return 0  # return 0 to indicate success


@hydra.main(version_base='1.3', config_path='../conf/', config_name='inference_config')
def main(cfg: DictConfig) -> None:

    start = time.time()

    dict_cfg = NConf.to_container(cfg, resolve=True, throw_on_missing=True)

    dict_cfg = unwind_dict(dict_cfg, keys=['common', 'i6_extract_segmentation'])

    _hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    logging.info(f'Hydra run dir: {_hydra_run_dir}')
    dict_cfg['_hydra_run_dir'] = _hydra_run_dir
    logging.info(f'Config: {dict_cfg}')
    extract_segmentation(**dict_cfg)
    logging.info(f"Took {time.time() - start} seconds to extract segmentation from LUT")


if __name__ == "__main__":
    main()