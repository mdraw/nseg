# https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/segment.ipynb

import logging
from pathlib import Path
from typing import Optional
import torch
import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np
import waterz
import zarr
from funlib.evaluate import rand_voi
from scipy.ndimage import label
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

from lsd.train.local_shape_descriptor import get_local_shape_descriptors
from lsd.train.gp import AddLocalShapeDescriptor


from nseg.shared import create_lut, build_mtlsdmodel, WeightedMSELoss, import_symbol
from nseg.gpx.gp_predict import Predict
from nseg.gpx.gp_sources import ZarrSource
from nseg.gpx.gp_scan import Scan
from nseg.gpx.gp_boundaries import ArgMax, SoftMax, Take

from nseg.conf import DictConfig, hydra
from nseg.eval_utils import CubeEvalResult


# Agglomeration method specs for waterz. From https://github.com/funkelab/lsd/blob/fc812095328ffe6640b2b3bec77230b384e8687f/lsd/tutorial/scripts/workers/agglomerate_worker.py#L29
waterz_merge_function = {
    'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
    'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
    'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
    'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
    'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
    'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
    'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
    'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
    'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
    'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
    'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
}


def spatial_center_crop_nd(large, small, ndim_spatial=2):
    """Return center-cropped version of `large` image, with spatial dims cropped to spatial shape of `small` image"""
    ndim_nonspatial = large.ndim - ndim_spatial

    # Get spatial shapes of the two input images
    sha = np.array(large.shape[-ndim_spatial:])
    shb = np.array(small.shape[-ndim_spatial:])

    assert np.all(sha >= shb)

    # Compute the starting and ending indices for each dimension
    lo = (sha - shb) // 2
    hi = lo + shb

    # Calculate slice indices
    nonspatial_slice = [  # Slicing all available content in these dims.
        slice(0, None) for _ in range(ndim_nonspatial)
    ]
    spatial_slice = [  # Slice only the content within the coordinate bounds
        slice(lo[i], hi[i]) for i in range(ndim_spatial)
    ]
    full_slice = tuple(nonspatial_slice + spatial_slice)

    # Perform the center crop
    cropped = large[full_slice]

    return cropped, small


def cr(a, b, ndim_spatial=2):
    # Number of nonspatial axes (like the C axis). Usually this is one
    ndim_nonspatial = a.ndim - ndim_spatial
    # Get the shapes of the two input images
    sha = np.array(a.shape)
    shb = np.array(b.shape)
    # Compute the minimum shape along each dimension
    min_shape = np.minimum(sha, shb)
    # Compute the starting and ending indices for each dimension
    lo = (sha - min_shape) // 2
    hi = lo + min_shape

    # Calculate slice indices
    nonspatial_slice = [  # Slicing all available content in these dims.
        slice(0, a.shape[i]) for i in range(ndim_nonspatial)
    ]
    spatial_slice = [  # Slice only the content within the coordinate bounds
        slice(lo[i], hi[i]) for i in range(ndim_spatial)
    ]
    full_slice = tuple(nonspatial_slice + spatial_slice)
    a_cropped = a[full_slice]
    if b is None:
        return a_cropped, b

    if b.ndim == a.ndim - 1:  # a: (C, [D,], H, W), b: ([D,], H, W)
        full_slice = full_slice[1:]  # Remove C axis from slice because b doesn't have it
    b_cropped = b[full_slice]

    return a_cropped, b_cropped



def center_crop(a, b):  # todo: from secgan
    import math

    a_dim = list(a.shape)
    b_dim = list(b.shape)

    for i in range(-1, -4, -1):
        # take the last 3 dimensions. a and b don't need to have the same number of dimensions
        # (i.e. only one has channels)
        if a_dim[i] != b_dim[i]:
            if a_dim[i] > b_dim[i]:
                crop_val = (a_dim[i] - b_dim[i]) / 2
            else:
                crop_val = (b_dim[i] - a_dim[i]) / 2

            left = math.floor(crop_val)
            right = -math.ceil(crop_val)

            if a_dim[i] > b_dim[i]:
                slice_window = tuple(
                    [slice(None)] * (len(a.shape) - 3)
                    + (
                        [
                            slice(left, right) if i == j else slice(None)
                            for j in range(-1, -4, -1)
                        ][::-1]
                    )
                )
                a = a[slice_window]
            else:
                slice_window = tuple(
                    [slice(None)] * (len(b.shape) - 3)
                    + (
                        [
                            slice(left, right) if i == j else slice(None)
                            for j in range(-1, -4, -1)
                        ][::-1]
                    )
                )
                b = b[slice_window]

    return a, b


def predict_unlabeled_zarr(cfg, raw_path: Path | str, checkpoint_path: Optional[Path | str] = None) -> None:
    """Run scan inference on unlabeled data (or labeled data where labels should not be used).
    Directly returns the outputs as numpy arrays."""

    voxel_size = gp.Coordinate(cfg.dataset.voxel_size)
    # Prefer ev_inp_shape if specified, use regular inp_shape otherwise
    input_shape = gp.Coordinate(
        cfg.model.backbone.get('ev_inp_shape', cfg.model.backbone.inp_shape)
    )
    input_size = input_shape * voxel_size
    offset = gp.Coordinate(cfg.model.backbone.offset)
    output_shape = input_shape - offset
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_boundaries = gp.ArrayKey('PRED_BOUNDARIES')
    pred_hardness = gp.ArrayKey('PRED_HARDNESS')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)
    scan_request.add(pred_boundaries, output_size)
    scan_request.add(pred_hardness, output_size)

    # labels = gp.ArrayKey('LABELS')
    # scan_request.add(labels, output_size)

    # TODO: Investigate input / output shapes w.r.t. offsets - output sizes don't always match each other
    context = (input_size - output_size) / 2

    source_data_dict = {
        # raw: 'cfg.dataset.raw_name',
        raw: 'volumes/raw',
        # labels: cfg.dataset.gt_name,
    }
    source_array_specs = {
        raw: gp.ArraySpec(interpolatable=True),
        # labels: gp.ArraySpec(interpolatable=False),
    }

    print(raw_path)
    source = ZarrSource(
        str(raw_path),
        source_data_dict,
        source_array_specs,
    )

    source += gp.Unsqueeze([raw])

    # if cfg.dataset.labels_padding is not None:
    #     labels_padding = gp.Coordinate(cfg.dataset.labels_padding)
    #     source += gp.Pad(labels, labels_padding)

    with gp.build(source):
        if cfg.eval.roi_shape is None:
            source_roi = source.spec[raw].roi
            total_input_roi = source_roi
            total_output_roi = source_roi.grow(-context, -context)
        else:
            _off = voxel_size * 0  # ~0 is not intuitive but it works? The ROI shape is apparently auto-centered~. Edit: Apparently not...
            # _off = voxel_size *
            raise NotImplementedError
            _sha = voxel_size * tuple(cfg.eval.roi_shape)
            total_output_roi = gp.Roi(offset=_off, shape=_sha)
            total_input_roi = total_output_roi.grow(context, context)
            # total_input_roi = gp.Roi(offset=_off, shape=_sha)
            # total_output_roi = total_input_roi.grow(-context, -context)

        # _gtlabel_shape = voxel_size * (8, 8, 8)  # TODO!
        # _gtlabel_off = voxel_size * (250, 250, 250)
        # label_roi = gp.Roi(offset=_gtlabel_off, shape=_gtlabel_shape)


    # model = get_mtlsdmodel()  # MtlsdModel()
    model = build_mtlsdmodel(cfg.model)

    # set model to eval mode
    model.eval()

    if checkpoint_path is None:  # Fall back to cfg checkpoint
        checkpoint_path = cfg.eval.checkpoint

    # add a predict node
    predict = Predict(
        model=model,
        checkpoint=checkpoint_path,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs,
            2: pred_boundaries,
            3: pred_hardness,
        }
    )

    # this will scan in chunks equal to the input/output sizes of the respective arrays
    scan = Scan(
        scan_request,
        num_workers=0
    )

    pipeline = source
    pipeline += gp.Normalize(raw)
    # pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += predict

    pipeline += ArgMax(pred_boundaries)

    pipeline += gp.Squeeze([
        raw,
        pred_lsds,
        pred_affs,
        pred_boundaries,
        # pred_hardness,
    ])


    # Scale to uint8 range for zarr storage
    pipeline += gp.IntensityScaleShift(pred_affs, 255, 0)
    pipeline += gp.IntensityScaleShift(pred_lsds, 255, 0)
    pipeline += gp.IntensityScaleShift(pred_boundaries, 255, 0)

    # Uncomment to make very small values visible
    # pipeline += gp.IntensityScaleShift(pred_hardness, 10_000_000, 0)

    # from numcodecs import LZ4
    # compressor = LZ4()

    gp_write_zarr = True
    if gp_write_zarr:
        out_file = str(Path(cfg.eval.result_zarr_root) / 'zres_nw0.zarr')
        f = zarr.open(out_file, 'w')

        out_datasets = {
            'raw': {'out_dims': 1, 'out_dtype': 'uint8'},
            'pred_affs': {'out_dims': 3, 'out_dtype': 'uint8'},
            'pred_lsds': {'out_dims': 10, 'out_dtype': 'uint8'},
            'pred_boundaries': {'out_dims': 1, 'out_dtype': 'uint8'},
            'pred_hardness': {'out_dims': 1, 'out_dtype': 'float32'},
        }

        for ds_name, data in out_datasets.items():
            ds = f.create_dataset(
                ds_name,
                shape=[data['out_dims']] + list(total_output_roi.get_shape() / voxel_size),
                dtype=data['out_dtype'],
                chunks=(data['out_dims'], 256, 256, 256)
                # chunks=output_size // 2,
                # compressor=compressor,
            )
            ds.attrs['resolution'] = voxel_size
            ds.attrs['offset'] = total_output_roi.get_offset()
        zarr_write = gp.ZarrWrite(
            output_dir=str(Path(cfg.eval.result_zarr_root)),
            output_filename='zres.zarr',
            dataset_names={
                raw: 'raw',
                pred_affs: 'pred_affs',
                pred_lsds: 'pred_lsds',
                pred_boundaries: 'pred_boundaries',
                pred_hardness: 'pred_hardness',
            }
        )
        pipeline += zarr_write

    pipeline += scan

    # request an empty batch from scan
    predict_request = gp.BatchRequest()

    # # this lets us know to process the full image. we will scan over it until it is done
    # predict_request.add(raw, total_input_roi.get_end())
    # predict_request.add(pred_lsds, total_output_roi.get_end())
    # predict_request.add(pred_affs, total_output_roi.get_end())
    # predict_request.add(pred_hardness, total_output_roi.get_end())

    # predict_request.add(labels, total_output_roi.get_end())

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)



# TODO: For roi-constrained inference, try using something like this: https://github.com/funkelab/lsd/blob/fc812095328ffe6640b2b3bec77230b384e8687f/lsd/tutorial/scripts/01_predict_blockwise.py#L91-L100

# TODO: Add option to use empty requests and directly write outputs to zarr, see https://funkelab.github.io/gunpowder/tutorial_simple_pipeline.html#predicting-on-a-whole-image
def predict_unlabeled(cfg, raw_path: Path | str, checkpoint_path: Optional[Path | str] = None) -> tuple[np.ndarray, ...]:
    """Run scan inference on unlabeled data (or labeled data where labels should not be used).
    Directly returns the outputs as numpy arrays."""

    output_cfg = cfg.eval.output_cfg

    output_names = list(output_cfg.keys())

    voxel_size = gp.Coordinate(cfg.dataset.voxel_size)
    # Prefer ev_inp_shape if specified, use regular inp_shape otherwise
    input_shape = gp.Coordinate(
        cfg.model.backbone.get('ev_inp_shape', cfg.model.backbone.inp_shape)
    )
    input_size = input_shape * voxel_size
    offset = gp.Coordinate(cfg.model.backbone.offset)
    output_shape = input_shape - offset
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    output_arrkeys = {k: gp.ArrayKey(k.upper()) for k in output_names}

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)

    for arrkey in output_arrkeys.values():
        scan_request.add(arrkey, output_size)

    # labels = gp.ArrayKey('LABELS')
    # scan_request.add(labels, output_size)

    # TODO: Investigate input / output shapes w.r.t. offsets - output sizes don't always match each other
    context = (input_size - output_size) / 2

    source_data_dict = {
        raw: cfg.dataset.raw_name,
        # labels: cfg.dataset.gt_name,
    }
    source_array_specs = {
        raw: gp.ArraySpec(interpolatable=True),
        # labels: gp.ArraySpec(interpolatable=False),
    }

    source = ZarrSource(
        str(raw_path),
        datasets=source_data_dict,
        array_specs=source_array_specs,
    )

    source += gp.Unsqueeze([raw])

    # if cfg.dataset.labels_padding is not None:
    #     labels_padding = gp.Coordinate(cfg.dataset.labels_padding)
    #     source += gp.Pad(labels, labels_padding)

    with gp.build(source):
        if cfg.eval.roi_shape is None:
            source_roi = source.spec[raw].roi
            total_input_roi = source_roi
            total_output_roi = source_roi.grow(-context, -context)
        else:
            _off = voxel_size * 0  # ~0 is not intuitive but it works? The ROI shape is apparently auto-centered~. Edit: Apparently not...
            # _off = voxel_size *
            raise NotImplementedError
            _sha = voxel_size * tuple(cfg.eval.roi_shape)
            total_output_roi = gp.Roi(offset=_off, shape=_sha)
            total_input_roi = total_output_roi.grow(context, context)
            # total_input_roi = gp.Roi(offset=_off, shape=_sha)
            # total_output_roi = total_input_roi.grow(-context, -context)

        # _gtlabel_shape = voxel_size * (8, 8, 8)  # TODO!
        # _gtlabel_off = voxel_size * (250, 250, 250)
        # label_roi = gp.Roi(offset=_gtlabel_off, shape=_gtlabel_shape)


    # model = get_mtlsdmodel()  # MtlsdModel()
    model = build_mtlsdmodel(cfg.model)

    # set model to eval mode
    model.eval()

    if checkpoint_path is None:  # Fall back to cfg checkpoint
        checkpoint_path = cfg.eval.checkpoint

    # _predict_outputs = {output_cfg[n]['idx']: output_arrkeys[n] for n in output_names}
    _predict_outputs = {n: output_arrkeys[n] for n in output_names}

    # add a predict node
    predict = Predict(
        model=model,
        checkpoint=checkpoint_path,
        inputs={
            'input': raw
        },
        outputs=_predict_outputs,
    )

    # this will scan in chunks equal to the input/output sizes of the respective arrays
    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)
    # pipeline += gp.Unsqueeze([raw])

    pipeline += gp.IntensityScaleShift(raw, 2, -1)  # Rescale/shift to training value range

    pipeline += gp.Stack(1)

    pipeline += predict

    if 'pred_boundaries' in output_names:
        boundary_arrkey = output_arrkeys['pred_boundaries']
        # pipeline += ArgMax(boundaries)
        pipeline += SoftMax(boundary_arrkey)
        pipeline += Take(boundary_arrkey, 1, 1)  # Take channel 1

    outputs_to_squeeze = [
        output_arrkeys[k]
        for k, v in output_cfg.items()
        if v['squeeze']
    ]
    pipeline += gp.Squeeze([
        raw,
        *outputs_to_squeeze
    ])

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)  # Rescale/shift back for snapshot outputs

    pipeline += scan

    predict_request = gp.BatchRequest()

    # this lets us know to process the full image. we will scan over it until it is done
    predict_request.add(raw, total_input_roi.get_end())
    for arrkey in output_arrkeys.values():
        predict_request.add(arrkey, total_output_roi.get_end())

    # predict_request.add(pred_lsds, total_output_roi.get_end())
    # predict_request.add(pred_affs, total_output_roi.get_end())
    # predict_request.add(pred_hardness, total_output_roi.get_end())
    # predict_request.add(labels, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    batch_ret = {'raw': batch[raw].data}
    for output_name, arrkey in output_arrkeys.items():
        batch_ret[output_name] = batch[arrkey].data

    return batch_ret

    # return batch[raw].data, batch[pred_lsds].data, batch[pred_affs].data, batch[pred_boundaries].data batch[pred_hardness].data  #, batch[labels].data


def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret


def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=10,
        fragment_threshold=0.5
):
    mean_affs = np.mean(affs, axis=0)
    boundary_mask = mean_affs > (fragment_threshold * max_affinity_value)
    boundary_distances = distance_transform_edt(boundary_mask)

    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)

    return ret[0], boundary_distances


# Based on the configuration resulting from https://github.com/funkelab/lsd/blob/a4955739b5defb252db2d466647b47db940bc157/README.md#agglomerate
# Using LSD's default merge function "hist_quant_75" from https://github.com/funkelab/lsd/blob/fc812095328ffe6640b2b3bec77230b384e8687f/lsd/tutorial/scripts/workers/agglomerate_worker.py#L36
# Other params are based on the waterz call here (minus the ones that are irrelevant here): https://github.com/funkelab/lsd/blob/399e901ab9462da6119addceaa8a34aee7bcde2d/lsd/post/parallel_aff_agglomerate.py#L130
def get_mf_segmentation(affinities, waterz_threshold=1.0, fragment_threshold=0.5, merge_function_name: str = 'hist_quant_75', gt_seg=None):
    fragments, boundary_distances = watershed_from_affinities(affinities, fragment_threshold=fragment_threshold)  # [0]

    merge_function_spec = waterz_merge_function[merge_function_name]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),  # .squeeze(1),
        fragments=np.copy(fragments),  # .squeeze(0),
        thresholds=[waterz_threshold],
        scoring_function=merge_function_spec,
        discretize_queue=256,
    )
    segmentation = next(generator)

    return segmentation, fragments, boundary_distances


def get_segmentation(affinities, waterz_threshold=0.043, fragment_threshold=0.5, gt_seg=None):
    fragments, boundary_distances = watershed_from_affinities(affinities, fragment_threshold=fragment_threshold)  # [0]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),  # .squeeze(1),
        fragments=np.copy(fragments),  # .squeeze(0),
        thresholds=[waterz_threshold],
    )
    segmentation = next(generator)

    return segmentation, fragments, boundary_distances


def sweep_segmentation_threshs(affinities, waterz_thresholds, fragment_threshold=0.5, merge_function_name='hist_quant_75', gt_seg=None):
    fragments, boundary_distances = watershed_from_affinities(affinities, fragment_threshold=fragment_threshold)  # [0]

    merge_function_spec = waterz_merge_function[merge_function_name]

    vois = []

    # threshs = np.arange(0.0, 2.0, 0.1)
    # threshs = np.arange(0.043, 0.43, 0.0001)

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),  # .squeeze(1),
        fragments=np.copy(fragments),  # .squeeze(0),
        thresholds=waterz_thresholds,
        scoring_function=merge_function_spec,
        discretize_queue=256,
    )
    for thresh, segmentation in zip(waterz_thresholds, generator, strict=True):
        segmentation, gt_seg = center_crop(segmentation, gt_seg)

        rand_voi_report = rand_voi(
            gt_seg,
            segmentation,  # segment_ids,
            return_cluster_scores=False
        )
        voi = rand_voi_report["voi_split"] + rand_voi_report["voi_merge"]
        # print(f'VOI: {voi}')
        vois.append(voi)

    for t, v in zip(waterz_thresholds, vois, strict=True):
        print(f'{t=}, {v=}')

    _min_voi_idx = np.argmin(vois)
    min_voi = vois[_min_voi_idx]
    argmin_thresh = waterz_thresholds[_min_voi_idx]

    print(f'Optimum threshold: {argmin_thresh}, with VOI {min_voi}')

    # TODO: Don't print, but return list of (argmin_thresh, min_voi) pairs so they can be globally evaluated across cubes

    return argmin_thresh, min_voi
    # return segmentation, fragments, boundary_distances


def get_scoring_segmentation(affinities, fragment_threshold=0.5, epsilon_agglomerate=0, gt_seg=None):
    fragments, boundary_distances = watershed_from_affinities(affinities, fragment_threshold=fragment_threshold)  # [0]

    # Based on parralel_fragments.watershed_in_block(), see
    #  https://github.com/funkelab/lsd/blob/b6aee2fd0c87bc70a52ea77e85f24cc48bc4f437/lsd/post/parallel_fragments.py#L149
    # TODO: Haven't managed to get good results from this yet. Maybe some arguments have to be changed? scoring_function?
    assert epsilon_agglomerate > 0

    print(f'Performing initial fragment agglomeration until {epsilon_agglomerate}')

    # TODO: Multiply with mask
    fragments_data = fragments.astype(np.uint64)
    affs = affinities.astype(np.float32)

    generator = waterz.agglomerate(
        affs=affs,
        thresholds=[epsilon_agglomerate],
        fragments=fragments_data,
        scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        discretize_queue=256,
        return_merge_history=False,
        return_region_graph=False,
        gt=gt_seg
    )
    fragments_data[:] = next(generator)

    # cleanup generator
    for _ in generator:
        pass

    segmentation = fragments_data

    return segmentation, fragments, boundary_distances


def mpl_vis(raw, segmentation, pred_affs, pred_lsds):
    n_channels = raw.shape[0]
    fig, axes = plt.subplots(
        3,
        10,  # 14 + n_channels,
        figsize=(20, 6),
        sharex=True,
        sharey=True,
        squeeze=False)
    # view predictions (for lsds we will just view the mean offset component)
    for i in range(n_channels):
        axes[0][i].imshow(raw[i][raw.shape[1] // 2], cmap='gray')
    axes[1][0].imshow(np.squeeze(pred_affs[0][pred_affs.shape[1] // 2]), cmap='jet')
    axes[1][1].imshow(np.squeeze(pred_affs[1][pred_affs.shape[1] // 2]), cmap='jet')
    axes[1][2].imshow(np.squeeze(pred_affs[2][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][0].imshow(np.squeeze(pred_lsds[0][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][1].imshow(np.squeeze(pred_lsds[1][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][2].imshow(np.squeeze(pred_lsds[2][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][3].imshow(np.squeeze(pred_lsds[3][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][4].imshow(np.squeeze(pred_lsds[4][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][5].imshow(np.squeeze(pred_lsds[5][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][6].imshow(np.squeeze(pred_lsds[6][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][7].imshow(np.squeeze(pred_lsds[7][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][8].imshow(np.squeeze(pred_lsds[8][pred_affs.shape[1] // 2]), cmap='jet')
    axes[2][9].imshow(np.squeeze(pred_lsds[9][pred_affs.shape[1] // 2]), cmap='jet')
    axes[1][3].imshow(create_lut(np.squeeze(segmentation)[segmentation.shape[0] // 2]))
    # plt.show()
    return fig, axes


def get_mean_report(reports: dict) -> dict:
    # Get keys of second level, i.e. values of first level. These are the keys that will be used for aggregation.
    report_keys = next(iter(reports.values()))
    mean_report = {}
    for k in report_keys:
        kvalues = [reports[fname][k] for fname in reports.keys()]
        try:
            mean_report[k] = np.mean(kvalues)
        except TypeError:
            # print(f'{k=}: {kvalues=} are not np.mean()-able')
            # Ignore non-numerical values. TODO: If they are useful we can recurse into subdicts and apply mean over leaves
            pass
    return mean_report


def get_per_cube_vois(reports: dict) -> dict:
    per_cube_vois = {}
    for fname in reports.keys():
        per_cube_vois[f'voi_{fname}'] = reports[fname]['voi']
    return per_cube_vois


def get_per_cube_metrics(reports: dict, metric_name: str) -> dict:
    per_cube_metrics = {}
    for fname in reports.keys():
        per_cube_metrics[f'{metric_name}_{fname}'] = reports[fname][metric_name]
    return per_cube_metrics


def eval_cubes(cfg: DictConfig, checkpoint_path: Optional[Path] = None, enable_zarr_results=True):
    cube_root = Path(cfg.eval.cube_root)
    if checkpoint_path is None:  # fall back to cfg path if not overridden
        checkpoint_path = cfg.eval.checkpoint_path

    raw_paths = list(cube_root.glob('*.zarr'))

    if cfg.eval.max_eval_cubes is not None:
        raw_paths = raw_paths[:cfg.eval.max_eval_cubes]
        logging.info(f'Limiting eval to {cfg.eval.max_eval_cubes} cubes')

    cube_eval_results: dict[str, CubeEvalResult] = {}
    rand_voi_reports: dict[str, dict] = {}
    assert len(raw_paths) > 0

    for raw_path in raw_paths:
        name = raw_path.name

        cevr = run_eval(cfg=cfg, raw_path=raw_path, checkpoint_path=checkpoint_path, enable_zarr_results=enable_zarr_results)
        cube_eval_results[name] = cevr
        rand_voi_reports[name] = cevr.report

    logging.info(f'Results summary with\n- cube_root: {cfg.eval.cube_root}\n- checkpoint_path: {checkpoint_path}\n- Stats:\n')

    # rand_voi_reports = {name: cube_eval_results[name].rand_voi_report for name in cube_eval_results.keys()}

    for name, rep in rand_voi_reports.items():
        logging.info(f'{name}:\n{rep}\n')
        # logging.info(f'{name}:\n{rep["voi"]:.4f}\n')  # Just log VOIs for now

    mean_report = get_mean_report(rand_voi_reports)
    mean_voi = mean_report['voi']
    logging.info(f'Mean VOI: {mean_voi:.4f}')


    return cube_eval_results


def run_eval(cfg: DictConfig, raw_path: Path, checkpoint_path: Optional[Path] = None, enable_zarr_results=True):
    # raw, pred_lsds, pred_affs, pred_hardness = predict_unlabeled(cfg=cfg, raw_path=raw_path, checkpoint_path=checkpoint_path)
    batch = predict_unlabeled(cfg=cfg, raw_path=raw_path, checkpoint_path=checkpoint_path)
    # raw_path = '/cajal/scratch/projects/misc/mdraw/data/fullraw_fromknossos_j0126/j0126.zarr'
    # predict_unlabeled_zarr(cfg=cfg, raw_path=raw_path, checkpoint_path=checkpoint_path)
    # exit()
    ## Uncomment when predict_labeled works
    # batch = predict_labeled(cfg=cfg, raw_path=raw_path, checkpoint_path=checkpoint_path)
    raw = batch['raw']
    pred_affs = batch['pred_affs']


    data = zarr.open(str(raw_path), 'r')
    gt_seg = np.array(data[cfg.dataset.gt_name])  # type: ignore

    ws_affs = pred_affs

    # Get GT affs and LSDs

    # TODO: Don't do these computations here, use gunpowder nodes in predict() instead and let it fill the `batch`

    aff_nhood = cfg.labels.aff.nhood
    gt_affs = gp.add_affinities.seg_to_affgraph(gt_seg.astype(np.int32), nhood=aff_nhood).astype(np.float32)
    cropped_pred_affs, _ = center_crop(pred_affs, gt_affs)

    lsd_sigma = (cfg.labels.lsd.sigma, ) * 3
    lsd_downsample = cfg.labels.lsd.downsample

    gt_lsds = get_local_shape_descriptors(
        segmentation=gt_seg,
        sigma=lsd_sigma,
        downsample=lsd_downsample,
    )

    pred_lsds = batch.get('pred_lsds')
    pred_boundaries = batch.get('pred_boundaries')
    pred_hardness = batch.get('pred_hardness')

    cropped_raw, _ = center_crop(raw, gt_seg)

    cropped_pred_lsds, _ = center_crop(pred_lsds, gt_lsds)

    if pred_hardness is None:
        cropped_pred_hardness = None
    else:
        cropped_pred_hardness, _ = center_crop(pred_hardness, gt_seg)

    # TODO: These weights are actually different during training - see gp.BalanceLabels
    lsds_weights = 1.
    affs_weights = 1.

    # loss_class = import_symbol(cfg.loss.loss_class)
    # loss_init_kwargs = cfg.loss.get('init_kwargs', {})
    # loss_f = loss_class(**loss_init_kwargs)

    # eval_loss = loss_f(
    eval_loss = WeightedMSELoss()(
        pred_lsds=torch.as_tensor(cropped_pred_lsds),
        gt_lsds=torch.as_tensor(gt_lsds),
        lsds_weights=torch.as_tensor(lsds_weights),
        pred_affs=torch.as_tensor(cropped_pred_affs),
        gt_affs=torch.as_tensor(gt_affs),
        affs_weights=torch.as_tensor(affs_weights),
        # hardness_prediction=torch.as_tensor(cropped_pred_hardness),
    ).item()

    logging.info(f'Eval loss (hardcoded MSE): {eval_loss:.3f}')

    # print(f'{pred_affs.shape=}, {ws_affs.shape=}')

    # higher thresholds will merge more, lower thresholds will split more
    threshold = cfg.eval.threshold
    fragment_threshold = cfg.eval.fragment_threshold

    waterz_threshold_sweep_linspace = cfg.eval.waterz_threshold_sweep_linspace
    if waterz_threshold_sweep_linspace is not None:
        logging.info('Sweeping over waterz agglomeration thresholds...')
        start, end, num = waterz_threshold_sweep_linspace
        thresh_sweep_values = np.linspace(start, end, num)
        argmin_thresh, min_voi = sweep_segmentation_threshs(
            ws_affs,
            waterz_thresholds=thresh_sweep_values,
            fragment_threshold=fragment_threshold,
            merge_function_name=cfg.eval.merge_function,
            gt_seg=gt_seg,
        )
        logging.info(f'Sweep finished. Doing segmentation with argmin threshold {argmin_thresh}')
        threshold = argmin_thresh

    pred_seg, pred_frag, boundary_distances = get_mf_segmentation(
        ws_affs,
        waterz_threshold=threshold,
        fragment_threshold=fragment_threshold,
        merge_function_name=cfg.eval.merge_function,
    )

    cropped_pred_frag, _ = center_crop(pred_frag, gt_seg)
    cropped_pred_seg, _ = center_crop(pred_seg, gt_seg)

    rand_voi_report = rand_voi(
        gt_seg,
        cropped_pred_seg,  # segment_ids,
        return_cluster_scores=False
    )
    voi = rand_voi_report["voi_split"] + rand_voi_report["voi_merge"]
    rand_voi_report['voi'] = voi

    rand_voi_report['loss'] = eval_loss

    result_arrays = dict(
        raw=raw,
        pred_affs=pred_affs,
        pred_lsds=pred_lsds,
        pred_seg=pred_seg,
        pred_frag=pred_frag,
        cropped_raw=cropped_raw,
        cropped_pred_affs=cropped_pred_affs,
        cropped_pred_lsds=cropped_pred_lsds,
        cropped_pred_seg=cropped_pred_seg,
        cropped_pred_frag=cropped_pred_frag,
        cropped_pred_hardness=cropped_pred_hardness,
        gt_seg=gt_seg,
        gt_affs=gt_affs,
        gt_lsds=gt_lsds,
    )
    result_arrays = {k: v for k, v in result_arrays.items() if v is not None}

    eval_result = CubeEvalResult(
        report=rand_voi_report,
        arrays=result_arrays,
    )

    result_zarr_root = cfg.eval.result_zarr_root
    write_groups = cfg.eval.get('write_groups', None)
    if enable_zarr_results and result_zarr_root is not None:
        _fname = raw_path.name
        _checkpoint_path = cfg.eval.checkpoint if checkpoint_path is None else checkpoint_path  # Fall back to cfg ckpt
        _run_name = _get_run_name_from_checkpoint_path(_checkpoint_path)
        result_zarr_path = Path(result_zarr_root) / f'results_{_run_name}_{_fname}'
        eval_result.write_zarr(result_zarr_path, groups=write_groups)

    logging.info(f'VOI: {voi}')

    return eval_result


# TODO: Fix remaining errors
def predict_labeled(cfg, raw_path: Path | str, checkpoint_path: Optional[Path | str] = None) -> dict[str, np.ndarray]:
    """Run scan inference on labeled data."""

    voxel_size = gp.Coordinate(cfg.dataset.voxel_size)
    # Prefer ev_inp_shape if specified, use regular inp_shape otherwise
    input_shape = gp.Coordinate(
        cfg.model.backbone.get('ev_inp_shape', cfg.model.backbone.inp_shape)
    )
    input_size = input_shape * voxel_size
    offset = gp.Coordinate(cfg.model.backbone.offset)
    output_shape = input_shape - offset
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_hardness = gp.ArrayKey('PRED_HARDNESS')

    labels_mask = gp.ArrayKey('GT_LABELS_MASK')
    gt_affs_mask = gp.ArrayKey('GT_AFFINITIES_MASK')


    # TODO: Investigate input / output shapes w.r.t. offsets - output sizes don't always match each other
    context = (input_size - output_size) / 2

    source_data_dict = {
        raw: cfg.dataset.raw_name,
        labels: cfg.dataset.gt_name,
        labels_mask: cfg.dataset.mask_name,
    }
    source_array_specs = {
        raw: gp.ArraySpec(interpolatable=True),
        labels: gp.ArraySpec(interpolatable=False),
        labels_mask: gp.ArraySpec(interpolatable=False),
    }

    source = ZarrSource(
        str(raw_path),
        source_data_dict,
        source_array_specs,
    )

    # if cfg.dataset.labels_padding is not None:
    #     # We need more padding than for training, so multiply by constant factor
    #     # TODO: Figure out how much padding is actually needed
    #     # labels_padding = gp.Coordinate(cfg.dataset.labels_padding) * 5
    #     labels_padding = gp.Coordinate(cfg.dataset.labels_padding)
    #     # labels_padding = gp.Coordinate([8400, 7200, 7200])
    #     source += gp.Pad(labels, labels_padding)
    #     source += gp.Pad(labels_mask, labels_padding)

    source += gp.GrowBoundary(
        labels,
        mask=labels_mask,
        steps=1,
        only_xy=True
    )

    source += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,
        **cfg.labels.lsd
    )

    neighborhood = cfg.labels.aff.nhood

    source += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        labels_mask=labels_mask,
        affinities_mask=gt_affs_mask,
        dtype=np.float32
    )

    # TODO (IMPORTANT): Synchronize balance with training - here we rebalance based on _current_ batch stats.
    # Currently there is no way to keep this in sync (not even among training batches), so results are not directly comparable!
    source += gp.BalanceLabels(
        gt_affs,
        affs_weights,
        gt_affs_mask,
    )

    source += gp.Unsqueeze([raw])

    # if cfg.dataset.labels_padding is not None:
    #     labels_padding = gp.Coordinate(cfg.dataset.labels_padding)
    #     source += gp.Pad(labels, labels_padding)

    with gp.build(source):
        if cfg.eval.roi_shape is None:
            # source_roi = source.spec[raw].roi
            # total_input_roi = source_roi
            # total_output_roi = source_roi.grow(-context, -context)
            output_roi = source.spec[labels].roi
            # output_roi = gp.Roi((100, 100, 100), (150, 150, 150)) * voxel_size
            # output_roi = gp.Roi((100, 40, 40), output_shape) * voxel_size
            source_roi = output_roi.grow(context, context)
            total_input_roi = source_roi
            total_output_roi = output_roi
        else:
            _off = voxel_size * 0  # ~0 is not intuitive but it works? The ROI shape is apparently auto-centered~. Edit: Apparently not...
            # _off = voxel_size *
            raise NotImplementedError
            _sha = voxel_size * tuple(cfg.eval.roi_shape)
            total_output_roi = gp.Roi(offset=_off, shape=_sha)
            total_input_roi = total_output_roi.grow(context, context)
            # total_input_roi = gp.Roi(offset=_off, shape=_sha)
            # total_output_roi = total_input_roi.grow(-context, -context)

        # _gtlabel_shape = voxel_size * (8, 8, 8)  # TODO!
        # _gtlabel_off = voxel_size * (250, 250, 250)
        # label_roi = gp.Roi(offset=_gtlabel_off, shape=_gtlabel_shape)


    # model = get_mtlsdmodel()  # MtlsdModel()
    model = build_mtlsdmodel(cfg.model)

    # set model to eval mode
    model.eval()

    if checkpoint_path is None:  # Fall back to cfg checkpoint
        checkpoint_path = cfg.eval.checkpoint

    # add a predict node
    predict = Predict(
        model=model,
        checkpoint=checkpoint_path,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs,
            2: pred_hardness
        }
    )

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(labels, output_size)
    scan_request.add(gt_lsds, output_size)
    scan_request.add(lsds_weights, output_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(gt_affs, output_size)
    scan_request.add(affs_weights, output_size)
    scan_request.add(pred_affs, output_size)
    scan_request.add(pred_hardness, output_size)

    # this will scan in chunks equal to the input/output sizes of the respective arrays
    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)
    # pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)  # Rescale/shift to training value range

    pipeline += predict

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)  # Rescale/shift back for snapshot outputs

    pipeline += scan
    # pipeline += gp.Squeeze([raw, pred_lsds, pred_affs, pred_hardness])

    predict_request = gp.BatchRequest()

    # predict_request.add(raw, total_input_roi.get_end())
    # predict_request.add(pred_lsds, total_output_roi.get_end())
    # predict_request.add(pred_affs, total_output_roi.get_end())
    # predict_request.add(pred_hardness, total_output_roi.get_end())

    # # Also add gt stuff for loss calculation
    # predict_request.add(gt_lsds, total_output_roi.get_end())
    # predict_request.add(lsds_weights, total_output_roi.get_end())
    # predict_request.add(gt_affs, total_output_roi.get_end())
    # predict_request.add(affs_weights, total_output_roi.get_end())


    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi
    predict_request[pred_hardness] = total_output_roi

    # Also add gt stuff for loss calculation
    predict_request[gt_lsds] = total_output_roi
    predict_request[lsds_weights] = total_output_roi
    predict_request[gt_affs] = total_output_roi
    predict_request[affs_weights] = total_output_roi


    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    loss_class = import_symbol(cfg.loss.loss_class)
    loss_init_kwargs = cfg.loss.get('init_kwargs', {})
    loss_f = loss_class(**loss_init_kwargs)

    # Expects matching shapes for outputs and targets - TODO: make sure they are cropped to the same shape
    eval_loss = loss_f(
        pred_lsds=torch.as_tensor(batch[pred_lsds].data),
        gt_lsds=torch.as_tensor(batch[gt_lsds].data),
        lsds_weights=torch.as_tensor(batch[lsds_weights].data),
        pred_affs=torch.as_tensor(batch[pred_affs].data),
        gt_affs=torch.as_tensor(batch[gt_affs].data),
        affs_weights=torch.as_tensor(batch[affs_weights].data),
        pred_hardness=torch.as_tensor(batch[pred_hardness].data),
    ).item()

    logging.info(f'Eval loss: {eval_loss:.3f}')

    np_batch = {}
    for gp_key, gp_array in batch.items():
        np_batch[gp_key.identifier.lower()] = gp_array.data

    return np_batch


def _get_run_name_from_checkpoint_path(checkpoint_path):
    """Get name of parent directory of the checkpoint, i.e. the run name"""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent.name


@hydra.main(version_base='1.3', config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    eval_cubes(cfg)


if __name__ == "__main__":
    main()
