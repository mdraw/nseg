# https://github.com/funkelab/lsd/blob/master/lsd/tutorial/notebooks/segment.ipynb

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gunpowder.torch
import gunpowder as gp
import matplotlib.pyplot as plt
import napari
import numpy as np
import waterz
import zarr
from funlib.evaluate import rand_voi
from scipy.ndimage import label
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed

from params import input_size, output_size
from shared import create_lut, get_mtlsdmodel


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


def predict(
        checkpoint,
        raw_file,
        raw_dataset):
    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    # TODO: Investigate input / output shapes w.r.t. offsets - output sizes don't always match each other
    context = (input_size - output_size) / 2

    source = gp.ZarrSource(
        raw_file,
        {
            raw: raw_dataset
        },
        {
            raw: gp.ArraySpec(interpolatable=True)
        })

    source += gp.Unsqueeze([raw])

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context, -context)

    model = get_mtlsdmodel()  # MtlsdModel()

    # set model to eval mode
    model.eval()

    # add a predict node
    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs
        }
    )

    # this will scan in chunks equal to the input/output sizes of the respective arrays
    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)

    # raw shape = h,w

    # pipeline += gp.Unsqueeze([raw])

    # raw shape = c,h,w

    pipeline += gp.Stack(1)

    # raw shape = b,c,h,w

    pipeline += predict
    pipeline += scan
    # pipeline += gp.Squeeze([raw])

    # raw shape = c,h,w
    # pred_lsds shape = b,c,h,w
    # pred_affs shape = b,c,h,w

    pipeline += gp.Squeeze([raw, pred_lsds, pred_affs])

    # raw shape = h,w
    # pred_lsds shape = c,h,w
    # pred_affs shape = c,h,w

    predict_request = gp.BatchRequest()

    # this lets us know to process the full image. we will scan over it until it is done
    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred_lsds, total_output_roi.get_end())
    predict_request.add(pred_affs, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[raw].data, batch[pred_lsds].data, batch[pred_affs].data


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
        min_seed_distance=10, fragment_threshold=0.5):
    mean_affs = (1 / 3) * \
                (affs[0] +
                 affs[1] + affs[2])  # todo: other affinities? *0.5

    # fragments = np.zeros(mean_affs.shape, dtype=np.uint64)

    boundary_mask = mean_affs > (fragment_threshold * max_affinity_value)
    boundary_distances = distance_transform_edt(boundary_mask)

    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)

    return ret[0], boundary_distances


# @title segmentation wrapper
def get_segmentation(affinities, waterz_threshold, fragment_threshold):
    fragments, boundary_distances = watershed_from_affinities(affinities, fragment_threshold=fragment_threshold)  # [0]
    thresholds = [waterz_threshold]  # todo: add 0?

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),  # .squeeze(1),
        fragments=np.copy(fragments),  # .squeeze(0),
        thresholds=thresholds,
    )

    segmentation = next(generator)

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


@dataclass
class CubeEvalResult:
    """
    segmentation.shape=(150, 150, 150)
    raw.shape=(1, 350, 550, 550)
    pred_affs.shape=(3, 330, 530, 530)
    pred_lsds.shape=(10, 330, 530, 530)
    """
    rand_voi_report: dict[str, Any]
    raw: np.ndarray
    gt_seg: np.ndarray
    pred_frag: np.ndarray
    pred_seg: np.ndarray
    # gt_affs: np.ndarray
    # gt_lsds: np.ndarray
    pred_affs: np.ndarray
    pred_lsds: np.ndarray


def eval_cubes(cube_root, checkpoint, show_in_napari=False):
    # 'model_checkpoint_50000'
    val_root = Path(cube_root)
    raw_files = list(val_root.glob('*.zarr'))
    raw_dataset = 'volumes/raw'

    cube_eval_results: dict[str, CubeEvalResult] = {}
    rand_voi_reports = {}
    assert len(raw_files) > 0

    for raw_file in raw_files:
        name = raw_file.name

        # pred_affs, pred_lsds, rand_voi_report, raw, segmentation = run_eval(checkpoint, raw_dataset, str(raw_file), show_in_napari=show_in_napari)
        cevr = run_eval(checkpoint, raw_dataset, str(raw_file), show_in_napari=show_in_napari)
        cube_eval_results[name] = cevr


    for name, rep in rand_voi_reports.items():
        print(f'{name}:\n{rep}\n')

    return cube_eval_results


def run_eval(checkpoint, raw_dataset, raw_file, show_in_napari=False):
    raw, pred_lsds, pred_affs = predict(checkpoint, raw_file, raw_dataset)
    # watershed assumes 3d arrays, create fake channel dim (call these watershed affs - ws_affs)
    ws_affs = np.stack([
        # np.zeros_like(pred_affs[0]),
        pred_affs[0],  # todo
        pred_affs[1],
        pred_affs[2],

    ]
    )
    # just test a 0.5 threshold. higher thresholds will merge more, lower thresholds will split more
    threshold = 0.2
    fragment_threshold = 0.5
    pred_seg, pred_frag, boundary_distances = get_segmentation(
        ws_affs,
        waterz_threshold=threshold,
        fragment_threshold=fragment_threshold
    )
    data = zarr.open(raw_file, 'r')
    print('Before roi report...')
    gt_seg = np.array(data.volumes.labels.neuron_ids)

    # labels = labels[100:-100, 200:-200, 200:-200]
    # segmentation = segmentation[100:-100, 200:-200, 200:-200]
    pred_seg, gt_seg = center_crop(pred_seg, gt_seg)
    # TODO: Also crop preds, fragments, ...
    rand_voi_report = rand_voi(
        gt_seg,
        pred_seg,  # segment_ids,
        return_cluster_scores=False
    )
    voi = rand_voi_report["voi_split"] + rand_voi_report["voi_merge"]
    rand_voi_report['voi'] = voi

    if show_in_napari:
        # fragments_new = np.reshape(np.arange(fragments.size, dtype=np.uint64), fragments.shape) + 1
        # generator = waterz.agglomerate(
        #    affs=ws_affs.astype(np.float32),  # .squeeze(1),
        #    fragments=np.copy(fragments_new),  # .squeeze(0),
        #    thresholds=[0.5],
        # )
        # new_segmentation = next(generator)
        # (fragments!=0).mean(), (data["labels"][0][10:-10, 10:-10, 10:-10]!=0).mean()
        viewer = napari.Viewer()
        viewer.add_image(pred_lsds[0:3], channel_axis=0, name="lsd0,2", gamma=2)
        viewer.add_image(pred_lsds[3:6], channel_axis=0, name="lsd3,6", gamma=2)
        viewer.add_image(pred_lsds[6:9], channel_axis=0, name="lsd6,9", gamma=2)
        viewer.add_image(pred_lsds[9:10], channel_axis=0, name="lsd9", gamma=2)
        viewer.add_image(pred_affs, channel_axis=0, name="affs", gamma=2)
        viewer.add_image(raw[:, 10:-10, 10:-10, 10:-10], channel_axis=0, name="raw", opacity=0.2, gamma=2)
        # viewer.add_image(pred_affs, channel_axis=0, name="affs")
        # viewer.add_image(boundary_distances, name="boundary_distances")
        viewer.add_labels(pred_seg, name="seg")
        viewer.add_labels(fragments, name="frag")
        viewer.add_labels(gt_seg, name="gt")
        napari.run()

    eval_result = CubeEvalResult(
        raw=raw,
        pred_affs=pred_affs,
        pred_lsds=pred_lsds,
        pred_seg=pred_seg,
        pred_frag=pred_frag,
        gt_seg=gt_seg,
        rand_voi_report=rand_voi_report,
    )
    return eval_result


if __name__ == "__main__":
    eval_cubes(
        val_root=Path('/cajal/u/mdraw/lsdex/data/zebrafinch_msplit/validation/').expanduser(),
        checkpoint='/cajal/u/mdraw/lsdex/experiments/zebrafinch/model_checkpoint_20'
    )
