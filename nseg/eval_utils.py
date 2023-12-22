from copy import deepcopy

import logging

from functools import wraps, cached_property
import time
from dataclasses import dataclass
from funlib.evaluate import expected_run_length, rand_voi
from networkx import get_node_attributes
from typing import Any, Optional
import numpy as np
from numpy.typing import DTypeLike, ArrayLike
import zarr
# import zarr.storage
# from numcodecs import Zstd


# zarr.storage.default_compressor = Zstd(level=1)

ArrayDict = dict[str, np.ndarray]
Shape = tuple[int, ...] | np.ndarray

# Code mostly copied from Franz Rieger's implementation in
#  https://gitlab.mpcdf.mpg.de/riegerfr/synthetictreesegmentations/-/blob/45038e311781660af192ad0f850477ba367b0b7b/boundary_pred.py#L285
def compute_synem_metrics(pred, gt, gt_skel, tag, mode, verbose=True):

    _print = print if verbose else lambda *args, **kwargs: None

    _print(f"tag: {tag}")

    voi_report_dense = rand_voi(gt.astype(np.uint64), pred.astype(np.uint64), return_cluster_scores=False)
    voi_dense = voi_report_dense["voi_split"] + voi_report_dense["voi_merge"]

    _print(f"{mode}_voi_dense_{tag}", voi_dense)
    _print(f"{mode}_voi_dense_split_{tag}", voi_report_dense["voi_split"])
    _print(f"{mode}_voi_dense_merge_{tag}", voi_report_dense["voi_merge"])

    # pred_skel =
    skel = deepcopy(gt_skel)
    n_missed = 0
    nodes_to_be_removed = []

    for node in skel.nodes:
        x, y, z = skel.nodes[node]["index_position"]
        try:
            skel.nodes[node]["pred_id"] = pred[x, y, z]
        except IndexError:  # Node not in segmentation (can happen depending on crop/padding) -> remove node
            n_missed += 1
            nodes_to_be_removed.append(node)

    skel.remove_nodes_from(nodes_to_be_removed)

    _print(f"n_missed: {n_missed}")

    voi_report_skel = rand_voi(np.array(list(get_node_attributes(skel, "id").values())).astype(np.uint64),
                               np.array(list(get_node_attributes(skel, "pred_id").values())).astype(np.uint64),
                               return_cluster_scores=False)
    voi_skel = voi_report_skel["voi_split"] + voi_report_skel["voi_merge"]

    _print(f"{mode}_skel_voi_{tag}", voi_skel)
    _print(f"{mode}_skel_voi_split_{tag}", voi_report_skel["voi_split"])
    _print(f"{mode}_skel_voi_merge_{tag}", voi_report_skel["voi_merge"])

    erl = expected_run_length(skel, "id", "edge_length",
                              get_node_attributes(skel, "pred_id"),
                              skeleton_position_attributes=["nm_position"], return_merge_split_stats=True)

    max_erl = expected_run_length(skel, "id", "edge_length",
                                  get_node_attributes(skel, "id"),
                                  skeleton_position_attributes=["nm_position"], return_merge_split_stats=True)
    _print(f"{mode}_max_erl_{tag}", max_erl[0])
    nerl = erl[0] / max_erl[0]
    _print(f"nerl: {nerl}")
    _print(f"{mode}_erl_{tag}", erl[0])
    _print(f"{mode}_nerl_{tag}", nerl)

    return nerl, voi_skel, voi_dense



def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__}(...) took {total_time:.4f} seconds')
        # print(f'{func.__name__}({", ".join(args)}, kwargs={kwargs}) took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


# @timeit
def pad_to_shape(arr: np.ndarray, new_shape: Shape) -> np.ndarray:
    """
    Zero-pad to match new_shape
    """
    ash = np.array(arr.shape)
    tsh = np.array(new_shape)
    if np.any(ash > tsh):
        raise ValueError(f'{ash=} larger than {tsh=} in at least one dimension. Can\'t pad.')
    if np.all(ash == tsh):  # No need to pad, but return copy to prevent accidental writes
        return arr.copy()

    # Create a central slice with the size of the output
    lo = (tsh - ash) // 2
    hi = tsh - lo
    slc = tuple(slice(l, h) for l, h in zip(lo, hi))
    padded_arr = np.zeros_like(arr, shape=tsh)
    padded_arr[slc] = arr
    return padded_arr


# @timeit
def crop_to_shape(arr: np.ndarray, new_shape: Shape) -> np.ndarray:
    """
    Center-crop to match new_shape
    """
    ash = np.array(arr.shape)
    tsh = np.array(new_shape)
    if np.any(ash < tsh):
        raise ValueError(f'{ash=} smaller than {tsh=} in at least one dimension. Can\'t crop.')
    if np.all(ash == tsh):  # No need to crop, but return copy to prevent accidental writes
        return arr.copy()

    # Find slice coords for cropping
    lo = (ash - tsh) // 2
    hi = ash - lo
    slc = tuple(slice(l, h) for l, h in zip(lo, hi))
    cropped_arr = arr[slc]
    return cropped_arr


# @timeit
def _split_mc(array_dict: ArrayDict) -> ArrayDict:
    """Split multichannel arrays into channel index-named top-level subarrays, keep others as is."""
    split_mc = {}
    for k, v in array_dict.items():
        if v.ndim == 3:
            split_mc[k] = v.copy().astype(np.uint32)  # TODO: External casting
        elif v.ndim == 4:
            for i in range(v.shape[0]):
                split_mc[f'{k}{i}'] = v[i].copy().astype(np.float32)  # TODO: External casting
    return split_mc


# @timeit
def _get_max_shape(arrays: ArrayDict) -> np.ndarray:
    shapes = np.array([arr.shape for arr in arrays.values()])
    max_shape = np.max(shapes, axis=0)
    return max_shape


# @timeit
def _get_min_shape(arrays: ArrayDict) -> np.ndarray:
    shapes = np.array([arr.shape for arr in arrays.values()])
    min_shape = np.min(shapes, axis=0)
    return min_shape


# @timeit
def _pad_arrays(arrays: ArrayDict, new_shape: Shape) -> ArrayDict:
    """Pad all arrays to new_shape"""
    padded_arrays = {}
    for k, v in arrays.items():
        padded_arrays[k] = pad_to_shape(v, new_shape)
    return padded_arrays


# @timeit
def _crop_arrays(arrays: ArrayDict, new_shape: Shape) -> ArrayDict:
    """Center-crop all arrays to new_shape"""
    cropped_arrays = {}
    for k, v in arrays.items():
        cropped_arrays[k] = crop_to_shape(v, new_shape)
    return cropped_arrays


# TODO: Not working yet
# @timeit
def _cast_as(arrays: ArrayDict, dtype_map: dict[np.dtype, np.dtype]) -> ArrayDict:
    cast_arrays = {}
    for k, v in arrays.items():
        if isinstance(v.flat[0], np.floating):
            cast_arrays[k] = v.astype(np.float16).copy()
        else:
            cast_arrays[k] = v.astype(np.uint32).copy()

    for k, v in arrays.items():
        if v.dtype in dtype_map.keys():
            print(v.dtype)
            new_dtype = dtype_map[v.dtype]
            cast_arrays[k] = v.astype(new_dtype).copy()
        else:
            cast_arrays[k] = v.copy()

    # if any(np.issubdtype(v.dtype, orig_dtype) for orig_dtype in dtype_map.keys()):
    # for orig_dtype, new_dtype in dtype_map.items():
    #     import IPython ; IPython.embed(); raise SystemExit
    #     if np.issubdtype(v.dtype, orig_dtype):
    #         cast_arrays[k] = v.astype(new_dtype).copy()
    #     else:
    #         cast_arrays[k] = v.copy()
    return cast_arrays



# TODO: Optionally cache expensive results
@dataclass
class CubeEvalResult:
    """
    Evaluation results on one cube. Contains inputs, gt labels, model outputs and metrics.

    Example shapes for arrays:
        segmentation.shape=(150, 150, 150)
        raw.shape=(1, 350, 550, 550)
        pred_affs.shape=(3, 330, 530, 530)
        pred_lsds.shape=(10, 330, 530, 530)
    """
    report: dict[str, Any]
    arrays: ArrayDict

    _array_dtype_map = {
        np.float32: np.float16,
        np.float64: np.float16,
        np.uint64: np.uint32,
    }

    def print_stats(self, mc_split=False) -> None:
        """
        Example split array stats:
            gt_seg: min=0, max=40000156, n_unique=32, dtype=uint64
            pred_affs0: min=0.0, max=1.0, n_unique=18147474, dtype=float64
            pred_affs1: min=0.0, max=1.0, n_unique=17893015, dtype=float64
            pred_affs2: min=0.0, max=1.0, n_unique=18299675, dtype=float64
            pred_frag: min=0, max=10213, n_unique=8651, dtype=uint64
            pred_lsds0: min=0.0, max=0.9696524739265442, n_unique=9713844, dtype=float64
            pred_lsds1: min=0.0, max=0.9192934632301331, n_unique=8072445, dtype=float64
            pred_lsds2: min=0.0, max=0.94820237159729, n_unique=8227610, dtype=float64
            pred_lsds3: min=0.0, max=0.5376353859901428, n_unique=9575323, dtype=float64
            pred_lsds4: min=0.0, max=0.4648045301437378, n_unique=9415026, dtype=float64
            pred_lsds5: min=0.0, max=0.48246848583221436, n_unique=9476977, dtype=float64
            pred_lsds6: min=0.0, max=0.8840417861938477, n_unique=9309527, dtype=float64
            pred_lsds7: min=0.0, max=0.8831238746643066, n_unique=7962563, dtype=float64
            pred_lsds8: min=0.0, max=0.8513205647468567, n_unique=8345352, dtype=float64
            pred_lsds9: min=0.0, max=0.9996299743652344, n_unique=16387345, dtype=float64
            pred_seg: min=0, max=7107, n_unique=24, dtype=uint64
            raw0: min=0.003921568859368563, max=1.0, n_unique=255, dtype=float32
        """
        arrays = _split_mc(self.arrays) if mc_split else self.arrays
        for k, v in arrays.items():
            print(f'{k}: min={np.min(v)}, max={np.max(v)}, n_unique={len(np.unique(v))}, dtype={v.dtype}')

    # TODO: Make nview support differently shaped arrays so we can always "keep" by default
    # @timeit
    def _prepare_arrays_for_writing(self, mode='pad_to_max') -> ArrayDict:
        split_arrays = _split_mc(self.arrays)
        cast_arrays = split_arrays  # TODO
        # cast_arrays = _cast_as(split_arrays, dtype_map=self._array_dtype_map)  # TODO
        if mode == 'pad_to_max':
            max_shape = _get_max_shape(cast_arrays)
            return _pad_arrays(cast_arrays, new_shape=max_shape)
        elif mode == 'crop_to_min':
            min_shape = _get_min_shape(cast_arrays)
            return _crop_arrays(cast_arrays, new_shape=min_shape)
        elif mode == 'keep':
            return cast_arrays
        else:
            raise ValueError(f'{mode=} invalid.')

    #TODO: Optimize writes: threading?
    # @timeit
    def write_zarr(self, path, groups=None) -> None:
        zstore = zarr.DirectoryStore(path)
        print(f'Writing to {path}')
        zroot = zarr.group(store=zstore, overwrite=True)
        writable_arrays = self._prepare_arrays_for_writing()
        for k, v in writable_arrays.items():
            if groups is None or k in groups:
                zroot.create_group(k)
                zroot[k] = v

        print(zroot.tree())
        # zroot['report'] = self.report  # ValueError: missing object_codec for object array
