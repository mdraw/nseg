import napari
import numpy as np
import zarr
import zarr.hierarchy
import argparse

from pathlib import Path


def pad_labels(raw: np.ndarray, labels: np.ndarray, copy=False):
    if copy:
        labels = labels.copy()  # Separate return value from input. Otherwise it shares input memory
    rsh = np.array(raw.shape)
    lsh = np.array(labels.shape)
    if np.any(rsh != lsh):
        # Zero-pad labels to match raw shape
        # Create a central slice with the size of the output
        lo = (rsh - lsh) // 2
        hi = rsh - lo
        slc = tuple(slice(l, h) for l, h in zip(lo, hi))

        padded_labels = np.zeros_like(raw, dtype=labels.dtype)
        padded_labels[slc] = labels
    return padded_labels


def get_subgroup(arr: zarr.hierarchy.Group, grouppath: str) -> zarr.Array:
    sub = arr
    for g in grouppath.split('.'):
        try:
            sub = sub[g]
        except KeyError as e:
            print(f'Group "{g}" not found in {sub}.\nSubvolume tree:\n{arr.tree()}')
            raise e
    return sub


def main():
    parser = argparse.ArgumentParser(description='Show zarr neurons with labels in napari.')
    parser.add_argument('path')
    parser.add_argument('-r', default='volumes.raw')
    # parser.add_argument('-l', default='volumes.labels.neuron_ids')
    parser.add_argument('-l', default=None)
    # parser.add_argument('--group-prefix', default='volumes')
    args = parser.parse_args()

    # fn = '/home/m4/dev/expmiclsd/validation_data.zarr'
    # fn = '/home/m4/data/funke/zebrafinch/training/gt_z255-383_y1407-1663_x1535-1791.zarr'
    # group_prefix = args.group_prefix
    raw_group = args.r
    labels_group = args.l

    # raw_group = f'{group_prefix}.{raw_group}'
    # labels_group = f'{group_prefix}.{labels_group}'

    fn = args.path
    fn = Path(fn).expanduser()
    z = zarr.open(fn, mode='r')

    viewer = napari.Viewer()

    raw = get_subgroup(z, raw_group)
    viewer.add_image(raw, name='raw')

    if labels_group is not None:
        labels = get_subgroup(z, labels_group)
        padded_labels = pad_labels(raw, labels)
        viewer.add_labels(padded_labels, name='lab')

    napari.run()
    # viewer = napari.view_image(np.moveaxis(h['image'][()], -1, 0), name='raw')
    #viewer.add_labels(h['label'], name='label')


if __name__ == '__main__':
    main()
