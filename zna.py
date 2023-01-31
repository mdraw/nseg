import napari
import numpy as np
import zarr
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Show zarr neurons with labels in napari.')
    parser.add_argument('path')
    args = parser.parse_args()
    
    # fn = '/home/m4/dev/expmiclsd/validation_data.zarr'
    # fn = '/home/m4/data/funke/zebrafinch/training/gt_z255-383_y1407-1663_x1535-1791.zarr'
    fn = args.path
    fn = Path(fn).expanduser()
    z = zarr.open(fn, mode='r')

    raw = z.volumes.raw
    labels = z.volumes.labels.neuron_ids

    padded_labels = pad_labels(raw, labels)


    viewer = napari.Viewer()
    # import IPython ; IPython.embed(); raise SystemExit
    viewer.add_image(raw, name='raw')
    viewer.add_labels(padded_labels, name='lab')

    napari.run()
    # viewer = napari.view_image(np.moveaxis(h['image'][()], -1, 0), name='raw')
    #viewer.add_labels(h['label'], name='label')

if __name__ == '__main__':
    main()