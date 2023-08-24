# Based on https://github.com/funkelab/daisy/blob/6293ba/examples/visualize.py

# The original code no longer works with daisy 1.0 and daisy 0.2.1 also needs some modifications in third-party libraries.
# This version replaces daisy calls with other libraries but aims to achieve the same functionality as the original code
# probably had earlier before dependencies were messed up.


# TODO: Render LSDs and affinities as RGB using add_layer(..., shader='rgb')

#!/usr/bin/env python

from funlib.show.neuroglancer import add_layer
import argparse
import daisy
import glob
import neuroglancer
import os
import webbrowser
import numpy as np
import zarr
import re
from urllib.parse import urlparse

from funlib.persistence import open_ds

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file',
    '-f',
    type=str,
    action='append',
    help="The path to the container to show")
parser.add_argument(
    '--datasets',
    '-d',
    type=str,
    nargs='+',
    action='append',
    default=[['']],
    help="The datasets in the container to show")
parser.add_argument(
    '--graphs',
    '-g',
    type=str,
    nargs='+',
    action='append',
    help="The graphs in the container to show")
parser.add_argument(
    '--no-browser',
    '-n',
    type=bool,
    nargs='?',
    default=False,
    const=True,
    help="If set, do not open a browser, just print a URL")
parser.add_argument(
    '--port',
    '-p',
    type=int,
    nargs='?',
    default=0,
    help="Set port number. Default is 34343.")

args = parser.parse_args()

neuroglancer.set_server_bind_address('0.0.0.0', args.port)
viewer = neuroglancer.Viewer()



def get_heatmap_shader(channel, scale=1.0, shift=0.0, cmap='inferno'):
    assert cmap in ['inferno', 'viridis', 'colormapJet']

    # Almost jinja...

    heatmap_shader = """

    // Approximation from https://observablehq.com/@flimsyhat/webgl-color-maps
    vec3 inferno(float t) {
        const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
        const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
        const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
        const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
        const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
        const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
        const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);

        return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
    }

    // Approximation from https://observablehq.com/@flimsyhat/webgl-color-maps
    vec3 viridis(float t) {
        const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
        const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
        const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
        const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
        const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
        const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
        const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);

        return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
    }

    void main() {
        float v = toNormalized({{ SCALE }} * (getDataValue({{ CHANNEL }}) + {{ SHIFT }}));
        vec4 rgba = vec4(0,0,0,0);
        if (v != 0.0) {
            rgba = vec4({{ CMAP }}(v), 1.0);
        }
        emitRGBA(rgba);
    }
    """\
    .replace('{{ CHANNEL }}', str(channel))\
    .replace('{{ CMAP }}', cmap)\
    .replace('{{ SCALE }}', str(scale))\
    .replace('{{ SHIFT }}', str(shift))

    return heatmap_shader


def to_slice(slice_str):

    values = [int(x) for x in slice_str.split(':')]
    if len(values) == 1:
        return values[0]

    return slice(*values)

def parse_ds_name(ds):

    tokens = ds.split('[')

    if len(tokens) == 1:
        return ds, None

    ds, slices = tokens
    slices = list(map(to_slice, slices.rstrip(']').split(',')))

    return ds, slices

class Project:

    def __init__(self, array, dim, value):
        self.array = array
        self.dim = dim
        self.value = value
        self.shape = array.shape[:self.dim] + array.shape[self.dim + 1:]
        self.dtype = array.dtype

    def __getitem__(self, key):
        slices = key[:self.dim] + (self.value,) + key[self.dim:]
        ret = self.array[slices]
        return ret

def slice_dataset(a, slices):

    dims = a.roi.dims

    for d, s in list(enumerate(slices))[::-1]:

        if isinstance(s, slice):
            raise NotImplementedError("Slicing not yet implemented!")
        else:
            index = (s - a.roi.get_begin()[d])//a.voxel_size[d]
            a.data = Project(a.data, d, index)
            a.roi = daisy.Roi(
                a.roi.get_begin()[:d] + a.roi.get_begin()[d + 1:],
                a.roi.get_shape()[:d] + a.roi.get_shape()[d + 1:])
            a.voxel_size = a.voxel_size[:d] + a.voxel_size[d + 1:]

    return a

def open_dataset(f, ds):
    original_ds = ds
    ds, slices = parse_ds_name(ds)
    slices_str = original_ds[len(ds):]

    try:
        dataset_as = []
        if all(key.startswith("s") for key in zarr.open(f)[ds].keys()):
            raise AttributeError("This group is a multiscale array!")
        for key in zarr.open(f)[ds].keys():
            dataset_as.extend(open_dataset(f, f"{ds}/{key}{slices_str}"))
        return dataset_as
    except AttributeError as e:
        # dataset is an array, not a group
        pass

    print("ds    :", ds)
    print("slices:", slices)
    try:
        zarr.open(f)[ds].keys()
        is_multiscale = True
    except:
        is_multiscale = False

    if not is_multiscale:
        a = open_ds(f, ds)

        if slices is not None:
            a = slice_dataset(a, slices)

        if a.roi.dims == 2:
            print("ROI is 2D, recruiting next channel to z dimension")
            a.roi = daisy.Roi((0,) + a.roi.get_begin(), (a.shape[-3],) + a.roi.get_shape())
            a.voxel_size = daisy.Coordinate((1,) + a.voxel_size)

        if a.roi.dims == 4:
            print("ROI is 4D, stripping first dimension and treat as channels")
            a.roi = daisy.Roi(a.roi.get_begin()[1:], a.roi.get_shape()[1:])
            a.voxel_size = daisy.Coordinate(a.voxel_size[1:])

        if a.data.dtype == np.int64 or a.data.dtype == np.int16:
            print("Converting dtype in memory...")
            a.data = a.data[:].astype(np.uint64)

        return [(a, ds)]
    else:
        return [([open_ds(f, f"{ds}/{key}") for key in zarr.open(f)[ds].keys()], ds)]


def get_layer_opts(dataset: str) -> dict:
    if dataset.endswith('hardness'):
        return {'shader': get_heatmap_shader(0, scale=1.0, shift=0.)}
    if dataset.endswith('lsds'):
        return {'c': [0, 1, 2]}
        # return {'c': [3, 4, 5]}
        # return {'c': [6, 7, 8]}
        # return {'shader': get_heatmap_shader(9)}

    return {}


for f, datasets in zip(args.file, args.datasets):

    arrays = []
    for ds in datasets:
        try:

            print("Adding %s, %s" % (f, ds))
            dataset_as = open_dataset(f, ds)

        except Exception as e:

            print(type(e), e)
            print("Didn't work, checking if this is multi-res...")

            scales = glob.glob(os.path.join(f, ds, 's*'))
            if len(scales) == 0:
                print(f"Couldn't read {ds}, skipping...")
                raise e
            print("Found scales %s" % ([
                os.path.relpath(s, f)
                for s in scales
            ],))
            a = [
                open_dataset(f, os.path.relpath(scale_ds, f))
                for scale_ds in scales
            ]
        for a in dataset_as:
            arrays.append(a)

    with viewer.txn() as s:
        for array, dataset in arrays:

            layer_opts = get_layer_opts(dataset)
            add_layer(s, array, dataset, **layer_opts)

if args.graphs:
    for f, graphs in zip(args.file, args.graphs):

        for graph in graphs:

            graph_annotations = []
            try:
                ids = open_ds(f, graph + '-ids').data
                loc = open_ds(f, graph + '-locations').data
            except:
                loc = open_ds(f, graph).data
                ids = None
            dims = loc.shape[-1]
            loc = loc[:].reshape((-1, dims))
            if ids is None:
                ids = range(len(loc))
            for i, l in zip(ids, loc):
                if dims == 2:
                    l = np.concatenate([[0], l])
                graph_annotations.append(
                    neuroglancer.EllipsoidAnnotation(
                        center=l[::-1],
                        radii=(5, 5, 5),
                        id=i))
            graph_layer = neuroglancer.AnnotationLayer(
                annotations=graph_annotations,
                voxel_size=(1, 1, 1))

            with viewer.txn() as s:
                s.layers.append(name='graph', layer=graph_layer)

url = str(viewer)
_parsed_url = urlparse(url)
localhost_url = _parsed_url._replace(netloc=re.sub('^[^:]*', 'localhost', _parsed_url.netloc)).geturl()
print(f'Remote URL:\n  {url}\n\nLocal URL (use with SSH tunnel):\n  {localhost_url}\n')


if os.environ.get("DISPLAY") and not args.no_browser:
    webbrowser.open_new(url)

# print("Press ENTER to quit")
input()