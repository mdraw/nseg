from pathlib import Path
import zarr


def print_shapes(root):
    root = Path(root).expanduser()
    print(f'{root}:')
    files = root.glob('*.zarr')
    for p in files:
        z = zarr.open(p, mode='r')
        rsh = z.volumes.raw.shape
        lsh = z.volumes.labels.neuron_ids.shape
        print(f'  {p.name: <50}: raw: {rsh}, lab: {lsh}')
    print()

if __name__ == '__main__':
    for r in [
        '~/data/zebrafinch_msplit/training/',
        '~/data/zebrafinch_msplit/validation/',
        '~/data/zebrafinch_msplit/excluded/',
    ]:
        print_shapes(r)
