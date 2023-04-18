import os

import h5py
import zarr


def create_data(
        data_path,  # url,
        name,
        offset,
        resolution,
        # sections=None,
        # squeeze=True
):
    container = zarr.open(name, 'a')
    for index, file_name in enumerate(os.listdir(data_path)):

        in_f = h5py.File(data_path + file_name, 'r')

        raw = in_f["image"][:].transpose((3, 0, 1, 2))
        labels = in_f["label"]

        for ds_name, data in [
            ('raw', raw),
            ('labels', labels)]:
            container[f'{ds_name}/{index}'] = data
            container[f'{ds_name}/{index}'].attrs['offset'] = offset
            container[f'{ds_name}/{index}'].attrs['resolution'] = resolution


if __name__ == "__main__":
    if "training_data.zarr" not in os.listdir():
        create_data("./data/training_data/",
                    # "/cajal/nvmescratch/users/riegerfr/expMic_data/training_data/",
                    # "/home/franz/scratch/FFN/training_data/",  # 'https://cremi.org/static/data/sample_A_20160501.hdf',
                    'training_data.zarr',
                    offset=[0, 0, 0],
                    resolution=[1, 1, 1],  # [4, 4, 20]  # todo: is this nm?
                    )
        create_data("./data/validation_data/",
                    # "/cajal/nvmescratch/users/riegerfr/expMic_data/validation_data/",
                    'validation_data.zarr',
                    offset=[0, 0, 0],
                    resolution=[1, 1, 1],
                    )
        print("done")
