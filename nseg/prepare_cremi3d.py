import os
import h5py
import zarr
import io
import requests


def create_data(
        url,
        name,
        offset,
        resolution,
        squeeze=True):

    in_f = h5py.File(io.BytesIO(requests.get(url).content), 'r')

    raw = in_f['volumes/raw']
    labels = in_f['volumes/labels/neuron_ids']
    
    container = zarr.open(name, 'a')

    index = 0
    for ds_name, data in [
        ('raw', raw),
        ('labels', labels)]:
        
        container[f'{ds_name}/{index}'] = data
        container[f'{ds_name}/{index}'].attrs['offset'] = offset
        container[f'{ds_name}/{index}'].attrs['resolution'] = resolution



if __name__ == "__main__":
    # if "training_data.zarr" not in os.listdir():
    if True:
        create_data(
            'https://cremi.org/static/data/sample_A_20160501.hdf',
            'training_data.zarr',
            # offset=[0, 0, 0],
            # resolution=[4, 4, 20],
            offset=[0,0,0],
            resolution=[1,1,1],
        )

        # create_data("./data/training_data/",
        #             # "/cajal/nvmescratch/users/riegerfr/expMic_data/training_data/",
        #             # "/home/franz/scratch/FFN/training_data/",  # 'https://cremi.org/static/data/sample_A_20160501.hdf',
        #             'training_data.zarr',
        #             offset=[0, 0, 0],
        #             resolution=[1, 1, 1],  # [4, 4, 20]  # todo: is this nm?
        #             )
        # create_data("./data/validation_data/",
        #             # "/cajal/nvmescratch/users/riegerfr/expMic_data/validation_data/",
        #             'validation_data.zarr',
        #             offset=[0, 0, 0],
        #             resolution=[1, 1, 1],
        #             )
        print("done")