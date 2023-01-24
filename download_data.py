# Based on https://github.com/funkelab/lsd/blob/a495573/lsd/tutorial/notebooks/lsd_data_download.ipynb

import os
import boto3
import numpy as np
import tqdm


# # list data
# client.list_objects(Bucket=bucket, Prefix="funke")

# # download directory structure file - this shows exactly how the s3 data is stored
# client.download_file(
#     Bucket=bucket,
#     Key="funke/structure.md",
#     Filename="structure.md")


# function to download all files nested in a bucket path
def downloadDirectory(
    bucket_name,
    remote_path,
    # local_path,
    access_key,
    secret_key
):

    resource = boto3.resource(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key)
    
    bucket = resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=remote_path):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))

        key = obj.key

        print(f'Downloading {key}')
        bucket.download_file(key, key)


# set bucket credentials
# (these were already public on the official lsd repo, so it should be fine to keep them here)
access_key = 'AKIA4XXGEV6ZQOTMTHX6'
secret_key = '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'
bucket = 'open-neurodata'

# connect to client
client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

# local_path = '/cajal/scratch/projects/misc/mdraw/data'

dataset_names = [
    'training',
    'testing/ground_truth/validation',
    # 'testing'
]

for dataset_name in dataset_names:
    downloadDirectory(
        bucket,
        f'funke/zebrafinch/{dataset_name}',
        # local_path=local_path,
        access_key,
        secret_key
    )

