import boto3
import os

s3 = boto3.client('s3')
bucket_name = "eeg-denoise"

prefix = 'raw/'

local_data_path = 'eeg_data_from_s3/'
os.makedirs(local_data_path, exist_ok=True)

paginator = s3.get_paginator('list_objects_v2')
for result in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in result.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue  # skip folder placeholders

        # Remove prefix to get relative path
        relative_path = os.path.relpath(key, prefix)
        local_file_path = os.path.join(local_data_path, relative_path)

        # Ensure local directories exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download file
        s3.download_file(bucket_name, key, local_file_path)
        # print(f"Downloaded: {key} -> {local_file_path}")
