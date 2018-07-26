#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import subprocess
import os
import argparse
import datetime
import re

from logging import basicConfig, getLogger, DEBUG
basicConfig(level=DEBUG, format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)

__version__ = "0.0.1"

def sort_num(img_list):
    img_list2 = [(re.findall("[0-9]+", x)[-1], x) for x in img_list]
    img_list2.sort(cmp = lambda x, y: cmp(int(x[0]), int(y[0])))
    return [x[1] for x in img_list2]

def load_image(path, height=None, width=None):
    import skimage
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def download_file(gcspath, inpath):
    from apache_beam.io.gcp.gcsio import GcsIO
    gcs = GcsIO()
    r = gcs.open(gcspath, "r")
    w = open(inpath, "w")
    w.write(r.read())
    w.close()
    r.close()
def upload_file(gcspath, outpath):
    from apache_beam.io.gcp.gcsio import GcsIO
    gcs = GcsIO()
    w = gcs.open(gcspath, "w")
    r = open(outpath, "r")
    w.write(r.read())
    r.close()
    w.close()





def get_runner(cloud):
    if cloud:
        return 'DataflowRunner'
    else:
        return 'DirectRunner'

def default_args(argv, job_name, project_id, bucket_name):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        default=bucket_name,
        help='Output directory to write results to.')
    parser.add_argument(
        '--project',
        type=str,
        default=project_id,
        help='The cloud project name to be used for running this pipeline')

    parser.add_argument(
        '--job_name',
        type=str,
        default=job_name + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
        help='A unique job identifier.')
    #parser.add_argument(
        #'--num_workers', default=20, type=int, help='The number of workers.')
    parser.add_argument('--cloud', default=False, action='store_true')
    parser.add_argument(
        '--runner',
        help='See Dataflow runners, may be blocking'
        ' or not, on cloud or not, etc.')

    parsed_args, _ = parser.parse_known_args(argv)

    if parsed_args.cloud:
        # Flags which need to be set for cloud runs.
        default_values = {
            'project': get_cloud_project(),
            'temp_location': os.path.join("gs://"+parsed_args.output_path, 'temp'),
            'runner': 'DataflowRunner',
            'save_main_session': True,
            'staging_location':
            os.path.join("gs://"+parsed_args.output_path, 'staging'),
            'job_name': job_name + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            # Dataflow needs a copy of the version of the cloud ml sdk that
            # is being used.
            #'extra_packages': ['scikit-image'],
            "disk_size_gb": 20,
            "num_workers": 1000,
            "worker_machine_type": "n1-standard-1",
            #'experiments=shuffle_mode': "service",
            }
        if os.path.exists('./setup.py'):
            default_values['setup_file'] = './setup.py'
        if os.path.exists('requirements.txt'):
            default_values['requirements_file'] = 'requirements.txt'
    else:
        # Flags which need to be set for local runs.
        default_values = {
                'runner': 'DirectRunner',
                }
    if "sdk_location" in parsed_args:
        default_values['sdk_location'] = parsed_args.sdk_location

    for kk, vv in default_values.iteritems():
        if kk not in parsed_args or not vars(parsed_args)[kk]:
            vars(parsed_args)[kk] = vv

    info(parsed_args)
    return parsed_args


def get_cloud_project():
    cmd = [
            'gcloud', '-q', 'config', 'list', 'project',
            '--format=value(core.project)'
            ]
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            print(res)
            if not res:
                raise Exception('--cloud specified but no Google Cloud Platform '
                  'project found.\n'
                  'Please specify your project name with the --project '
                  'flag or set a default project: '
                  'gcloud config set project YOUR_PROJECT_NAME')
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('gcloud is not installed. The Google Cloud SDK is '
                        'necessary to communicate with the Cloud ML service. '
                        'Please install and set up gcloud.')
                raise

def list_blobs(bucket_name):
    from google.cloud import storage
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    return blobs


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    return blobs
