#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import argparse
import datetime
import re

import numpy as np
from google.cloud import storage
import apache_beam as beam
from logging import (
        info,
        debug,
)

__version__ = "0.0.1"
BUCKET_NAME = "kagglebu-g"

def sort_num(img_list):
    img_list2 = [(re.findall("[0-9]+", x)[-1], x) for x in img_list]
    img_list2.sort(cmp = lambda x, y: cmp(int(x[0]), int(y[0])))
    return [x[1] for x in img_list2]


class TransformDoFn(beam.DoFn):

    def start_bundle(self):
        pass

    def process(self, element):
        info(element)
        path = element
        input_path = "/tmp/input.png"
        self.download_file(path, input_path)
        pb_path = '/tmp/tz6-s-stitch-sblur-lowtv-gen.npy'
        self.download_file("graph/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy", pb_path)
        self.download_file(path, input_path)

        try:
            out_path = self.render(pb_path=pb_path, img_path=input_path, out='/tmp')
        except Exception as e:
            info(e)
            print(e)
            try:
                out_path = self.render(pb_path=pb_path, img_path=input_path, out='/tmp')
            except Exception as e:
                info(e)
                print(e)
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(input_path)
        out_path = name + "_" + pb_name + ext
        if os.path.exists(out_path):
            path = path.replace("input", "output")
            self.upload_file(path, out_path)
            info("Task finished: "+out_path + ", " + path)

    def load_image(self, path, height=None, width=None):
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


    def create_tiles(self, large, height, width, num):
        import tensorflow as tf
        h_stride = height / num
        w_stride = width / num
        t_tiles = []
        for y in xrange(num):
            row = []
            for x in xrange(num):
                t_tile = tf.slice(large, [0, y * h_stride, x * w_stride, 0], [1, h_stride, w_stride, 3])
                row.append(t_tile)
            t_tiles.append(row)
        return t_tiles


    def render(self, pb_path, img_path, out):
        import time
        import skimage
        import skimage.io
        import skimage.transform
        import tensorflow as tf

        from pixai.tensorzoom_net import TensorZoomNet
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        debug("start render: ", out_path)
        img = self.load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        net = TensorZoomNet(pb_path, False)
        net.build(contents)
        fast_output = net.output

        start_time = time.time()
        sess = tf.Session()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        info("output calculated: %.10f sec" % duration)

        # print image
        skimage.io.imsave(out_path, output[0])
        return out_path


    def render_sliced(self, pb_path, img_path, side_num, out):
        import time
        import skimage
        import skimage.io
        import skimage.transform
        import tensorflow as tf
        from pixai.tensorzoom_net import TensorZoomNet
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        debug("start render: ", out_path)
        img = self.load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        # use stitch training method, slice the image into tiles and concat as batches
        tiles = self.create_tiles(contents, img.shape[0], img.shape[1], side_num)
        batch = tf.concat(0, [tf.concat(0, tiles[y]) for y in xrange(side_num)])  # row1, row2, ...

        net = TensorZoomNet(pb_path, False)
        net.build(batch)

        # stitch the tiles back together after split the batches
        split = tf.split(0, side_num ** 2, net.output)
        fast_output = tf.concat(1, [
                tf.concat(2, [split[x] for x in xrange(side_num * y, side_num * y + side_num)])
                for y in xrange(side_num)])

        start_time = time.time()
        sess = tf.Session()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        info("output calculated: %.10f sec" % duration)

        # print image
        skimage.io.imsave(out_path, output[0])
        debug("img saved:", out_path)
        return out_path
    def download_file(self, path, input_path):
        from apache_beam.io.gcp.gcsio import GcsIO
        gcs = GcsIO()
        r = gcs.open("gs://"+BUCKET_NAME+"/"+path, "r")
        w = open(input_path, "w")
        w.write(r.read())
        w.close()
        r.close()
    def upload_file(self, path, out_path):
        from apache_beam.io.gcp.gcsio import GcsIO
        gcs = GcsIO()
        w = gcs.open("gs://"+BUCKET_NAME+"/"+path, "w")
        r = open(out_path, "r")
        w.write(r.read())
        r.close()
        w.close()





def get_pipeline_name(cloud):
    if cloud:
        return 'DataflowRunner'
    else:
        return 'DirectRunner'

def default_args(argv):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_path',
        default=BUCKET_NAME,
        help='Output directory to write results to.')
    parser.add_argument(
        '--project',
        type=str,
        default='kagglebu',
        help='The cloud project name to be used for running this pipeline')

    parser.add_argument(
        '--job_name',
        type=str,
        default='pixai-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
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
            'job_name': ('cloud-ml-sample-iris' + '-' +
                        datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
            # Dataflow needs a copy of the version of the cloud ml sdk that
            # is being used.
            #'extra_packages': ['scikit-image'],
            "disk_size_gb": 60,
            'setup_file': './setup.py',
            'requirements_file': 'requirements.txt',
            "num_workers": 1000,
            'experiments=shuffle_mode': "service",
            }
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
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    return blobs


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    return blobs


def run(args):
    options = beam.utils.pipeline_options.PipelineOptions.from_dictionary(vars(args))
    info(options)

    p = beam.Pipeline(get_pipeline_name(args.cloud), options=options)
    #paths = [u'input/AddamsFamily/AddamsFamily.mp4.frame.{}.png'.format(i) for i in range(2699)]
    # exclude already proceed file
    blobs = list_blobs_with_prefix(BUCKET_NAME, "input/AddamsFamily/")
    paths = [blob.name for blob in blobs]
    blobs = list_blobs_with_prefix(BUCKET_NAME, "output/AddamsFamily")
    ex_paths = [blob.name.replace("output", "input") for blob in blobs]
    paths = np.setdiff1d(paths, ex_paths).tolist()
    print(paths)
    
    # run pipeline
    (p  | 'create' >> beam.Create(paths)
        | 'transform' >> beam.ParDo(TransformDoFn())
        )

    p.run()
    #p.run().wait_until_finish()

if __name__ == "__main__":
    run(default_args(sys.argv[1:]))
