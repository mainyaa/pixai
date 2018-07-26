import sys
import os
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(levelname)s %(filename)-s:%(lineno)-s: %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)

from util import (
        get_runner,
        default_args
        )

__version__ = "0.0.2"
JOB_NAME = "pixai-similarity-task-" + __version__.replace(".", "-")
PROJECT_ID = "topgate-ai-dev"
BUCKET_NAME = PROJECT_ID + "-df"


class TransformDoFn(beam.DoFn):

    def start_bundle(self):
        from util import download_file
        if not os.path.exists("input"):
            try:
                os.mkdir("input")
                os.mkdir("image")
                os.mkdir("output")
            except:
                pass
            big = "input/nadesico2.tar"
            try:
                download_file("gs://"+BUCKET_NAME+"/"+big, ""+big)
                self.extract(big , "image")
            except:
                download_file("gs://"+BUCKET_NAME+"/"+big, ""+big)
                self.extract(big , "image")


    def process(self, element):
        from similarity import compare_image
        from util import upload_file
        import pandas as pd
        res1, res2, res3, res4, res5, w, h = range(7)
        el1, el2 = element.strip().split(",")
        pel1 = "" + el1
        pel2 = "" + el2
        try:
            res1, res2, res3, res4, res5, w, h = compare_image(pel1, pel2)
        except Exception as e:
            print(e)
            print(e)
            try:
                res1, res2, res3, res4, res5, w, h = compare_image(pel1, pel2)
            except Exception as e:
                print(e)
                print(e)
        row = pd.DataFrame([[el1, el2, res1, res2, res3, res4, res5, w, h]], columns=["truth", "scale", "ssim", "nssim", "ssim2", "nssim2", "psnr", "width", "height"])
        print(row)
        path = os.path.basename(el1) + "_" + os.path.basename(el2)
        path = os.path.join("output", path)
        csvpath = path + ".csv"
        row.to_csv(csvpath, index=None, header=None)
        upload_file("gs://"+BUCKET_NAME+"/"+csvpath, csvpath)
        print("Task finished: " + csvpath)

    def extract(self, f, path):
        import tarfile
        tar = tarfile.open(f)
        tar.extractall(path)
        tar.close()

def run(args):
    print(args)
    args.num_workers = 1000
    options = PipelineOptions.from_dictionary(vars(args))
    print(options)

    # run pipeline
    inputtext = "gs://{}/input/nadesico2.txt".format(BUCKET_NAME)
    #outputtext = "gs:///output/nadesico2.csv".format(BUCKET_NAME)
    print(args)
    p = beam.Pipeline(get_runner(args.cloud), options=options)
    (p      | 'read' >> ReadFromText(inputtext)
            | 'transform' >> beam.ParDo(TransformDoFn())
            )

    p.run()
    #p.run().wait_until_finish()

if __name__ == "__main__":
    run(default_args(sys.argv[1:], JOB_NAME, PROJECT_ID, BUCKET_NAME))
