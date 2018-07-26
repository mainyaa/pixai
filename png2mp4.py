import cv2
import sys
import os
import re
import tqdm
import numpy as np
import shutil

CV_CAP_PROP_POS_MSEC = 0
CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FOURCC = 6
CV_CAP_PROP_FRAME_COUNT = 7
CV_CAP_PROP_FORMAT = 8
CV_CAP_PROP_MODE = 9
CV_CAP_PROP_BRIGHTNESS = 10
CV_CAP_PROP_CONTRAST = 11
CV_CAP_PROP_SATURATION = 12
CV_CAP_PROP_HUE = 13
CV_CAP_PROP_GAIN = 14
CV_CAP_PROP_EXPOSURE = 15
CV_CAP_PROP_CONVERT_RGB = 16
CV_CAP_PROP_WHITE_BALANCE = 17
CV_CAP_PROP_RECTIFICATION = 18

def sort_num(img_list):
    img_list2 = [(re.findall("[0-9]+", x)[-1], x) for x in img_list]
    img_list2.sort(cmp = lambda x, y: cmp(int(x[0]), int(y[0])))
    return [x[1] for x in img_list2]

def png2mp4(input_dir, ref_video=None):
    if not input_dir:
        exit()
    if ref_video:
        ext = ref_video.split(".")[-1]
        cap = cv2.VideoCapture(ref_video)
        #length = int(cap.get(CV_CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(CV_CAP_PROP_FPS)
        fourcc  = cap.get(CV_CAP_PROP_FOURCC)
        print((width, height), fps, fourcc)
        if fourcc == 0.:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            print(fourcc)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            print(fourcc)
        cap.release()
    else:
        ext = "mp4"
        fps = 29.97
        #fps = 23.98
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    paths = os.listdir(input_dir)
    paths = sort_num(paths)
    img = cv2.imread(os.path.join(input_dir, paths[0]))
    (width, height, channel) = img.shape
    out_path = os.path.basename(os.path.normpath(input_dir)) + "." + ext
    out_path = os.path.join("output", out_path)

    print((width, height))
    print(img.shape, fps, fourcc)
    print(os.path.join(input_dir, paths[0]), out_path)
    video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    length = int(video.get(CV_CAP_PROP_FRAME_COUNT))
    width  = int(video.get(CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(CV_CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(CV_CAP_PROP_FPS)
    fourcc  = video.get(CV_CAP_PROP_FOURCC)
    print((width, height))
    print(length, fps, fourcc)
    for path in tqdm.tqdm(paths):
        img = cv2.imread(os.path.join(input_dir, path))
        print(img.shape, video)
        video.write(img)

    video.release()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        png2mp4(sys.argv[1], sys.argv[2])
    else:
        png2mp4(sys.argv[1])
