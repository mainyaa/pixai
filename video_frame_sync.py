import cv2
import sys
import skimage
import skimage.transform
import numpy as np
import glob
import tqdm

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

def detect_fourcc(video):
    result = []
    resultstr = []
    for cc in "H264 VP90 MPG4 X264 AVC1 MP4V".split(" "):
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*cc)
        #cap.set(CV_CAP_PROP_FOURCC, fourcc);
        success, frame1 = cap.read()
        if success:
            result.append(fourcc)
            resultstr.append(cc)
        cap.release()
    return result, resultstr

def video_frame_sync(video1, video2):
    cap1 = cv2.VideoCapture(video1)

    length1 = int(cap1.get(CV_CAP_PROP_FRAME_COUNT))
    width1  = int(cap1.get(CV_CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(CV_CAP_PROP_FRAME_HEIGHT))
    fps1    = cap1.get(CV_CAP_PROP_FPS)
    perframe = int(length1 / 10)
    pershift = int(perframe / 2)
    print(length1, width1, height1, fps1)
    print(perframe)

    filelist = glob.glob("./input/sailormoon_1_big.mp4*.png")
    if len(filelist) == 0:
        frames1 = []
        for i in tqdm.tqdm(range(length1)):
            success, frame1 = cap1.read()
            if not success:
                raise
            if (i + pershift) % perframe != 0:
                continue
            frames1.append(frame1)
            cv2.imwrite(video1+"%06d.png" % (i+pershift), frame1)
        cap1.release()
    else:
        frames1 = [cv2.imread(f) for f in filelist]
    
    diff = []
    diff_count = []
    for c in range(len(frames1)):
        cap2 = cv2.VideoCapture(video2)
        length2 = int(cap2.get(CV_CAP_PROP_FRAME_COUNT))
        width2  = int(cap2.get(CV_CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(CV_CAP_PROP_FRAME_HEIGHT))
        fps2    = cap2.get(CV_CAP_PROP_FPS)
        pos1 = c * perframe
        if c == 0:
            print(length2, width2, height2, fps2)
        mins = []
        maxs = []
        if width1/height1 >= 1.77:
            pad = 3/4*width1
            frames1[c] = frames1[c][pad:width1-pad,0:height2]
        img1 = skimage.transform.resize(frames1[c], (height2, width1))
        for i in tqdm.tqdm(range(length2/5)):
            success, frame2 = cap2.read()
            if not success:
                raise
            res = cv2.matchTemplate(frame2, np.array(img1, np.uint8), cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            mins.append(min_val)
            maxs.append(max_val)
        minv = np.where(min_val > 0.8)
        maxv = np.where(max_val > 0.8)
        c2 = np.argmax(max_val)
        pos2 = c2 * perframe
        diff.append(pos1 - pos2)
        diff_count.append(len(maxv))
        cap2.release()
        print(diff)
        print(diff_count)
    


    # When everything done, release the capture
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_frame_sync(sys.argv[1], sys.argv[2])
