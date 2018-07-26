from skimage.measure import compare_ssim as ssim
from skimage.io import imread
from skimage.color import rgb2gray
import sys
import os
import pyssim
import re
import numpy as np
import pandas as pd
import glob
import shutil
import math
from logging import basicConfig, getLogger, INFO
basicConfig(level=INFO, format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
logger = getLogger(__name__)
info = lambda x: logger.info(x)



# https://github.com/aizvorski/video-quality

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_image_list(file1, file2):
    info("compare_image_list")
    filelist1 = glob.glob(file1)
    filelist2 = glob.glob(file2)
    result = []
    result2 = {}
    for i, f1 in enumerate(filelist1):
        truth = imread(f1)
        w, h, c = truth.shape
        truths = rgb2gray(truth)
        mse_list = np.zeros((len(filelist2)))
        ssim_list = np.zeros((len(filelist2)))
        ssim2_list = np.zeros((len(filelist2)))
        ssim_ins = pyssim.SSIM(truth, size=(w,h))
        for j, f2 in enumerate(filelist2):
            bicubic = imread(f2)
            bicubics = rgb2gray(bicubic)
            mse_list[j] = mse(truth, bicubic)
            ssim_list[j] = ssim(truths, bicubics)
            ssim2_list[j] = ssim_ins.cw_ssim_value(bicubic)
        sim1 = np.abs(ssim_list).argmax()
        sim3 = np.abs(mse_list).argmin()
        sim4 = np.abs(ssim2_list).argmax()
        result.append({f1, filelist2[sim4]})
        result2[f1] = {k: ssim2_list[k] for k in filelist2}
    return result, result2

def compare_image(file1, file2):
    info("compare_image :" + file1 + ", " + file2)
    truth = imread(file1)
    w, h, c = truth.shape
    truths = rgb2gray(truth)
    ssim_ins = pyssim.SSIM(truth, size=(w,h))
    bicubic = imread(file2)
    bicubics = rgb2gray(bicubic)
    psnr_result = psnr(truth, bicubic)
    ssim_result = ssim(truths, bicubics)
    ssim2_result = ssim_ins.cw_ssim_value(bicubic)
    w = truth.shape[0]
    h = truth.shape[1]
    return ssim_result, 0, ssim2_result, 0, psnr_result, w, h

def search_sim():
    if len(sys.argv) > 1:
        info(sys.argv)
        info(compare_image(sys.argv[1], sys.argv[2]))
        exit()
    filelist1 = "./input/ranma_1_2_big_crop/ranma_1_2_big_crop.mp4.frame.000132.png"
    filelist2 = "./input/ranma_1_2_small_bicubic/ranma_1_2_small_bicubic.mp4.frame.0001[23]*.png"
    res1, res2 = compare_image(filelist1, filelist2)
    info(res1, res2)

    filelist1 = "./input/ranma_1_2_big_crop/ranma_1_2_big_crop.mp4.frame.000903.png"
    filelist2 = "./input/ranma_1_2_small_bicubic/ranma_1_2_small_bicubic.mp4.frame.0010[89]*.png"
    res1, res2 = compare_image(filelist1, filelist2)
    info(res1, res2)


def compare_image_negative(file1, file2):
    truth = imread(file1)
    w, h, c = truth.shape
    truths = rgb2gray(truth)
    negatruth = 255 - truth
    negatruths = 255 - truths
    bicubic = imread(file2)
    bicubics = rgb2gray(bicubic)
    negabicubic = 255 - bicubic
    negabicubics = 255 - bicubics
    negassim_ins = pyssim.SSIM(negatruth, size=(w,h))
    negassim_result = ssim(negatruths, negabicubics)
    negassim2_result = negassim_ins.cw_ssim_value(negabicubic)
    return [negassim_result, negassim2_result]

def compare_images_ssim(file1, file2):
    filelist1 = glob.glob(os.path.join(file1, "*.png"))
    filelist2 = glob.glob(os.path.join(file2, "*.png"))
    name1 = file1.split("/")[1]
    name2 = file2.split("/")[1]
    csv_name = "ssim_" + name1 + "_" + name2 + ".csv"
    l1 = len(filelist1)
    l2 = len(filelist2)
    result = []
    for i in range(min(l1, l2)):
        res1, res2, res3, res4, res5, w, h = compare_image(filelist1[i], filelist2[i])
        row = pd.DataFrame([[filelist1[i], filelist2[i], res1, res2, res3, res4, res5, w, h]], columns=["truth", "scale", "ssim", "nssim", "ssim2", "nssim2", "psnr", "width", "height"])
        result.append(row)
        if i % 1000 == 0:
            r = pd.concat(result)
            r.to_csv(os.path.join("image", csv_name), index=None, header=None)
    result = pd.concat(result)
    result.to_csv(os.path.join("image", csv_name), index=None, header=None)
    info(result)
    info(csv_name)
    return result

def compare_images_negative(file1, file2):
    df = pd.read_csv("image/ssim_nadesico_big_crop_nadesico_small_1080_crop.csv")
    filelist1 = glob.glob(os.path.join(file1, "*.png"))
    filelist2 = glob.glob(os.path.join(file2, "*.png"))
    name1 = file1.split("/")[1]
    name2 = file2.split("/")[1]
    csv_name = "ssim_" + name1 + "_" + name2 + ".csv"
    l1 = len(filelist1)
    l2 = len(filelist2)
    for i in range(min(l1, l2)):
        res1, res2 = compare_image_negative(filelist1[i], filelist2[i])
        df.ix[i, "nssim"] = res1
        df.ix[i, "nssim2"] = res2
        if i % 1000 == 0:
            df.to_csv(os.path.join("image", csv_name), index=None)
    df.to_csv(os.path.join("image", csv_name), index=None)
    return df

def compare_images_match(file1, file2):
    import cv2
    import os

    filelist1 = glob.glob(os.path.join(file1, "*.png"))
    filelist2 = glob.glob(os.path.join(file2, "*.png"))
    name1 = file1.split("/")[1]
    name2 = file2.split("/")[1]
    csv_name = "match_" + name1 + "_" + name2 + ".csv"
    IMG_SIZE = (200, 200)

    result = []
    for f1 in filelist1:
        target_img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, IMG_SIZE)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE)
        search_params = dict()   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(target_des,None)

        for f2 in filelist2:
            comparing_img = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
            comparing_img = cv2.resize(comparing_img, IMG_SIZE)
            (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
            kp2, des2 = sift.detectAndCompute(comparing_des,None)
            #matches = bf.match(target_des, comparing_des)
            matches = bf.knnMatch(target_des, comparing_des, k=2)
            #matches = flann.knnMatch(target_des,comparing_des,k=2)

# Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask,
                               flags = 0)

            img3 = cv2.drawMatchesKnn(target_img,kp1,comparing_img,kp2,matches,None,**draw_params)
            from matplotlib import pyplot as plt
            plt.imshow(img3,),plt.show()
            print(matches)
            dist = [m.distance for m in matches]
            res = []
            for m in matches:
                d = dict((name, getattr(m, name)) for name in dir(m) if not name.startswith('__'))
                res.append(d)
            print(res)
            res = sum(dist) / len(dist)
            row = pd.DataFrame([[f1, f2, res]], columns=["truth", "scale", "match"])
            result.append(row)
            print(f2, res)
            exit()
    result = pd.concat(result)
    result.to_csv(os.path.join("image", csv_name), index=None, header=None)
    info(result)
    info(csv_name)
    return result

def rename(file1):
    # image/nadesico_small_1080_crop/nadesico_small_1080.frames.000059.png.crop.00.png
    if file1.find("crop") > 0:
        filelist = glob.glob(os.path.join(file1, "*00.png"))
        for i in range(len(filelist)):
            offset = 3
            prefix = filelist[i][:-22]
            suffix = filelist[i][-4:]
            for j in range(30):
                name = prefix + "%06d.png.crop.%02d" + suffix
                name2 = prefix + "%06d.png.crop.%02d" + suffix
                name2 = name2.replace("_crop", "_crop_ss")
                #info(name % (i+offset, j), name2 % (i, j))
                shutil.copy(name % (i+offset, j), name2 % (i, j))
    else:
        filelist = glob.glob(os.path.join(file1, "*.png"))
        for i in range(len(filelist)):
            offset = 3
            prefix = filelist[i][:-10]
            suffix = filelist[i][-4:]
            name = prefix + "%06d" + suffix
            name2 = prefix + "%06d" + suffix
            name2 = name2.replace("_crop", "_crop_ss")
            shutil.copy(name % (i+offset), name2 % (i+1))
    filelist = glob.glob(os.path.join(file1, "*.png"))
    return filelist


def get_width(df):
    return imread(df).shape[0]

def get_height(df):
    return imread(df).shape[1]

crop_size = 255
crop_count = 30
ssimthrethold = 0.5
psnrthrethold = -1000
if __name__ == "__main__":
    if sys.argv[1] == "rename":
        if len(sys.argv) > 2:
            rename(sys.argv[2])
    if sys.argv[1] == "compare":
        if len(sys.argv) > 1:
            csv_name = compare_images_ssim(sys.argv[2], sys.argv[3])
            info(csv_name)
    if sys.argv[1] == "match":
        if len(sys.argv) > 1:
            csv_name = compare_images_match(sys.argv[2], sys.argv[3])
            info(csv_name)
    if sys.argv[1] == "ssim":
        #df = pd.read_csv("output/ssim_nadesico.csv", header=None)
        df = pd.read_csv("image/ssim_nadesico_big_nadesico_small_1080.csv")
        #df.columns = ["truth", "scale", "ssim", "nssim", "ssim2", "nssim2", "psnr", "width", "height"]
        df = df[df.ssim != 0]
        if "width" in df:
            df = df[df.width == 255]
            df = df[df.height == 255]
        df["base_truth"] = df["truth"].apply(os.path.basename)
        df["base_scale"] = df["scale"].apply(os.path.basename)
        df["ssim2_"] = np.round(df["ssim2"], decimals=1)
        df["ssim_"] = np.round(df["ssim"], decimals=1)
        if "psnr" in df:
            df["psnr_"] = np.round(df["psnr"], decimals=1)
            sortby = ["psnr_", "ssim2_", "ssim_", "ssim2", "ssim"]
        else:
            df["rmse_"] = np.round(df["rmse"], decimals=1)
            sortby = ["rmse_", "ssim2_", "ssim_", "ssim2", "ssim"]
        """
        if "nssim" in df:
            df["nssim2_"] = np.round(df["nssim2"], decimals=1)
            df["nssim_"] = np.round(df["nssim"], decimals=1)
            sortby = ["ssim2_", "psnr_", "ssim_", "nssim2_", "nssim_", "ssim2", "ssim"]
        """
        df = df.sort_values(by=sortby, ascending=False)
        """
        df = df[df.ssim >= 0.7]
        df = df[df.ssim <= 0.85]
        df = df[df.ssim2 >= 0.7]
        df = df[df.ssim2 <= 0.85]
        """
        print(df)
        info(df)
        n = int(df.shape[0]/3)
        end = df.shape[0]
        for i in range(0, end, n):
            pass
            #info(df.iloc[i, :])
            true = df.iloc[i, 0]
            scale = df.iloc[i, 1]
            num = re.findall("[0-9]+", scale)[-1]
            print(true)
            print(scale)
            print(num)
            os.system("open "+true)
            os.system("open "+scale)

