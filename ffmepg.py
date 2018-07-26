import sys
import os
import subprocess
import glob
import signal
from subprocess import call, PIPE
import ffmpeg
import skvideo.io
import numpy as np

ftime = 1/30
# python ffmepg.py  output/gundom_linear.mp4  output/gundom_zoom_test2.mp4
left_skip = 0.6666
right_skip = 0
# python ffmepg.py  output/gundom_linear.mp4  output/gundom_zoom_test2.mp4
left_skip = 0
right_skip = 0.2333
# python ffmepg.py  input/sailormoon_stars_big.mp4 input/sailormoon_stars_small_720.mp4
left_skip = 0
right_skip = 0
# python ffmepg.py  input/ranma_1_2_big_crop2.mp4 input/ranma_1_2_small_cc_1080.mp4
left_skip = 0.
right_skip = 0. # 0.0333*8

from ffmpeg.nodes import filter_operator, GlobalNode, FilterNode


@filter_operator()
def blend(*streams, **kwargs):
    return FilterNode(streams, "blend", kwargs=kwargs, max_inputs=None).stream()
@filter_operator()
def vstack(*streams, **kwargs):
    return FilterNode(streams, "vstack", kwargs=kwargs, max_inputs=None).stream()
@filter_operator()
def hstack(*streams, **kwargs):
    return FilterNode(streams, "hstack", kwargs=kwargs, max_inputs=None).stream()

@filter_operator()
def pad(stream, width='iw', height='ih', x='0', y='0'):
    return FilterNode(stream, "pad", args=[width, height, x, y]).stream()

@filter_operator()
def crop(stream, width='iw', height='ih', x='0', y='0'):
    return FilterNode(stream, "crop", args=[width, height, x, y]).stream()

@filter_operator()
def scale(stream, width='iw', height='ih'):
    return FilterNode(stream, "scale", args=[width, height]).stream()

@filter_operator()
def fps(stream, **kwargs):
    return FilterNode(stream, "fps", kwargs=kwargs).stream()

ffmpeg.blend = blend
ffmpeg.vstack = vstack
ffmpeg.hstack = hstack
ffmpeg.pad = pad
ffmpeg.crop = crop
ffmpeg.scale = scale
ffmpeg.fps = fps

class Transform(object):
    def __init__(self):
        pass

    @staticmethod
    def hcompare(left, right):
        """ffmpeg -ss 0.0666 -i output/gundom_linear.mp4  -i output/gundom_zoom_test2.mp4  -filter_complex "[1:v] crop=iw/2:ih:iw/2:0 [right]; [0:v][right] overlay=main_w/2:0" diff/gundom_zoom_diff2.mp4"""
        output_file = "diff/hcompare_{}_{}.mp4".format(os.path.basename(left), os.path.basename(right))
        (ffmpeg
            .input(left)
            .setpts("PTS-STARTPTS")
            .pad(width='2*iw')
                .overlay(ffmpeg.input(right), x='main_w/2', y=0)
                .scale('2*iw/3', '2*ih/3')
                .output(output_file)
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def hsplit(left, right):
        """ffmpeg -i output/{}_superzoomai.mp4 -vf "[in] pad=iw:ih [left];movie=input/{}_bicubic.mp4,crop=1024:1536:1024:0[right];[left][right] overlay=1024:0[out]" output/{}_diff2.mp4"""
        output_file = "diff/hsplit_{}_{}.mp4".format(os.path.basename(left), os.path.basename(right))
        right_stream = (ffmpeg
            .input(right)
            .crop('in_w/2', 'in_h', 'in_w/2', '0')
            )
        (ffmpeg
            .input(left)
                .setpts("PTS-STARTPTS")
                .overlay(right_stream, x='main_w/2', y=0)
                .output(output_file)
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def blend(left, right, left_time=0, right_time=0):
        """ffmpeg -i output/{}_superzoomai.mp4 -i input/{}_bicubic.mp4 -filter_complex "[0:0]setpts=PTS-STARTPTS,split[a1][a2];[1:0]setpts=PTS-STARTPTS,split[b1][b2];[a1][b1]blend=c0_mode=difference,pad=2*iw:ih:0:0[blend];[a2][b2]hstack[tmp];[tmp][blend]vstack" output/{}_compare.mp4"""
        output_file = "diff/blend_{}_{}.mp4".format(os.path.basename(left), os.path.basename(right))
        left_input = (ffmpeg
            .input(left, ss=str(left_time))
            .setpts("PTS-STARTPTS")
        )
        right_input = (ffmpeg
            .input(right, ss=str(right_time))
        )
        (ffmpeg
            .blend(left_input, right_input, c0_mode="difference")
            .output(output_file)
            .overwrite_output()
            .run()
            )
        return output_file

    @staticmethod
    def check_video(left, right):
        lmeta = skvideo.io.ffprobe(left)["video"]
        rmeta = skvideo.io.ffprobe(right)["video"]
        for key in ["@width", "@height", "@sample_aspect_ratio", "@display_aspect_ratio", "@r_frame_rate"]:
            if lmeta[key] != rmeta[key]:
                import json
                print("left video:", json.dumps(lmeta, indent=2))
                print("right video:", json.dumps(rmeta, indent=2))
                raise Exception("{} not equaled: {}, {}".format(key, lmeta[key], rmeta[key]))
        return True
    @staticmethod
    def check_videos(args):
        columns = ["@width", "@height", "@sample_aspect_ratio", "@display_aspect_ratio", "@r_frame_rate"]
        check = []
        metas = []
        for video in args:
            meta = skvideo.io.ffprobe(video)["video"]
            check_list = [meta[key] for key in columns]
            check.append(check_list)
            metas.append(meta)
        # check properties all equal
        for i in range(1, len(check)):
            for j in range(len(check[i])):
                if check[0][j] != check[i][j]:
                    import json
                    print("videos:", json.dumps(metas, indent=2))
                    raise Exception("columns mismatch:", columns[j], check[0][j], check[i][j])
        return args

    def checkvideo(func):
        def inner_function(*args,**kwargs):
            Transform.check_videos(args)
            return func(*args)
        return inner_function

    @checkvideo
    @staticmethod
    def checkerboard(left, right):
        output_file = "diff/checker_{}_{}.mp4".format(os.path.basename(left), os.path.basename(right))
        print(output_file)
        left_input = (ffmpeg
            .input(left)
            .setpts("PTS-STARTPTS")
            .trim(duration="00:01:29.83")
        )
        right_input = (ffmpeg
            .input(right)
            .setpts("PTS-STARTPTS")
            .trim(duration="00:01:29.83")
        )
        (ffmpeg
            .blend(left_input, right_input, all_expr='if(gt(X,Y*(W/H)),A,B)')
            .output(output_file)
            .overwrite_output()
            .run()
            )
        print(output_file)
        return output_file

    @staticmethod
    def trim(input1, start_frame):
        output_file = "input/{}_trim.mp4".format(os.path.basename(input1))
        (ffmpeg
            .input(input1)
                .setpts("PTS-STARTPTS")
                .trim(start_frame=start_frame)
                .output(output_file)
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def ss(input1, start_time):
        output_file = "input/{}_trim.mp4".format(os.path.basename(input1))
        (ffmpeg
            .input(input1, ss=str(start_time))
                .setpts("PTS-STARTPTS")
                .output(output_file)
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def img2video(input1):
        """ffmpeg -i image-%03d.png video.mp4"""
        output_file = "image/concat_{}".format(os.path.basename(input1)[:-15])
        (ffmpeg
            .input(input1)
                .output(output_file)
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def video2img(input1):
        """ffmpeg -i image-%03d.png video.mp4"""
        path = "image/{}".format(os.path.basename(input1)[:-4])
        try:
            os.mkdir(path)
        except:
            pass
        output_file = path+"/"+os.path.basename(input1)[:-4]+".frames.%06d.png"
        (ffmpeg
            .input(input1)
                .output(output_file, format="image2")
                .overwrite_output()
                .run()
            )
        return output_file

    @staticmethod
    def imgcrop(input1):
        """convert -crop 100x100 original.jpg new.jpg"""
        for in1 in glob.glob(input1):
            base = os.path.basename(in1)
            path = os.path.join("image", base[:-18]+"_crop")
            tofile = os.path.join(path, base+".crop.%02d.png")
            try:
                os.mkdir(path)
            except:
                pass
            cmd = "convert -crop 255x255 {} {}".format(in1, tofile)
            print(cmd)
            subprocess.call(cmd, shell=True)
        return

    @staticmethod
    def scale(input1, start_frame):
        """ffmpeg -i input/ranma_1_2_small.mkv -s 1440x1080 -aspect 4:3 -vf "fps=23.98"  input/ranma_1_2_small_bicubic.mp4"""
        path = "image/{}".format(os.path.basename(input1)[:-4])
        try:
            os.mkdir(path)
        except:
            pass
        output_file = input1[:-4]+"_1080.mp4"
        (ffmpeg
            .input(input1, ss=str(start_frame))
                .fps(fps="23.98")
                .output(output_file, s="1440x1080", aspect="4:3")
                .overwrite_output()
                .run()
            )
        return output_file



if __name__ == "__main__":
    mode = sys.argv[1]
    left = sys.argv[2]
    if len(sys.argv) > 3:
        left = sys.argv[2]
        right = sys.argv[3]
    if mode == "hcompare":
        out = Transform.hcompare(left, right)
    if mode == "hsplit":
        out = Transform.hsplit(left, right)
    if mode == "trim-gundom":
        out = Transform.trim("output/gundom_zoom_test2.mp4", 20)
    if mode == "trim-ranma":
        out = Transform.trim("input/ranma_1_2_small_cc_1080.mp4", 8)
    if mode == "ss-nadesico":
        out = Transform.ss("input/nadesico_big.mp4", 0.125)
    if mode == "chkerboard":
        out = Transform.checkerboard(left, right)
    if mode == "video2img":
        out = Transform.video2img("input/nadesico_big.mp4")
        """
        out = Transform.video2img("input/conan_01_wide_1080.mp4")
        out = Transform.video2img("input/conan_01_small_1080.mp4")
        out = Transform.video2img("input/nadesico_small_1080.mp4")
        out = Transform.video2img("input/conan_big_pad.mkv")
        out = Transform.video2img("input/conan_small_1080.mkv")
        out = Transform.video2img("input/conan_01_big.mp4")
        out = Transform.video2img("input/conan_01_wide_1080.mp4")
        out = Transform.video2img("input/conan_01_small_1080.mp4")
        """
    if mode == "img2video":
        out = Transform.img2video(left)
    if mode == "imgcrop":
        #out = Transform.imgcrop("image/conan_01_small_1080/*.png")
        #out = Transform.imgcrop("image/conan_01_wide_1080/*.png")
        out = Transform.imgcrop("image/nadesico_big/*.png")
        #out = Transform.imgcrop("image/nadesico_small_1080/*.png")
    if mode == "scale-conan-01-small":
        out = Transform.scale(left, 15.375)
    if mode == "blend":
        out = Transform.blend(left, right)
    print(out)
    if out != None:
        os.system("open "+out)
