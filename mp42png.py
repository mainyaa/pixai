import cv2
import sys
import os

def mp42png(input_file):
    print(input_file)
    if not input_file:
        exit()
    input_dir = input_file.split(".")[0]
    try:
        os.mkdir(input_dir)
    except:
        pass
    vidcap = cv2.VideoCapture(input_file)
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      if not success:
          break
      filename = input_dir+"/"+os.path.basename(input_file)+".frame.%06d.png" % count
      print(filename)
      cv2.imwrite(filename, image)
      count += 1

if __name__ == "__main__":
    mp42png(sys.argv[1])
