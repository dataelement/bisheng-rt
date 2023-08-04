import argparse
import numpy as np
import sys
import os
import glob
import shutil
import cv2
import struct
from random import shuffle

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

NUM_CALIBRATION_IMAGES = 100

def resize_image(im, max_side_len=512):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    rd_scale = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    # rd_scale = float(max_side_len) / resize_w

    ratio_h = rd_scale
    ratio_w = rd_scale
    resize_h = int(resize_h * ratio_h)
    resize_w = int(resize_w * ratio_w)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    # pad_h = max_side_len - resize_h
    # pad_w = max_side_len - resize_w
    # im_pad = np.pad(im, ((0,pad_h), (0,pad_w), (0,0)), 'constant')

    im_pad = im
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im_pad, (ratio_h, ratio_w)


def main():

    CALIBRATION_DATASET_LOC = args.inDir + '/*.jpg'
    # images to test
    print("Location of dataset = " + CALIBRATION_DATASET_LOC)
    imgs = glob.glob(CALIBRATION_DATASET_LOC)

    # output
    outDir = args.outDir+"/"

    if os.path.exists(outDir):
        os.system("rm " + outDir +"/*")

    # prepare output
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    with open(args.outListFile, 'a') as f:
        for i in range(NUM_CALIBRATION_IMAGES):
            if fix_shape:
                im = Image.open(imgs[i]).resize((args.width, args.height), Image.NEAREST)
            else:
                im, _ = resize_image(cv2.imread(imgs[i]), max_side_len=1056)
                im = Image.fromarray(im[:, :, ::-1])
            imName = '%03d.ppm'%(i)
            print (imName, im.size)
            im.save(outDir + imName)
            f.write(imName + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inDir', required=True, help='Input directory')
    parser.add_argument('--outDir', required=True, help='Output directory')
    parser.add_argument('--height', type=int, default=1600, help='input height')
    parser.add_argument('--width', type=int, default=1600, help='input width')
    parser.add_argument('--outListFile', type=str, default="", help='output file names list')
    args = parser.parse_args()
    fix_shape = True
    main()



