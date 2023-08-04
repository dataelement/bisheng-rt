import numpy as np
import math
import os
import time
import cv2
from config import finalize_configs, config as cfg


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def preprocess(img, fix_shape):
    # preprocess image
    orig_shape = img.shape[:2]
    longer_edge_size = cfg.PREPROC.TEST_LONG_EDGE_SIZE
    h = orig_shape[0]
    w = orig_shape[1]
    scale = longer_edge_size * 1.0 / max(h, w)
    if h > w:
        newh, neww = longer_edge_size, scale * w
    else:
        newh, neww = scale * h, longer_edge_size
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    resized_img = cv2.resize(img, dsize=(neww, newh))
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])

    if fix_shape:
        img_pad = np.zeros([cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 3], dtype='uint8')
        img_pad[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
        resized_img = img_pad

    return resized_img, scale


def images_preprocess(src_dir, dst_dir, fix_shape):
    if not os.path.exists(dst_dir + '/bin'):
        os.makedirs(dst_dir + '/bin')
    if not os.path.exists(dst_dir + '/shape'):
        os.makedirs(dst_dir + '/shape')

    finalize_configs()
    names = load_files(src_dir)
    with open(os.path.join(dst_dir, 'image_file.txt'), 'w') as f:
        for name in names:
            p = name.rfind('.')
            f.write(name[:p] + '\n')
            im = cv2.imread(os.path.join(src_dir, name))
            origin_h, origin_w = im.shape[:2]
            im_resized, scale = preprocess(im, fix_shape)
            resized_h, resized_w = im_resized.shape[:2]
            # im_resized = np.transpose(im_resized, [2, 0, 1])

            shape = list(im_resized.shape)
            print(name, shape)
            im_name = dst_dir + '/bin/' + name[:p]
            shape_name = dst_dir + '/shape/' + name[:p]
            im_resized.astype(np.float32).tofile(im_name)
            np.array([resized_h, resized_w, origin_h, origin_w, scale], dtype=np.float32).tofile(shape_name)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fix_shape = True

    src_dir = '/home/gulixin/workspace/datasets/all_kinds_train_images_angle/val'
    data_dir = '../../../build/test_data/ocr_maskrcnn_data/all_kinds_train_images_angle_val_fix_shape_hwc_1600'
    images_preprocess(src_dir, data_dir, fix_shape)

