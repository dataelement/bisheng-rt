import os
import cv2
import numpy as np

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

def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]

def preprocess(im, fix_shape):
    h, w, _ = im.shape
    '''
    side0 = max(h, w)
    sides = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
    distance = np.abs(sides - side0)
    edge_size = sides[np.argmin(distance)]
    '''
    edge_size = 1056
    im = im[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=edge_size)
    if fix_shape:
        img_pad = np.zeros([edge_size, edge_size, 3], dtype='uint8')
        img_pad[0:im_resized.shape[0], 0:im_resized.shape[1]] = im_resized
        im_resized = img_pad

    return im_resized, ratio_h, ratio_w

def images_preprocess(src_dir, dst_dir, is_transpose=False, fix_shape=False):
    if not os.path.exists(dst_dir + '/bin'):
        os.makedirs(dst_dir + '/bin')
    if not os.path.exists(dst_dir + '/shape'):
        os.makedirs(dst_dir + '/shape')

    names = load_files(src_dir)
    for name in names:
        p = name.rfind('.')
        im = cv2.imread(os.path.join(src_dir, name))
        try:
            im_resized, ratio_h, ratio_w = preprocess(im, fix_shape)
        except:
            continue
        if is_transpose:
            im_resized = np.transpose(im_resized, [2,0,1])

        print(name, ratio_h, ratio_w)
        shape = list(im_resized.shape)
        im_name = dst_dir + '/bin/' + name[:p]
        shape_name = dst_dir + '/shape/' + name[:p]
        im_resized.astype(np.float32).tofile(im_name)
        print(name, shape)
        np.array(shape + [ratio_h, ratio_w],dtype=np.float32).tofile(shape_name)


if __name__ == '__main__':
    im_dir = '/home/gulixin/workspace/datasets/all_kinds_train_images_angle_v3/val'
    # output_dir = '/home/yujinbiao/code/tensorrt/tests/res/east/im_raw_tf_no_fix'
    output_dir = '/home/gulixin/workspace/nn_predictor/trt-lib/build/test_data/ocr_east_data/im_raw_no_fix_nchw'
    # input_type: trt or tf
    input_type = "trt"
    fix_shape = False
    assert input_type in ["tf", "trt"]
    if input_type == "tf":
        images_preprocess(im_dir, output_dir)
    if input_type == "trt":
        images_preprocess(im_dir, output_dir, is_transpose=True, fix_shape=fix_shape)
