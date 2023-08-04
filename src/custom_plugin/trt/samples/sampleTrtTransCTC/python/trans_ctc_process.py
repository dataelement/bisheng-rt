import numpy as np
import os
import json
import cv2


def preprocess_recog_batch(images, IMAGE_HEIGHT=32, MIN_WIDTH=40, channels=1, downsample_rate=8, max_img_side=800):
    # batching mode
    # images list of np.array
    assert channels in [1, 3], print('chanels must be 1 or 3. Gray or BGR')
    bs = len(images)
    shapes = np.array(list(map(lambda x : x.shape[:2], images)))
    # widths = np.round(IMAGE_HEIGHT / shapes[:,0] * shapes[:,1]).reshape([bs, 1])
    widths = np.array(np.ceil(IMAGE_HEIGHT / shapes[:,0] * shapes[:,1] / downsample_rate) * downsample_rate,
                      dtype=np.int).reshape([bs, 1])
    widths = np.minimum(widths, max_img_side)
    heights = np.ones([bs, 1]) * IMAGE_HEIGHT
    shapes = np.asarray(np.concatenate([heights, widths], axis=1), np.int32)
    # w_max = np.max(widths)
    w_max = max_img_side
    if w_max < MIN_WIDTH:
        w_max = MIN_WIDTH
    max_im_w = int(w_max + IMAGE_HEIGHT)
    img_canvas = np.zeros([bs, IMAGE_HEIGHT, max_im_w, channels], dtype=np.float32)

    for i, img in enumerate(images):
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = shapes[i]
        img = cv2.resize(img, (w, h))
        if channels == 1:
            img = np.expand_dims(img, -1)
        if w < MIN_WIDTH:
            diff = MIN_WIDTH - w
            pad_left = pad_right = int(diff / 2)
            if diff % 2 == 1:
                pad_right += 1
            img = np.pad(img, [(0, 0), (pad_left, pad_right), (0, 0)], 'constant', constant_values=255)
            w = MIN_WIDTH
            shapes[i][1] = MIN_WIDTH
        img_canvas[i,:,:w,:] = img / 255
    return img_canvas, shapes


def images_preprocess(src_dir, dst_dir):
    if not os.path.exists(dst_dir + '/inputs/bin'):
        os.makedirs(dst_dir + '/inputs/bin')
        os.makedirs(dst_dir + '/inputs/shape')
        os.makedirs(dst_dir + '/inputs_shape/bin')
        os.makedirs(dst_dir + '/inputs_shape/shape')
        os.makedirs(dst_dir + '/name')

    names = load_files(src_dir)
    # sort image list
    imgs = [cv2.imread(os.path.join(src_dir, name)) for name in names]
    imgs_ratio = [img.shape[1]/img.shape[0] for img in imgs]
    idxs = np.argsort(imgs_ratio)
    names = np.array(names)[idxs]

    batch_size = 64
    NB = int(len(names) / batch_size)
    N = len(names) - NB * batch_size
    batch_names = []
    for i in range(NB):
        batch_names.append(names[i*batch_size:i*batch_size+batch_size])
    if N > 0:
        batch_names.append(names[NB*batch_size:])

    cnt = 0
    for batch_name in batch_names:
        im_list = []
        N = len(batch_name)
        for name in batch_name:
            im = cv2.imread(os.path.join(src_dir, name))
            im_list.append(im)

        inputs, inputs_shape = preprocess_recog_batch(im_list, channels=3)

        inputs = np.transpose(inputs, [0,3,1,2])

        shape = list(inputs.shape)

        inputs_name = dst_dir + '/inputs/bin/batched_' + str(cnt).zfill(3)
        shape_name = dst_dir + '/inputs/shape/batched_' + str(cnt).zfill(3)
        inputs.astype(np.float32).tofile(inputs_name)
        np.array(shape,dtype=np.int32).tofile(shape_name)

        shape = list(inputs_shape.shape)
        inputs_shape_name = dst_dir + '/inputs_shape/bin/batched_' + str(cnt).zfill(3)
        shape_name = dst_dir + '/inputs_shape/shape/batched_' + str(cnt).zfill(3)
        inputs_shape.astype(np.int32).tofile(inputs_shape_name)
        np.array(shape,dtype=np.int32).tofile(shape_name)

        name = dst_dir + '/name/batched_' + str(cnt).zfill(3)+'.npy'
        np.save(name, np.array(batch_name))
        cnt += 1
        print(cnt)


def decodeCTC(chars, pred):
    t = pred.argmax(axis=-1)
    length = len(t)
    char_list = []
    n = len(chars)
    for i in range(length):
       if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
           char_list.append(chars[t[i]])
    return ''.join(char_list)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def post_process(src_dir, name_dir, char_path, dst_dir, transpose):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(char_path, 'r') as f:
        CHARS = ''.join(f.read().strip().splitlines())
    names = load_files(src_dir+'/bin')
    memo = {}
    for name in names:
        bin_name = src_dir + '/bin/' + name
        shape_name = src_dir + '/shape/' + name
        shape = np.fromfile(shape_name, dtype=np.int32)
        s = shape.astype(np.int32)
        feat = np.fromfile(bin_name,dtype=np.float32).reshape(s)
        if transpose:
            feat = np.transpose(feat, [0, 2, 1])

        name_name = name_dir + '/' + name
        batched_name = np.load(name_name+'.npy')
        N = feat.shape[0]
        for i in range(N):
            res = decodeCTC(CHARS, feat[i])
            memo[batched_name[i]] = {'value' : [res]}

    with open(os.path.join(dst_dir, 'res.txt'), 'w') as f:
        json.dump(memo, f, ensure_ascii=False)


if __name__ == "__main__":
    # im_dir = '/home/gulixin/workspace/datasets/ocr-recognition-data/std_recog_socr_v1.1/images'
    # inputs_dir = '/home/gulixin/workspace/nn_predictor/tensorrt/tests/res/trans_ctc/im_raw_gray_sort_socr_fix_shape_channel3'
    # images_preprocess(im_dir, inputs_dir)

    feat_dir = '../../../build/test_data/ocr_trans_ctc_data/ctc_revive_v1.1_fp16'
    name_dir = '../../../build/test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_channel3/name'
    char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset_v2.txt'
    dst_dir = 'trans_ctc_v1.1_socr_fp16_tf'
    post_process(feat_dir, name_dir, char_path, dst_dir, True)


