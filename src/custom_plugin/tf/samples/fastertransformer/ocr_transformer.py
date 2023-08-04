import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import cv2
import numpy as np
import math
from ocr_transformer_model import transformer_model
import time
import json
np.set_printoptions(threshold=1e6, suppress=True)


def load_charset(charsetpath, mode):
    assert mode in ['ctc', 'transformer'], print('invalid charset mode')
    with open(charsetpath) as f:
        lines = f.readlines()
        lines = list(map(lambda x : x.replace('\n', ''), lines))
        chars = [xx for xx in lines if len(xx) > 0]

    if len(chars) == 1:
        chars = list(chars[0])

    if mode == 'ctc':
        chars = chars + ['']
    elif mode == 'transformer':
        if '卍' not in chars:
            chars = ['卍',] + chars
        chars = ['[PAD]', '[EOS]'] + chars
    return chars


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
    w_max = np.max(widths)
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

        inputs, inputs_shape = preprocess_recog_batch(im_list)

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


def build_transformer(inputs, inputs_shape, chars, is_resnet_vd):
    target_vocab_size = len(chars)
    print('target_vocab_size:', target_vocab_size)
    if is_resnet_vd:
        inputs = transformer_model.build_resnet_vd(inputs)
        downsample = 4
    else:
        with tf.variable_scope('resnet_model'):
            inputs = transformer_model.build_resnet50(inputs)
            downsample = 8

    inputs = tf.layers.conv2d(inputs, filters=512, kernel_size=1, strides=1,
        name='ocr_transformer/backbone/post_conv', data_format='channels_first')
    inputs = tf.layers.batch_normalization(inputs, training=False, axis=1,
        fused=True, name="post_bn")

    inputs = tf.nn.relu(inputs)

    b, c, h, w = inputs.get_shape().as_list()
    b = tf.shape(inputs)[0]
    w = tf.shape(inputs)[3]
    inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
    inputs = tf.reshape(inputs, [b, w, h*c])

    inputs = tf.layers.dense(inputs, 512, name='ocr_transformer/backbone/dense')
    inputs = tf.nn.relu(inputs)

    inputs_length = tf.cast(inputs_shape[:,1:2] / downsample, dtype=tf.int32)
    logits, decoded, encoder_outputs = transformer_model.build_trans(inputs, inputs_length, target_vocab_size)
    prob_matrix = decoded['prob_matrix']
    scores = decoded['scores']
    sentences_id = decoded['outputs']
    all_decoded_res = decoded['all_decoded_res']
    all_decoded_scores = decoded['all_decoded_scores']
    debug_output = decoded['debug_output']

    sentences, _ = transformer_model.predicted_ids_with_eos_to_string_v2(sentences_id, chars)

    return sentences, all_decoded_res, all_decoded_scores, encoder_outputs, debug_output


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def tf_infer_transformer(src_dir, dst_dir, model_path, chars_path, is_resnet_vd):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    chars = load_charset(chars_path, 'transformer')

    inputs = tf.placeholder(tf.float32, shape=([None, 1, 32, None]), name='inputs')
    inputs_shape = tf.placeholder(tf.int32, shape=([None, 2]), name='inputs_shape')
    outputs, all_decoded_res, all_decoded_scores, encoder_outputs, debug_output = build_transformer(inputs, inputs_shape, chars, is_resnet_vd)
    outputs = tf.identity(outputs, name='outputs')

    config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    names = load_files(src_dir+'/inputs/bin')

    t_total = 0
    cnt = 0
    start_cnt = False
    memo = {}
    for name in names:
        bin_name = src_dir + '/inputs/bin/' + name
        shape_name = src_dir + '/inputs/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims = np.fromfile(bin_name,dtype=np.float32).reshape(s)
        # ims = ims.transpose(0, 2, 3, 1)

        imshape_name = src_dir + '/inputs_shape/bin/' + name
        shape_name = src_dir + '/inputs_shape/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims_shape = np.fromfile(imshape_name, dtype=np.int32).reshape(s)
        t = time.time()
        outputs_ = sess.run(outputs, feed_dict={inputs:ims, inputs_shape:ims_shape})
        outputs_ = list(map(lambda x : x.decode(), outputs_))
        # print(outputs_)

        t_total += (time.time() - t) * 1000
        cnt += 1
        if cnt == 10 and not start_cnt:
            start_cnt = True
            t_total = 0
            cnt = 0

        if cnt > 0 and cnt % 10 == 0:
            print('t:', t_total, ' cnt:', cnt, ' t/per_im:', t_total/cnt)

        name_name = os.path.join(src_dir, 'name') + '/' + name
        batched_name = np.load(name_name+'.npy')
        for index, image_name in enumerate(batched_name):
            memo[image_name] = {'value' : [outputs_[index]]}

    with open(os.path.join(dst_dir, 'res.txt'), 'w') as f:
        json.dump(memo, f, ensure_ascii=False)

    sess.close()
    print('t:', t_total, ' cnt:', cnt, ' t/per_im:', t_total/cnt)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    im_dir = '/home/gulixin/workspace/datasets/ocr-recognition-data/std_recog_socr_v1.1/images'
    inputs_dir = '/home/gulixin/workspace/nn_predictor/tensorrt/tests/res/trans_ctc/im_raw_gray_sort_socr'
    # images_preprocess(im_dir, inputs_dir)

    is_resnet_vd = True
    # model_path = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.7/ckpt/model-1010000'
    model_path = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8-gamma/ckpt/model'
    res_dir = 'socr_tf_fp32_origin'
    # char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset.txt'
    char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset_v2.txt'
    tf_infer_transformer(inputs_dir, res_dir, model_path, char_path, is_resnet_vd)




