import numpy as np
import os
import json
import cv2
import tensorflow as tf
from transformer_model import predicted_ids_with_eos_to_string_v2, finalize
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

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
    batch_image_names = []
    for batch_name in batch_names:
        im_list = []
        N = len(batch_name)
        for name in batch_name:
            im = cv2.imread(os.path.join(src_dir, name))
            im_list.append(im)

        inputs, inputs_shape = preprocess_recog_batch(im_list)

        # inputs = np.transpose(inputs, [0,3,1,2])

        batch_image_names.append('batched_' + str(cnt).zfill(3))

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

    with open(os.path.join(dst_dir, 'img_list_sort_decent.txt'), 'w') as f:
        for batch_image_name in batch_image_names[::-1]:
            f.write(batch_image_name + '\n')


def post_infer(output_ids_input, parent_ids_input, sequence_length_input, chars):
    beam_width = 5
    end_id = 1
    parent_ids_input = parent_ids_input % beam_width
    finalized_output_ids, finalized_sequence_lengths = finalize(beam_width,
                                                                parent_ids_input,
                                                                sequence_length_input,
                                                                output_ids_input,
                                                                end_id,
                                                                tf.shape(output_ids_input)[0])

    sentences_id = finalized_output_ids[:, 0, :]
    sentences_tf, _ = predicted_ids_with_eos_to_string_v2(sentences_id, chars)
    sentences_tf = tf.identity(sentences_tf, name="while/Exit_1")

    return sentences_tf


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def post_process(src_dir, data_dir, char_path, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    chars = load_charset(char_path, 'transformer')

    output_ids_input = tf.placeholder(tf.int32, shape=([None, None, 5]), name='output_ids')
    parent_ids_input = tf.placeholder(tf.int32, shape=([None, None, 5]), name='parent_ids')
    sequence_length_input = tf.placeholder(tf.int32, shape=([None, 5]), name='sequence_length')
    sentences_tf = post_infer(output_ids_input, parent_ids_input, sequence_length_input, chars)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)

    names = load_files(src_dir+'/output_ids/bin')
    memo = {}
    for name in names:
        bin_name = src_dir + '/output_ids/bin/' + name
        shape_name = src_dir + '/output_ids/shape/' + name
        shape = np.fromfile(shape_name, dtype=np.int32)
        s = shape.astype(np.int32)
        output_ids = np.fromfile(bin_name,dtype=np.int32).reshape(s)

        bin_name = src_dir + '/parent_ids/bin/' + name
        shape_name = src_dir + '/parent_ids/shape/' + name
        shape = np.fromfile(shape_name, dtype=np.int32)
        s = shape.astype(np.int32)
        parent_ids = np.fromfile(bin_name,dtype=np.int32).reshape(s)

        bin_name = src_dir + '/sequence_length/bin/' + name
        shape_name = src_dir + '/sequence_length/shape/' + name
        shape = np.fromfile(shape_name, dtype=np.int32)
        s = shape.astype(np.int32)
        sequence_length = np.fromfile(bin_name,dtype=np.int32).reshape(s)

        outputs_ = sess.run(sentences_tf, feed_dict={output_ids_input:output_ids, parent_ids_input:parent_ids,
                                                     sequence_length_input:sequence_length})
        outputs_ = list(map(lambda x : x.decode(), outputs_))
        print(outputs_)

        name_name = os.path.join(data_dir, 'name') + '/' + name
        batched_name = np.load(name_name+'.npy')
        for index, image_name in enumerate(batched_name):
            memo[image_name] = {'value' : [outputs_[index]]}

    with open(os.path.join(dst_dir, 'res.txt'), 'w') as f:
        json.dump(memo, f, ensure_ascii=False)


def save_post_model(char_path):
    chars = load_charset(char_path, 'transformer')

    output_ids_input = tf.placeholder(tf.int32, shape=([None, None, 5]), name='output_ids')
    parent_ids_input = tf.placeholder(tf.int32, shape=([None, None, 5]), name='parent_ids')
    sequence_length_input = tf.placeholder(tf.int32, shape=([None, 5]), name='sequence_length')
    sentences_tf = post_infer(output_ids_input, parent_ids_input, sequence_length_input, chars)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)

    graph = tf.get_default_graph()
    output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                 graph.as_graph_def(), ["while/Exit_1"],
                                                                 variable_names_whitelist=None,
                                                                 variable_names_blacklist=None)

    with gfile.GFile('transformer_post.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()


if __name__ == "__main__":
    # im_dir = '/home/gulixin/workspace/datasets/std_recog_socr_v1.1/images'
    # inputs_dir = '../../../build/test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_hwc'
    # images_preprocess(im_dir, inputs_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    feat_dir = '../../../build/test_data/ocr_trans_ctc_data/transformer_v2.8_socr_trt_fp16_hwc/'
    data_dir = '../../../build/test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_hwc'
    char_path = '../../../build/test_data/4pd_charset.txt'
    dst_dir = 'transformer_socr_fp16_op_hwc'
    post_process(feat_dir, data_dir, char_path, dst_dir)

    # char_path = '/home/gulixin/workspace/datasets/ocr-recognition-data/STD-charset/4pd_charset.txt'
    # save_post_model(char_path)

