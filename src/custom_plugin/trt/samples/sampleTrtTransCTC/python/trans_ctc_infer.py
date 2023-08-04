import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import cv2
import numpy as np
import math
np.set_printoptions(threshold=1e6, suppress=True)
import time

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format):
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, blocks, strides, training, name, data_format):
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1,
            strides=strides, data_format=data_format)

    inputs = building_block_v2(inputs, filters, training, projection_shortcut, strides,
                    data_format)

    for _ in range(1, blocks):
      inputs = building_block_v2(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def build_resnet31(inputs):
    num_filters = 16
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=num_filters, kernel_size=3,
        strides=1, data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_conv')
    block_sizes = [5,5,5]
    block_strides = [1,2,2]
    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs, filters=num_filters,
                     blocks=num_blocks, strides=block_strides[i], training=False,
                     name='block_layer{}'.format(i + 1), data_format='channels_first')
        num_filters *= 2

    inputs = batch_norm(inputs, False, 'channels_first')
    inputs = tf.nn.relu(inputs)
    return inputs


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


_NEG_INF = -1e9
def get_padding(inputs, inputs_unpadded_length):
    with tf.name_scope("padding"):
        input_shape = tf.shape(inputs)
        indexs = tf.tile(tf.range(input_shape[1]), tf.expand_dims(input_shape[0], axis=0))
        indexs = tf.reshape(indexs, input_shape[:2])  # shape: [batch_size, input_length]
        inputs_unpadded_length = tf.tile(inputs_unpadded_length, tf.stack([1, input_shape[1]]))
        conditions = indexs < inputs_unpadded_length
        return 1 - tf.to_float(conditions)


def get_padding_bias(inputs, inputs_unpadded_length):
    with tf.name_scope("attention_bias"):
        padding = get_padding(inputs, inputs_unpadded_length)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def output_normalization(x, scale, bias, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def feed_forward_network(x, padding=None):
    allow_pad = True
    filter_size = 1024
    hidden_size = 512
    padding = None if not allow_pad else padding
    batch_size = x.get_shape()[0]
    length = tf.shape(x)[1]
    output = tf.layers.dense(x, filter_size, use_bias=True, activation=tf.nn.relu, name='filter_layer')
    output = tf.layers.dense(output, hidden_size, use_bias=True, name='output_layer')

    if padding is not None:
        padding = 1-padding # nopaddings are ones and paddings are zeros.
        padding = tf.expand_dims(padding, axis=-1)  # [batch_size, length, 1]
        padding = tf.tile(padding, [1,1,hidden_size] ) # [batch_size, length, hidden_size]
        output = tf.multiply(output, padding)
    return output


def split_heads(x, hidden_size, num_heads):
    with tf.name_scope("split_heads"):
        batch_size = tf.shape(x)[0]
        depth = (hidden_size // num_heads)
        x = tf.reshape(x, [batch_size, -1, num_heads, depth])
        return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x, hidden_size):
    with tf.name_scope("combine_heads"):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, -1, hidden_size])


def self_attention_layer(x, bias):
    y = x
    hidden_size = 512
    num_heads = 8
    q = tf.layers.dense(x, hidden_size, use_bias=False, name="q")
    k = tf.layers.dense(y, hidden_size, use_bias=False, name="k")
    v = tf.layers.dense(y, hidden_size, use_bias=False, name="v")

    q = split_heads(q, hidden_size, num_heads)
    k = split_heads(k, hidden_size, num_heads)
    v = split_heads(v, hidden_size, num_heads)

    #[b, num_heads, w, depth]
    depth = (hidden_size // num_heads)
    q *= depth**-0.5

    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    attention_output = tf.matmul(weights, v)
    attention_output = combine_heads(attention_output, hidden_size)

    attention_output = tf.layers.dense(attention_output, hidden_size, use_bias=False, name="output_transform")
    return attention_output


def encoder_stack2(encoder_inputs, attention_bias, inputs_padding):
    hidden_size = 512
    filter_size = 1024
    num_heads = 8
    attention_dropout = 0.1
    attention_width = -1
    attention_causal = False
    relu_dropout = 0.1
    allow_ffn_pad = True
    postprocess_dropout = 0.1
    num_hidden_layers = 3

    scale1 = tf.get_variable("encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_scale", [hidden_size],
                                     initializer=tf.ones_initializer())
    bias1 = tf.get_variable("encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_bias", [hidden_size],
                        initializer=tf.zeros_initializer())
    for i in range(num_hidden_layers):
        #self_attention_layer = SelfAttention(hidden_size, num_heads, attention_dropout, False, attention_width, attention_causal)
        with tf.variable_scope("encoder_stack/layer_%d" % i):
            with tf.variable_scope("self_attention"):
                inputs = encoder_inputs
                encoder_inputs = tf.layers.conv1d(encoder_inputs,
                                                  hidden_size,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding='same',
                                                  dilation_rate=1,
                                                  activation=tf.nn.relu)

                encoder_inputs = output_normalization(encoder_inputs, scale1, bias1)

                encoder_inputs = tf.layers.conv1d(encoder_inputs,
                                                  hidden_size,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding='same',
                                                  dilation_rate=1,
                                                  activation=tf.nn.relu)
                encoder_inputs = output_normalization(encoder_inputs, scale1, bias1)
                encoder_inputs += inputs

                if i == 0:
                    scale = tf.get_variable("layer_normalization_1/layer_norm_scale", [hidden_size],
                                         initializer=tf.ones_initializer())
                    bias = tf.get_variable("layer_normalization_1/layer_norm_bias", [hidden_size],
                            initializer=tf.zeros_initializer())
                else:
                    scale = tf.get_variable("layer_normalization/layer_norm_scale", [hidden_size],
                                         initializer=tf.ones_initializer())
                    bias = tf.get_variable("layer_normalization/layer_norm_bias", [hidden_size],
                            initializer=tf.zeros_initializer())
                shortcut = encoder_inputs

                encoder_inputs = output_normalization(encoder_inputs, scale, bias)
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                encoder_inputs = shortcut + encoder_inputs

            with tf.variable_scope("ffn"):
                scale = tf.get_variable("layer_normalization/layer_norm_scale", [hidden_size],
                                     initializer=tf.ones_initializer())
                bias = tf.get_variable("layer_normalization/layer_norm_bias", [hidden_size],
                                    initializer=tf.zeros_initializer())
                shortcut = encoder_inputs
                encoder_inputs = output_normalization(encoder_inputs, scale, bias)
                with tf.variable_scope("feed_foward_network"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)
                encoder_inputs = shortcut + encoder_inputs

    return output_normalization(encoder_inputs, scale1, bias1)


def encode(inputs, attention_bias, inputs_unpadded_length):
    hidden_size = 512
    filter_size = 1024
    num_heads = 8
    attention_dropout = 0.1
    attention_width = -1
    attention_causal = False
    relu_dropout = 0.1
    allow_ffn_pad = True
    postprocess_dropout = 0.1
    num_hidden_layers = 3
    #encoder_stack = EncoderStack(False, num_hidden_layers, hidden_size, num_heads, filter_size, attention_dropout, relu_dropout, postprocess_dropout, allow_ffn_pad, attention_width, attention_causal)
    with tf.name_scope("encode"):
        encoder_inputs = inputs
        with tf.name_scope("add_pos_encoding"):
            #length = encoder_inputs.get_shape()[1]
            length = tf.shape(encoder_inputs)[1]
            encoder_inputs += get_position_encoding(length, hidden_size)
        inputs_padding = get_padding(inputs, inputs_unpadded_length)
        return encoder_stack2(encoder_inputs, attention_bias, inputs_padding)


def build_trans_encode(inputs, inputs_unpadded_length):
    initializer = tf.variance_scaling_initializer(1.0, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
        attention_bias = get_padding_bias(inputs, inputs_unpadded_length)
        encoder_outputs = encode(inputs, attention_bias, inputs_unpadded_length)
        return encoder_outputs


def build_trans_ctc(inputs, inputs_shape):
    with tf.variable_scope('backbone'):
        inputs = build_resnet31(inputs)

    inputs = tf.layers.conv2d(inputs, filters=512, kernel_size=1, strides=1,
        name='ocr_transformer/post_conv', data_format='channels_first')
    inputs = tf.layers.batch_normalization(inputs, training=False, axis=1,
        fused=True, name="ocr_transformer/post_bn")

    inputs = tf.nn.relu(inputs, name='ocr_transformer/relu')

    b, c, h, w = inputs.get_shape().as_list()
    b = tf.shape(inputs)[0]
    w = tf.shape(inputs)[3]
    inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
    inputs = tf.reshape(inputs, [b, w, h*c])

    inputs = tf.layers.dense(inputs, 512, name='ocr_transformer/dense')
    inputs = tf.nn.relu(inputs, name='ocr_transformer/relu')

    outputs = inputs
    inputs_length = tf.cast(inputs_shape[:,1:2] / 4, dtype=tf.int32)
    outputs = build_trans_encode(inputs, inputs_length)

    outputs = tf.layers.dense(outputs, 6410, activation=None,
                kernel_initializer=tf.variance_scaling_initializer(),
                bias_initializer=tf.constant_initializer(), name='ocr_ctc/logits')
    return outputs


def load_weights(sess, variables, path):
    dic = np.load(path, allow_pickle=True)
    dic = dict(dic['dic'][()])
    fetches = []
    feeds = {}
    var_names = []
    for var in variables:
        var_names.append(var.name)
        if var.name in dic:
            fetches.append(var.initializer)
            feeds[var.initializer.inputs[1]] = dic[var.name]
        else:
            print(var.name, 'not exsits!')
    sess.run(fetches, feed_dict=feeds)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def tf_infer(src_dir, dst_dir, model_dir):
    if not os.path.exists(dst_dir + '/shape'):
        os.makedirs(dst_dir + '/bin')
        os.makedirs(dst_dir + '/shape')

    inputs = tf.placeholder(tf.float32, shape=([None, 3, 32, None]), name='inputs')
    inputs_shape = tf.placeholder(tf.int32, shape=([None, 2]), name='inputs_shape')
    outputs = build_trans_ctc(inputs, inputs_shape)
    outputs = tf.identity(outputs, name='outputs')

    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)

    load_weights(sess, all_variables, model_dir)

    # names = load_files(src_dir+'/inputs/bin')
    names = []
    with open(os.path.join(src_dir, 'img_list_sort_decent.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            names.append(line)

    t_total = 0
    cnt = 0
    start_cnt = False
    for name in names:
        bin_name = src_dir + '/inputs/bin/' + name
        shape_name = src_dir + '/inputs/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims = np.fromfile(bin_name,dtype=np.float32).reshape(s)

        imshape_name = src_dir + '/inputs_shape/bin/' + name
        shape_name = src_dir + '/inputs_shape/shape/' + name
        s = np.fromfile(shape_name, dtype=np.int32)
        ims_shape = np.fromfile(imshape_name,dtype=np.int32).reshape(s)
        t = time.time()
        outputs_ = sess.run(outputs, feed_dict={inputs:ims, inputs_shape:ims_shape})
        t_total += (time.time() - t) * 1000
        cnt += 1
        if cnt == 10 and not start_cnt:
            start_cnt = True
            t_total = 0
            cnt = 0

        if cnt > 0 and cnt % 10 == 0:
            print('t:', t_total, ' cnt:', cnt, ' t/per_im:', t_total/cnt)

        outputs_ = np.transpose(outputs_, [0,2,1])
        outputs_.astype(np.float32).tofile(dst_dir+'/bin/'+name)
        np.array(list(outputs_.shape), dtype=np.int32).tofile(dst_dir+'/shape/'+name)
    sess.close()
    print('t:', t_total, ' cnt:', cnt, ' t/per_im:', t_total/cnt)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_dir = '../../../build/test_data/models/ctc-revive/1.1/trans_ctc_weights_trt.npz'
    inputs_dir = '../../../build/test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_channel3'
    dst_dir = '../../../build/test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_fix_shape_channel3_tf'
    tf_infer(inputs_dir, dst_dir, model_dir)


