import os
import argparse
import struct
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import image_ops
from tensorflow.contrib import slim
from nets import resnet_v1
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def float2str(data):
    return struct.pack('!f', data).hex()

def str2bytes(str_data):
    return bytes(str_data, encoding = 'utf-8')

def unpool(inputs, data_format=None):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def image_preprocess(image, mean=[123.68, 116.78, 103.94], data_format=None):
    means = tf.constant(mean)
    if data_format == "NCHW":
        image = tf.transpose(image, [0, 2, 3, 1])
        image = image - means
        image = tf.transpose(image, [0, 3, 1, 2])
    else:
        image = image - means
    return image

def build_east(inputs, is_training=False, text_scale=896, data_format="NHWC"):
    '''
    define the model, we use slim's implemention of resnet
    '''

    weight_decay=1e-5
    axis = 1 if data_format == "NCHW" else 3
    images = image_preprocess(inputs, data_format=data_format)
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay, data_format=data_format)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, output_stride=16, scope='resnet_v1_50', data_format=data_format)

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'data_format': data_format
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            data_format=data_format
                            ):
            f = [end_points['pool5'], end_points['pool3'], end_points['pool2']]
            for i in range(3):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [256, 64, 32]

            ASPP1 = slim.conv2d(f[0], num_outputs[0], 1)
            ASPP2 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=3)
            ASPP3 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=6)
            ASPP4 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=9)
            ASPP5 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=12)
            ASPP6 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=15)
            ASPP7 = slim.conv2d(f[0], num_outputs[0], [3, 3], rate=18)
            h[0] = slim.conv2d(tf.concat([ASPP1, ASPP2, ASPP3, ASPP4, ASPP5, ASPP6, ASPP7], axis=axis), num_outputs[0], 1)
            if data_format == "NCHW":
                with tf.name_scope("uppool_scope%s"%(0)):
                    h[0] = tf.transpose(h[0], [0, 2, 3, 1])
                    g[0] = unpool(h[0], data_format)
                    g[0] = tf.transpose(g[0], [0, 3, 1, 2])
                    # h[0] = tf.transpose(h[0], [0, 3, 1, 2])
            else:
                g[0] = unpool(h[0], data_format)
            print('Shape of h_{} {}, g_{} {}'.format(0, h[0].shape, 0, g[0].shape))

            c1_1 = slim.conv2d(tf.concat([g[0], f[1]], axis=axis), num_outputs[1], 1)
            h[1] = slim.conv2d(c1_1, num_outputs[1], [3, 3])
            if data_format == "NCHW":
                with tf.name_scope("uppool_scope%s"%(1)):
                    h[1] = tf.transpose(h[1], [0, 2, 3, 1])
                    g[1] = unpool(h[1], data_format)
                    g[1] = tf.transpose(g[1], [0, 3, 1, 2])
                    # h[1] = tf.transpose(h[1], [0, 3, 1, 2])
            else:
                g[1] = unpool(h[1], data_format)
            print('Shape of h_{} {}, g_{} {}'.format(1, h[1].shape, 1, g[1].shape))

            c1_1 = slim.conv2d(tf.concat([g[1], f[2]], axis=axis), num_outputs[2], 1)
            h[2] = slim.conv2d(c1_1, num_outputs[2], [3, 3])
            g[2] = slim.conv2d(h[2], num_outputs[2], [3, 3])
            print('Shape of h_{} {}, g_{} {}'.format(2, h[2].shape, 2, g[2].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[2], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[2], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * text_scale
            angle_map = (slim.conv2d(g[2], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=axis)

            F_cos_map = slim.conv2d(g[2], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            F_sin_map = slim.conv2d(g[2], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    return F_score, F_geometry, F_cos_map, F_sin_map

def gen_trt_weights_east(dic, dst_name):
    f_write = open(dst_name, 'wb')
    f_write.write(str2bytes('\n'))
    lines = []
    for name, weights in dic.items():
        print(name, weights.shape)
        if ('conv' in name or 'Conv' in name or 'shortcut' in name) and 'weights' in name:
            weights = np.transpose(weights, [3,2,0,1])
        size = 1
        if 'weights' in name:
            name = name.replace('weights', 'kernel')
        if 'bias' in name:
            name = name.replace('biases', 'bias')
        for s in weights.shape:
            size *= s
        line = name + ' ' + str(0) + ' ' + str(size)
        weights = weights.reshape([-1])
        for i in range(size):
            line += ' ' + float2str(weights[i])
        lines.append(line+'\n')
    f_write.write(str2bytes(str(len(lines)) + '\n'))
    for line in lines:
        f_write.write(str2bytes(line))
    f_write.close()

def main():
    inputs = tf.placeholder(tf.float32, shape=([None, None, None, 3]), name='inputs')
    F_score, F_geometry, F_cos_map, F_sin_map = build_east(inputs)
    config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    dic = {}
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        value = sess.run(var)
        print(str(var.name) + " " + str(var.shape) + " " + str(var.dtype))
        dic[var.name] = value
    gen_trt_weights_east(dic, trt_weights_path)
    print ("Generate trt weights file", trt_weights_path)

if __name__ == '__main__':
    ckpt_path = '../../../build/test_data/models/east_v5_angle/model-450000'
    trt_weights_path = '../../../build/test_data/models/east_v5_angle.wts'
    main()
