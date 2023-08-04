import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import cv2
import numpy as np
import math
np.set_printoptions(threshold=1e6, suppress=True)

DIC_MAP = {}
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d/kernel:0'] = 'layer_0/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d/bias:0'] = 'layer_0/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d_1/kernel:0'] = 'layer_0/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d_1/bias:0'] = 'layer_0/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/q/kernel:0'] = 'layer_0/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/k/kernel:0'] = 'layer_0/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/v/kernel:0'] = 'layer_0/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/output_transform/kernel:0'] = 'layer_0/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/filter_layer/kernel:0'] = 'layer_0/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/filter_layer/bias:0'] = 'layer_0/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/output_layer/kernel:0'] = 'layer_0/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/output_layer/bias:0'] = 'layer_0/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/layer_normalization/layer_norm_scale:0'] = 'layer_0/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/layer_normalization/layer_norm_bias:0'] = 'layer_0/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/layer_normalization_1/layer_norm_scale:0'] = 'layer_0/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/layer_normalization_1/layer_norm_bias:0'] = 'layer_0/attention/self/LayerNorm_2/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d/kernel:0'] = 'layer_1/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d/bias:0'] = 'layer_1/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d_1/kernel:0'] = 'layer_1/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d_1/bias:0'] = 'layer_1/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/q/kernel:0'] = 'layer_1/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/k/kernel:0'] = 'layer_1/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/v/kernel:0'] = 'layer_1/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/output_transform/kernel:0'] = 'layer_1/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/filter_layer/kernel:0'] = 'layer_1/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/filter_layer/bias:0'] = 'layer_1/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/output_layer/kernel:0'] = 'layer_1/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/output_layer/bias:0'] = 'layer_1/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/layer_normalization/layer_norm_scale:0'] = 'layer_1/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/layer_normalization/layer_norm_bias:0'] = 'layer_1/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/layer_normalization/layer_norm_scale:0'] = 'layer_1/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/layer_normalization/layer_norm_bias:0'] = 'layer_1/attention/self/LayerNorm_2/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d/kernel:0'] = 'layer_2/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d/bias:0'] = 'layer_2/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d_1/kernel:0'] = 'layer_2/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d_1/bias:0'] = 'layer_2/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/q/kernel:0'] = 'layer_2/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/k/kernel:0'] = 'layer_2/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/v/kernel:0'] = 'layer_2/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/output_transform/kernel:0'] = 'layer_2/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/filter_layer/kernel:0'] = 'layer_2/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/filter_layer/bias:0'] = 'layer_2/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/output_layer/kernel:0'] = 'layer_2/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/output_layer/bias:0'] = 'layer_2/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/layer_normalization/layer_norm_scale:0'] = 'layer_2/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/layer_normalization/layer_norm_bias:0'] = 'layer_2/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/layer_normalization/layer_norm_scale:0'] = 'layer_2/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/layer_normalization/layer_norm_bias:0'] = 'layer_2/attention/self/LayerNorm_2/beta:0'

DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/layer_normalization/layer_norm_bias:0'] = 'layer_2/attention/self/LayerNorm_2/beta:0'


def save_weights(sess, variables):
    dic = {}
    for var in variables:
        value = sess.run(var)
        print(str(var.name) + " " + str(var.shape) + " " + str(var.dtype))
        if str(var.name) == 'Transformer/encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_scale:0':
            dic['layer_0/attention/self/LayerNorm/gamma:0'] = value
            dic['layer_0/attention/self/LayerNorm_1/gamma:0'] = value
            dic['layer_1/attention/self/LayerNorm/gamma:0'] = value
            dic['layer_1/attention/self/LayerNorm_1/gamma:0'] = value
            dic['layer_2/attention/self/LayerNorm/gamma:0'] = value
            dic['layer_2/attention/self/LayerNorm_1/gamma:0'] = value
            dic['LayerNorm/gamma:0'] = value
        elif str(var.name) == 'Transformer/encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_bias:0':
            dic['layer_0/attention/self/LayerNorm/beta:0'] = value
            dic['layer_0/attention/self/LayerNorm_1/beta:0'] = value
            dic['layer_1/attention/self/LayerNorm/beta:0'] = value
            dic['layer_1/attention/self/LayerNorm_1/beta:0'] = value
            dic['layer_2/attention/self/LayerNorm/beta:0'] = value
            dic['layer_2/attention/self/LayerNorm_1/beta:0'] = value
            dic['LayerNorm/beta:0'] = value
        else:
            if var.name in DIC_MAP:
                dic[DIC_MAP[var.name]] = value
            else:
                dic[var.name] = value
    return dic


def load_savedmodel(model_path):
    config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    m = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=model_path)
    graph = tf.get_default_graph()
    signature = m.signature_def
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables_dict = save_weights(sess, variables)
    sess.close()

    return variables_dict


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model_path = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8-gamma/savedmodel'
    load_savedmodel(model_path)

