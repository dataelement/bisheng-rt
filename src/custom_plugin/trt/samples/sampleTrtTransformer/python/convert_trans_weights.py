import os
import numpy as np
import struct
import tensorflow as tf


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


def save_weights(sess, variables):
    dic = {}
    for var in variables:
        value = sess.run(var)
        # print(str(var.name) + " " + str(var.shape) + " " + str(var.dtype))
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


def str2float(str_data):
    return struct.unpack('!f', bytes.fromhex(str_data))[0]


def float2str(data):
    return struct.pack('!f', data).hex()


def str2bytes(str_data):
    return bytes(str_data, encoding = 'utf-8')


def gen_trt_weights(variables_dic, dst_name, ignore=None):
    f_write = open(dst_name, 'wb')
    f_write.write(str2bytes('\n'))
    lines = []
    for name, weights in variables_dic.items():
        if ignore == None or ignore not in name:
            print(name, weights.shape)
            if 'conv2d' in name and 'kernel' in name:
                weights = np.transpose(weights, [3,2,0,1])
            elif 'post_conv/kernel' in name:
                weights = np.transpose(weights, [3,2,0,1])
            elif 'conv1d' in name and 'kernel' in name:
                weights = np.transpose(weights, [2,1,0])
            elif 'ocr_ctc/logits/kernel' in name:
                weights = np.transpose(weights, [1,0])
            elif ('dense' in name or 'query' in name or 'key' in name or 'value' in name) and 'kernel' in name:
                weights = np.transpose(weights, [1,0])

        size = 1
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


def convert_model(src_name, dst_name, ignore=None):

    config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    m = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=src_name)
    graph = tf.get_default_graph()

    signature = m.signature_def
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    variables_dic = save_weights(sess, save_variables)
    np.savez(os.path.join(src_name, 'transformer_weights.npz'), dic=variables_dic)
    gen_trt_weights(variables_dic, dst_name, ignore)

    sess.close()


def loadweight(src_name):
    cnt = 0
    for line in open(src_name, 'rb'):
        line = line.strip()
        if len(line) == 0:
            continue
        cnt += 1
        line = str(line, encoding = 'utf-8')
        lines = line.split()
        if cnt == 1:
            count = int(lines[0])
            print(count)
        else:
            name = lines[0]
            data_type = int(lines[1])
            size = int(lines[2])
            print(name, size, len(lines))
            #fval = str2float(lines[3])
            #print(lines[3], fval, float2str(fval))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    src_name = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8/tf/savedmodel'
    dst_name = '/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8/tf/savedmodel/transformer_weights_trt_op.wts'
    convert_model(src_name, dst_name, ignore='layer_')
    loadweight(dst_name)

