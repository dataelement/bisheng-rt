import os
import struct
import tensorflow as tf
import numpy as np

def save_weights(sess, variables):
    dic = {}
    for var in variables:
        value = sess.run(var)
        print(str(var.name) + " " + str(var.shape) + " " + str(var.dtype))
        if 'W:0' in str(var.name):
            new_var_name = str(var.name).replace('W:0', 'kernel:0')
            dic[new_var_name] = value
        elif 'b:0' in  str(var.name):
            new_var_name = str(var.name).replace('b:0', 'bias:0')
            dic[new_var_name] = value
        elif 'bn/mean/EMA:0' in  str(var.name):
            new_var_name = str(var.name).replace('bn/mean/EMA:0', 'BatchNorm/moving_mean:0')
            dic[new_var_name] = value
        elif 'bn/variance/EMA:0' in  str(var.name):
            new_var_name = str(var.name).replace('bn/variance/EMA:0', 'BatchNorm/moving_variance:0')
            dic[new_var_name] = value
        elif 'bn/gamma:0' in  str(var.name):
            new_var_name = str(var.name).replace('bn/gamma:0', 'BatchNorm/gamma:0')
            dic[new_var_name] = value
        elif 'bn/beta:0' in  str(var.name):
            new_var_name = str(var.name).replace('bn/beta:0', 'BatchNorm/beta:0')
            dic[new_var_name] = value
        else:
            dic[str(var.name)] = value

        if str(var.name) == 'maskrcnn/deconv/W:0':
            new_var_name = 'maskrcnn/deconv/conv2d_transpose/kernel:0'
            dic[new_var_name] = value

        if str(var.name) == 'maskrcnn/deconv/b:0':
            new_var_name = 'maskrcnn/deconv/conv2d_transpose/bias:0'
            dic[new_var_name] = value

    return dic


def float2str(data):
    return struct.pack('!f', data).hex()


def str2bytes(str_data):
    return bytes(str_data, encoding = 'utf-8')


def gen_trt_weights(dic, dst_name):
    f_write = open(dst_name, 'wb')
    f_write.write(str2bytes('\n'))
    lines = []
    for name, weights in dic.items():
        # print(name, weights.shape)
        # if ('conv' in name or 'Conv' in name or 'shortcut' in name) and 'weights' in name:
        if len(weights.shape) == 4:
            weights = np.transpose(weights, [3,2,0,1])
        # print(name, weights.shape)
        if len(weights.shape) == 2:
            weights = np.transpose(weights, [1, 0])
            print (name, weights.shape)

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


def convert_model(src_name, dst_name):
    config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    m = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=src_name)
    graph = tf.get_default_graph()

    signature = m.signature_def
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

    save_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables_dic = save_weights(sess, save_variables)
    np.savez(os.path.join(src_name, 'ocr_maskrcnn_weights.npz'), dic=variables_dic)

    gen_trt_weights(variables_dic, trt_weights_path)
    print ("Generate trt weights file", trt_weights_path)
    sess.close()


if __name__ == '__main__':
    model_path = '/home/public/OCR-DETECTION-MODELS/maskrcnn_general_ch_v5.1/1'
    trt_weights_path = '/home/public/OCR-DETECTION-MODELS/maskrcnn_general_ch_v5.1/ocr_maskrcnn_weights.wts'
    convert_model(model_path, trt_weights_path)


