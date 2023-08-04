import os
import tensorflow as tf
import numpy as np
import pickle
import struct

def float2str(data):
    return struct.pack('!f', data).hex()

def str2bytes(str_data):
    return bytes(str_data, encoding = 'utf-8')

def gen_trt_weights(dic, dst_name):
    f_write = open(dst_name, 'wb')
    f_write.write(str2bytes('\n'))
    lines = []
    for name, weights in dic.items():
        print(name, weights.shape)
        if len(weights.shape) == 4:
            weights = np.transpose(weights, [3,2,0,1])
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

def main(ckpt_path, dst_path):
    f = open(ckpt_path, 'rb')
    value_dict = pickle.load(f)
    gen_trt_weights(value_dict, dst_path)

if __name__ == '__main__':
    ckpt_path = '/home/yujinbiao/code/trt-lib/build/test_data/ocr_classification_data/hw_print-20201223-gap-resize224-resNet50V2-generate-tf2.3-save-weights.pkl'
    trt_weights_path = '/home/yujinbiao/code/trt-lib/build/test_data/ocr_classification_data/classification.wts'
    main(ckpt_path, trt_weights_path)

