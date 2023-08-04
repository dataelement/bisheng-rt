import tensorflow as tf
import numpy as np
import os
import pickle

def preprocess(filepath):
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=[224, 224])
    x = tf.keras.preprocessing.image.img_to_array(img)
    return x

def main(src_dir, dst_dir):
    if not os.path.exists(dst_dir + '/bin'):
        os.makedirs(dst_dir + '/bin')
    if not os.path.exists(dst_dir + '/shape'):
        os.makedirs(dst_dir + '/shape')
    names = os.listdir(src_dir)
    names = [xx for xx in names if not xx.startswith('.')]

    for name in names:
        im_path = os.path.join(src_dir, name)
        im = preprocess(im_path)
        resized_h, resized_w, c = im.shape
        name = name[:-4]
        print(name, im.shape)
        im_name = dst_dir + '/bin/' + name
        shape_name = dst_dir + '/shape/' + name
        im.astype(np.float32).tofile(im_name)
        np.array([resized_h, resized_w, c], dtype=np.float32).tofile(shape_name)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    src_dir = '../../../build/test_data/ocr_classification_data/infer4071'
    data_dir = '../../../build/test_data/ocr_classification_data/im_raw_fix_nhwc'
    main(src_dir, data_dir)
