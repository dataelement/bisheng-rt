import numpy as np
import os

def load_feat(src_dir, name):
    bin_name = src_dir + '/bin/' + name
    shape_name = src_dir + '/shape/' + name
    shape = np.fromfile(shape_name,dtype=np.int32)
    feat = np.fromfile(bin_name,dtype=np.float32).reshape(shape)
    return feat

def main(src_dir, label_dir):
    names = os.listdir(src_dir+'/score/bin')
    names = [xx for xx in names if not xx.startswith('.')]
    gt_dict = {}
    with open(label_dir, 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            line = line.strip()
            name, label = line.split('\t')
            gt_dict[name[:-4]] = int(label)

    cor_num = 0
    for name in names:
        score = load_feat(src_dir+'/score', name).squeeze()
        pred = np.argmax(score)
        cmp = pred == gt_dict[name]
        cor_num += cmp
    print(cor_num/len(names))

if __name__ == '__main__':
    src_dir = "../../../build/test_data/ocr_classification_data/sampleTrtCLASSIFICATION_fix_trt_fp32"
    label_dir = '../../../build/test_data/ocr_classification_data/whole_gt_20201222.txt'
    main(src_dir, label_dir)
