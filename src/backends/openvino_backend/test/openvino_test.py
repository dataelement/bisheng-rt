# flake8: noqa
import copy
import math
import os
import time

import cv2
import numpy as np
from openvino.runtime import Core
from paddleocr.ppocr.data import create_operators, transform
from paddleocr.ppocr.postprocess import build_post_process


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                  (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def get_rotate_crop_image(img, points):
    assert len(points) == 4, 'shape of points must be 4*2'
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img,
                                  M, (img_crop_width, img_crop_height),
                                  borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    is_roted = 0
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        is_roted = 1
    return dst_img, is_roted


def preprocess(img, w, max_w, imageH=48):
    w = min(w, max_w)
    img_C = img.shape[-1]
    processed_img = img.astype('float32')
    processed_img = cv2.resize(processed_img, (w, imageH))
    processed_img = processed_img.transpose((2, 0, 1)) / 255
    processed_img -= 0.5
    processed_img /= 0.5
    padding_im = np.zeros((img_C, imageH, max_w), dtype=np.float32)
    padding_im[:, :, :w] = processed_img
    return padding_im


def load_chars(filename):
    res = [' ']
    with open(filename, 'rb') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip('\n').strip('\r\n')
            res.append(line)
    return res + [' ']


def get_text(inds, probs, chars):
    text = ''
    prob = 0
    cnt = 0
    for i in range(inds.shape[0]):
        if inds[i] == 0:
            continue
        text += chars[inds[i]]
        prob += probs[i]
        cnt += 1
    return text, prob / cnt


def det():
    pre_process_list = [{
        'DetResizeForTest': {
            'limit_side_len': 960,
            'limit_type': 'max',
        }
    }, {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225],
            'mean': [0.485, 0.456, 0.406],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }, {
        'ToCHWImage': None
    }, {
        'KeepKeys': {
            'keep_keys': ['image', 'shape']
        }
    }]

    postprocess_params = {}
    postprocess_params['name'] = 'DBPostProcess'
    postprocess_params['thresh'] = 0.3
    postprocess_params['box_thresh'] = 0.6
    postprocess_params['max_candidates'] = 1000
    postprocess_params['unclip_ratio'] = 1.5
    postprocess_params['use_dilation'] = False
    postprocess_params['score_mode'] = 'fast'

    preprocess_op = create_operators(pre_process_list)
    postprocess_op = build_post_process(postprocess_params)

    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_det_infer'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer = compiled_model.output(0)

    img_name = '/home/liuqingjie/images/1.jpg'
    img = cv2.imread(img_name)
    ori_img = img.copy()
    data = {'image': img}
    data = transform(data, preprocess_op)
    img, shape_list = data
    img = np.expand_dims(img, axis=0)
    shape_list = np.expand_dims(shape_list, axis=0)

    fs = cv2.FileStorage('det_im.ext', cv2.FILE_STORAGE_WRITE)
    fs.write('img', img)

    print(img.shape, np.sum(img), img.reshape(-1)[:16])
    ie_result = compiled_model([img])[output_layer]
    print(ie_result.shape, np.sum(ie_result), ie_result.reshape(-1)[:16])

    preds = {}
    preds['maps'] = ie_result
    post_result = postprocess_op(preds, shape_list)
    dt_boxes = post_result[0]['points']
    dt_boxes = filter_tag_det_res(dt_boxes, ori_img.shape)
    dt_boxes = sorted_boxes(dt_boxes)
    dt_boxes = np.expand_dims(np.array(dt_boxes),
                              axis=0) if len(dt_boxes) > 0 else np.empty(
                                  [1, 0, 4, 2], dtype=np.float32)
    print(dt_boxes)

    fs2 = cv2.FileStorage('det_feat.ext', cv2.FILE_STORAGE_READ)
    data2 = fs2.getNode('feat_map').mat()
    print(data2.shape, np.sum(data2), data2.reshape(-1)[:16])
    preds2 = {}
    preds2['maps'] = data2
    post_result2 = postprocess_op(preds2, shape_list)
    dt_boxes2 = post_result2[0]['points']
    dt_boxes2 = filter_tag_det_res(dt_boxes2, ori_img.shape)
    dt_boxes2 = sorted_boxes(dt_boxes2)
    dt_boxes2 = np.expand_dims(np.array(dt_boxes2),
                               axis=0) if len(dt_boxes2) > 0 else np.empty(
                                   [1, 0, 4, 2], dtype=np.float32)
    return ori_img, dt_boxes[0]


def cls(img, bboxes, imageH=48, maxW=192):
    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_ppocr_mobile_v2.0_cls_infer'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer = compiled_model.output(0)

    bbox_num = bboxes.shape[0]
    img_list = []
    rot_list = []
    width_list = []
    cropped_img_list = []
    for i in range(bbox_num):
        tmp_box = copy.deepcopy(bboxes[i])
        cropped_img, roted = get_rotate_crop_image(img, tmp_box)
        img_list.append(cropped_img)
        rot_list.append(roted)
        h, w = cropped_img.shape[:2]
        width_list.append(math.ceil(1.0 * imageH * w / h))
    width_list = np.array(width_list).astype(np.int32)
    indices = np.argsort(width_list)
    width_list = width_list[indices]
    rot_list = np.array(rot_list).astype(np.uint8)
    rot_list = rot_list[indices]
    max_w = min(maxW, np.max(width_list))

    processed_img_list = []
    for i in range(bbox_num):
        prep_img = preprocess(img_list[indices[i]], width_list[i], max_w)
        processed_img_list.append(prep_img)

    processed_img = np.array(processed_img_list).astype(np.float32)
    print(processed_img.shape, np.sum(processed_img))
    fs = cv2.FileStorage('cls_in_img.ext', cv2.FILE_STORAGE_WRITE)
    fs.write('img', processed_img)

    ie_result = compiled_model([processed_img])[output_layer]
    idx_batch = np.argmax(ie_result, axis=-1)
    prob_batch = np.max(ie_result, axis=-1)
    for i in range(bbox_num):
        if idx_batch[indices[i]] == 1 and prob_batch[indices[i]] >= 0.9:
            img_list[indices[i]] = cv2.rotate(img_list[indices[i]],
                                              cv2.ROTATE_180)
    return img_list, width_list


def rec(crop_imgs, width_list):
    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_rec_infer_matrix'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer = compiled_model.output(0)
    print(output_layer)

    num = len(crop_imgs)
    processed_img_list = []
    max_w = max(width_list)
    for i in range(num):
        print(crop_imgs[i].shape)
        prep_img = preprocess(crop_imgs[i], width_list[i], max_w)
        processed_img_list.append(prep_img)

    processed_img = np.array(processed_img_list).astype(np.float32)
    print(processed_img.shape)
    print(processed_img.shape, np.sum(processed_img))
    fs = cv2.FileStorage('rec_in_img.ext', cv2.FILE_STORAGE_WRITE)
    fs.write('img', processed_img)

    ie_result = compiled_model([processed_img])[output_layer]
    print('sum:', np.sum(ie_result), ie_result.shape)
    inds = np.argmax(ie_result, axis=-1)
    probs = np.max(ie_result, axis=-1)
    print(inds)
    print(probs)

    inds0 = []
    probs0 = []
    for i in range(ie_result.shape[0]):
        a = ie_result[i]
        print(a.shape)
        ind = []
        prob = []
        for j in range(a.shape[0]):
            _, p, _, k = cv2.minMaxLoc(a[j])
            ind.append(k[1])
            prob.append(p)
        inds0.append(ind)
        probs0.append(prob)
    print(inds0)
    print(probs0)

    chars = load_chars(
        '/home/liuqingjie/projects/TritonModels/ocr_lite/model_repository/rec_ch_graph/character_dict.txt'
    )
    for i in range(inds.shape[0]):
        text, prob = get_text(inds[i], probs[i], chars)
        print(text, prob)

    inds0 = np.array(inds0).astype(np.int32)
    probs0 = np.array(probs0)
    for i in range(inds0.shape[0]):
        text, prob = get_text(inds0[i], probs0[i], chars)
        print(text, prob)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def speed_test():
    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_rec_infer_fp16_matrix'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer0 = compiled_model.output(0)

    data_dir = '/home/liuqingjie/rec_test_sample'
    names = load_files(data_dir)
    cnt = 0
    elapes = 0
    for name in names:
        fs = cv2.FileStorage(os.path.join(data_dir, name),
                             cv2.FILE_STORAGE_READ)
        data = fs.getNode('imgs').mat()
        st = time.time()
        ie_result = compiled_model([data])[output_layer0]
        #preds_idx = ie_result.argmax(axis=2)
        #preds_prob = ie_result.max(axis=2)
        elapes += (time.time() - st)
        cnt += 1
    print(elapes, cnt, elapes / cnt, elapes / 20)


def test():
    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_rec_infer_half_matrix'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer0 = compiled_model.output(0)
    data_dir = '/home/liuqingjie/rec_test'
    #fs = cv2.FileStorage(os.path.join(data_dir, "rec_0_0.ext"), cv2.FILE_STORAGE_READ)
    #data = fs.getNode("imgs").mat()
    fs = cv2.FileStorage('rec_in_img.ext', cv2.FILE_STORAGE_READ)
    data = fs.getNode('img').mat()
    ie_result = compiled_model([data])[output_layer0]
    print(ie_result.shape, np.sum(ie_result))


def test_v2():
    model_dir = '/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_rec_infer_half_matrix'
    model_name = os.path.join(model_dir, 'model')
    ie = Core()
    model = ie.read_model(model=model_name + '.xml',
                          weights=model_name + '.bin')
    model.reshape([(1, 32), 3, 48, (48, 1200)])
    compiled_model = ie.compile_model(model=model, device_name='CPU')
    output_layer0 = compiled_model.output(0)
    data_dir = '/home/liuqingjie/rec_test'
    #fs = cv2.FileStorage(os.path.join(data_dir, "rec_0_0.ext"), cv2.FILE_STORAGE_READ)
    #data = fs.getNode("imgs").mat()
    fs = cv2.FileStorage('rec_in_img.ext', cv2.FILE_STORAGE_READ)
    data = fs.getNode('img').mat()
    ie_result = compiled_model([data])[output_layer0]
    print(ie_result.shape, np.sum(ie_result))


if __name__ == '__main__':
    #ori_img, dt_boxes = det()
    #crop_imgs, width_list = cls(ori_img, dt_boxes)
    #rec(crop_imgs, width_list)
    #speed_test()
    #test()
    test_v2()
