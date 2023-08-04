import tensorflow as tf
import numpy as np
import math
import os
import time
from ocr_mask_rcnn_net import build_ocr_maskrcnn, build_ocr_maskrcnn_cascade
from config import finalize_configs, config as cfg
import cv2
import lanms


def write_lines(p, lines, append_break = False):
    with open(p, 'w') as f:
        for line in lines:
            if append_break:
                f.write(line + '\n')
            else:
                f.write(line)


def write_result_as_txt(image_name, bboxes, path):
    filename = os.path.join(path, '%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        # values = [float(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    write_lines(filename, lines)


def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    mode = cv2.RETR_EXTERNAL
    try:
        contours, _ = cv2.findContours(mask, mode = mode,
                                   method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = mode,
                                  method = method)
    return contours


def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    # points = np.int0(points)
    # for i_xy, (x, y) in enumerate(points):
    #     x = get_valid_x(x)
    #     y = get_valid_y(y)
    #     points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def mask_to_bboxes(mask, scores, image_shape=None):
    # Minimal shorter side length and area are used for post- filtering and set to 10 and 300 respectively
    min_area = 0
    min_height = 0

    valid_scores = []
    bboxes = []
    max_bbox_idx = len(mask)

    for bbox_idx in range(0, max_bbox_idx):
        bbox_mask = mask[bbox_idx, :, :]
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        # 只回归最大面积的mask的box
        max_area = 0
        max_index = 0
        for index, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_index = index
        cnt = cnts[max_index]
        rect, rect_area = min_area_rect(cnt)
        w, h = rect[2:-1]
        if min(w, h) <= min_height:
            continue

        if rect_area <= min_area:
            continue

        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        valid_scores.append(scores[bbox_idx])

    return bboxes, valid_scores


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask

    return ret


def do_predict(sess, input_nodes, output_nodes, input_folder, output_path):
    image_file_path = os.path.join(input_folder, 'image_file.txt')
    with open(image_file_path, 'r') as f:
        time_tf = 0
        start_cnt = False
        cnt = 0
        for line in f.readlines():
            name = line.strip()
            bin_name = input_folder + '/bin/' + name
            shape_name = input_folder + '/shape/' + name
            shape = np.fromfile(shape_name, dtype=np.float32)
            resized_h, resized_w, origin_h, origin_w = shape[:4].astype(np.int32)
            scale = shape[4]
            resized_img = np.fromfile(bin_name, dtype=np.float32).reshape([1, resized_h, resized_w, 3])
            # resized_img = np.fromfile(bin_name, dtype=np.float32).reshape([1, 3, resized_h, resized_w])

            orig_shape = [origin_h, origin_w]
            # run model
            t = time.time()
            detections, masks = sess.run(output_nodes, feed_dict={input_nodes[0]:resized_img})
            time_tf += (time.time() - t) * 1000
            cnt += 1
            if cnt == 100 and not start_cnt:
                start_cnt = True
                time_tf = 0
                cnt = 0
            if start_cnt and cnt > 0 and cnt % 50 == 0:
                print('tf:', time_tf, ' cnt:', cnt, ' tf/per_im:', time_tf/cnt)

            boxes = detections[:, :4]
            scores = detections[:, 5]
            masks = masks.reshape([-1, 28, 28])
            # 过滤加padding的box
            valid_boxes = []
            valid_scores = []
            valid_masks = []
            for index in range(len(scores)):
                if scores[index] > 0 :
                    valid_boxes.append(boxes[index])
                    valid_scores.append(scores[index])
                    valid_masks.append(masks[index])
            boxes = np.array(valid_boxes)
            scores = np.array(valid_scores)
            masks = np.array(valid_masks)
            if len(boxes) == 0:
                continue

            # postprocess
            boxes = boxes / scale
            boxes = clip_boxes(boxes, orig_shape)
            full_masks = np.array([paste_mask(box, mask, orig_shape) for box, mask in zip(boxes, masks)])
            masks = full_masks

            boxes, scores = mask_to_bboxes(masks, scores, orig_shape)
            if len(boxes):
                boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
                boxes_scores[:, :8] = boxes
                boxes_scores[:, 8] = scores
                boxes_scores = lanms.merge_quadrangle_standard(boxes_scores.astype('float32'), 0.2)
                boxes = boxes_scores[:, :8]

            write_result_as_txt(name, boxes, output_path)

        print('tf:', time_tf, ' cnt:', cnt, ' tf/per_im:', time_tf/cnt)


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


def tf_infer_maskrcnn(src_dir, dst_dir, model_path, fix_shape):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    finalize_configs()
    # input tensor
    if fix_shape:
        image = tf.placeholder(tf.float32, shape=([1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 3]), name='image')
    else:
        image = tf.placeholder(tf.float32, shape=([1, None, None, 3]), name='image')

    # bulid model
    detections, final_masks = build_ocr_maskrcnn_cascade(image)

    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    # load weight
    load_weights(sess, all_variables, model_path)

    # do infer
    do_predict(sess, [image], [detections, final_masks], src_dir, dst_dir)
    sess.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_dir = '../../../build/test_data/ocr_maskrcnn_data/all_kinds_train_images_angle_val_fix_shape_hwc_1600'
    model_path = '../../../build/test_data/models/mask_rcnn_v5_angle/ocr_maskrcnn_weights.npz'
    res_dir = '../../../build/test_data/ocr_maskrcnn_data/all_kinds_train_images_angle_val_fix_shape_hwc_1600/txts'
    fix_shape = True
    tf_infer_maskrcnn(data_dir, res_dir, model_path, fix_shape)



