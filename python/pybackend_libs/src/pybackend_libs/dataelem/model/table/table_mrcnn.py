import base64
import math
from typing import Any, Dict, List

import cv2
import lanms
import numpy as np
from pybackend_libs.dataelem.framework.tf_graph import TFGraph


def paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))  # inclusive
    x1 = max(x0, x1)  # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally
    # computed for this shape. but it's hard to do better, because the network
    # does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask

    return ret


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


def find_contours(mask, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    # mode = cv2.RETR_CCOMP
    mode = cv2.RETR_EXTERNAL
    try:
        contours, _ = cv2.findContours(mask, mode=mode, method=method)
    except Exception:
        _, contours, _ = cv2.findContours(mask, mode=mode, method=method)
    return contours


def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes.
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an
            oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:
        [cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid
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


def mask_to_bboxes(mask,
                   scores,
                   boxes_cos,
                   boxes_sin,
                   image_shape=None,
                   geometry='quad',
                   labels=[]):
    # Minimal shorter side length and area are used for post- filtering and
    # set to 10 and 300 respectively
    min_area = 0
    min_height = 0

    valid_scores = []
    valid_boxes_cos = []
    valid_boxes_sin = []
    valid_labels = []

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

        if geometry == 'quad':
            # 拟合最小外接矩形
            rect, rect_area = min_area_rect(cnt)
            w, h = rect[2:-1]
            if min(w, h) <= min_height:
                continue
            if rect_area <= min_area:
                continue
            xys = rect_to_xys(rect, image_shape)
        elif geometry == 'curve':
            # # 拟合多边形
            # xys = fit_quadrangle(cnt)

            # 原始轮廓直接返回
            xys = cnt.reshape(-1)
        else:
            assert False, 'geometry must be set quad or curve.'

        bboxes.append(xys)
        valid_scores.append(scores[bbox_idx])
        valid_boxes_cos.append(boxes_cos[bbox_idx])
        valid_boxes_sin.append(boxes_sin[bbox_idx])
        if len(labels) > 0:
            valid_labels.append(labels[bbox_idx])

    return bboxes, valid_scores, valid_boxes_cos, valid_boxes_sin, valid_labels


def start_point_boxes(boxes, boxes_cos, boxes_sin):
    """
    确定boxes起始点，第一个点总是左上点
    """
    for box_index in range(len(boxes)):
        cos_value = boxes_cos[box_index]
        sin_value = boxes_sin[box_index]
        cos_value_norm = 2 * cos_value - 1
        sin_value_norm = 2 * sin_value - 1
        cos_value = cos_value_norm / math.sqrt(
            math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))
        sin_value = sin_value_norm / math.sqrt(
            math.pow(cos_value_norm, 2) + math.pow(sin_value_norm, 2))
        # print('cos_value:', cos_value, 'sin_value:', sin_value)

        cos_angle = math.acos(cos_value) * 180 / np.pi
        sin_angle = math.asin(sin_value) * 180 / np.pi
        if cos_angle <= 90 and sin_angle <= 0:
            angle = 360 + sin_angle
        elif cos_angle <= 90 and sin_angle > 0:
            angle = sin_angle
        elif cos_angle > 90 and sin_angle > 0:
            angle = cos_angle
        elif cos_angle > 90 and sin_angle <= 0:
            angle = 360 - cos_angle
        # print('angle:', angle)

        box = boxes[box_index]
        box = box[:8].reshape((4, 2))
        box_angle_vector = box[1] - box[0]
        box_cos_value = box_angle_vector[0] / np.linalg.norm(box_angle_vector)
        box_sin_value = box_angle_vector[1] / np.linalg.norm(box_angle_vector)
        box_cos_angle = math.acos(box_cos_value) * 180 / np.pi
        box_sin_angle = math.asin(box_sin_value) * 180 / np.pi
        if box_cos_angle <= 90 and box_sin_angle <= 0:
            box_angle = 360 + box_sin_angle
        elif box_cos_angle <= 90 and box_sin_angle > 0:
            box_angle = box_sin_angle
        elif box_cos_angle > 90 and box_sin_angle > 0:
            box_angle = box_cos_angle
        elif box_cos_angle > 90 and box_sin_angle <= 0:
            box_angle = 360 - box_cos_angle
        box_angle = np.array([
            box_angle, (box_angle + 90) % 360, (box_angle + 180) % 360,
            (box_angle + 270) % 360
        ])

        delta_angle = np.append(np.abs(box_angle - angle),
                                360 - np.abs(box_angle - angle))
        start_point_index = np.argmin(delta_angle) % 4
        box = box[[
            start_point_index, (start_point_index + 1) % 4,
            (start_point_index + 2) % 4, (start_point_index + 3) % 4
        ]]
        boxes[box_index] = box.reshape((-1))
    return boxes


class Mrcnn(object):
    def __init__(self, **kwargs):
        sig = {
            'inputs': ['image:0'],
            'outputs': [
                'output/scores:0', 'output/masks:0', 'output/boxes:0',
                'output/boxes_cos:0', 'output/boxes_sin:0', 'output/labels:0'
            ]
        }

        devices = kwargs.get('devices')
        used_device = devices.split(',')[0]

        self.graph = TFGraph(sig, used_device, **kwargs)
        if 'scale_list' in kwargs:
            scale_list = kwargs.get('scale_list')
            self.scale_list = np.asarray(scale_list, dtype=np.float32)
        else:
            self.scale_list = np.asarray(
                [200, 400, 600, 800, 1000, 1200, 1600])

    def predict(self, context: Dict[str, Any], inputs) -> List[np.ndarray]:
        # b64_image = context.pop('b64_image')
        # img = base64.b64decode(b64_image)
        # img = np.fromstring(img, np.uint8)
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        img = inputs[0]
        context, prep_outputs = self.preprocess(context, [img])
        graph_outputs = self.graph.run(prep_outputs)
        outputs = self.postprocess(context, graph_outputs)
        return outputs

    def postprocess(self, context: Dict[str, Any], inputs: List[Any]):
        return {}, []

    def preprocess(self, context: Dict[str, Any],
                   inputs: List[Any]) -> Dict[str, Any]:
        img = inputs[0]
        longer_edge_size = context.get('longer_edge_size', None)
        padding = context.get('padding', False)

        orig_shape = img.shape[:2]
        h = orig_shape[0]
        w = orig_shape[1]

        if longer_edge_size is None:
            max_side = max(h, w)
            distance = np.abs(self.scale_list - max_side)
            longer_edge_size = self.scale_list[np.argmin(distance)]

        # print(self.scale_list, longer_edge_size)

        scale = longer_edge_size * 1.0 / max(h, w)
        if h > w:
            newh, neww = longer_edge_size, scale * w
        else:
            newh, neww = scale * h, longer_edge_size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        resized_img = cv2.resize(img, dsize=(neww, newh))
        scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] *
                        resized_img.shape[1] / img.shape[1],
                        dtype=np.float32)

        if padding:
            resized_h, resized_w, channel = resized_img.shape
            pad_img = np.zeros([longer_edge_size, longer_edge_size, channel],
                               dtype=resized_img.dtype)
            pad_img[:resized_h, :resized_w, :] = resized_img
            resized_img = pad_img

        context.update(scale=scale, orig_shape=orig_shape)
        prep_outputs = [resized_img]
        return context, prep_outputs


class MrcnnTableDetect(Mrcnn):
    def __init__(self, **kwargs):
        super(MrcnnTableDetect, self).__init__(**kwargs)

    def predict(self, context: Dict[str, Any]) -> List[np.ndarray]:
        b64_image = context.pop('b64_image')
        img = base64.b64decode(b64_image)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        context, prep_outputs = self.preprocess(context, [img])
        graph_outputs = self.graph.run(prep_outputs)
        outputs = self.postprocess(context, graph_outputs)
        return outputs

    def postprocess(self, context, inputs):
        """OcrMrcnn postprocess

        Args:
            context (dict): global information
            inputs (list[np.array]): [scores, masks, boxes,
                                      boxes_cos, boxes_sin], outputs of infer
        Returns:
            context (dict): global information
            post_outputs (list[np.array]): [boxes, scores],
                                            outputs of algorithm
        """
        pre_scores, pre_masks, pre_boxes, pre_boxes_cos, pre_boxes_sin, \
            pre_labels = inputs
        scale = context.get('scale', 1.0)
        orig_shape = context.get('orig_shape', [1600, 1600])

        # post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes(pre_boxes, orig_shape)
        full_masks = [
            paste_mask(box, mask, orig_shape)
            for box, mask in zip(pre_boxes, pre_masks)
        ]
        pre_masks = full_masks

        boxes, scores, boxes_cos, boxes_sin, labels = mask_to_bboxes(
            np.asarray(pre_masks),
            pre_scores,
            pre_boxes_cos,
            pre_boxes_sin,
            image_shape=orig_shape,
            geometry='quad')

        if len(boxes):
            boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
            boxes_scores[:, :8] = boxes
            boxes_scores[:, 8] = scores
            boxes_scores[:, 9] = boxes_cos
            boxes_scores[:, 10] = boxes_sin
            boxes_scores = lanms.merge_quadrangle_standard(
                boxes_scores.astype('float32'), 0.2)
            boxes = boxes_scores[:, :8]
            boxes_cos = boxes_scores[:, 9]
            boxes_sin = boxes_scores[:, 10]
            boxes = start_point_boxes(boxes, boxes_cos, boxes_sin)

        # post_outputs = [np.asarray(boxes)]
        result = {'bboxes': np.asarray(boxes).tolist()}
        return result


class MrcnnTableCellDetect(Mrcnn):
    def __init__(self, **kwargs):
        super(MrcnnTableCellDetect, self).__init__(**kwargs)

    def postprocess(self, context, inputs):
        """OcrMrcnn postprocess

        Args:
            context (dict): global information
            inputs (list[np.array]): [scores, masks, boxes,
                                      boxes_cos, boxes_sin], outputs of infer
        Returns:
            context (dict): global information
            post_outputs (list[np.array]): [boxes, scores],
                                            outputs of algorithm
        """
        pre_scores, pre_masks, pre_boxes, pre_boxes_cos, pre_boxes_sin, \
            pre_labels = inputs
        scale = context.get('scale', 1.0)
        orig_shape = context.get('orig_shape', [1600, 1600])

        # post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes(pre_boxes, orig_shape)
        full_masks = [
            paste_mask(box, mask, orig_shape)
            for box, mask in zip(pre_boxes, pre_masks)
        ]
        pre_masks = full_masks

        boxes, scores, boxes_cos, boxes_sin, _ = mask_to_bboxes(
            np.asarray(pre_masks),
            pre_scores,
            pre_boxes_cos,
            pre_boxes_sin,
            image_shape=orig_shape,
            geometry='quad')

        if len(boxes):
            boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
            boxes_scores[:, :8] = boxes
            boxes_scores[:, 8] = scores
            boxes_scores[:, 9] = boxes_cos
            boxes_scores[:, 10] = boxes_sin
            boxes_scores = lanms.merge_quadrangle_standard(
                boxes_scores.astype('float32'), 0.2)
            boxes = boxes_scores[:, :8]
            boxes_cos = boxes_scores[:, 9]
            boxes_sin = boxes_scores[:, 10]
            boxes = start_point_boxes(boxes, boxes_cos, boxes_sin)

        post_outputs = [np.asarray(boxes)]
        return post_outputs


class MrcnnTableRowColDetect(Mrcnn):
    def __init__(self, **kwargs):
        super(MrcnnTableRowColDetect, self).__init__(**kwargs)

    def postprocess(self, context, inputs):
        """OcrMrcnn postprocess

        Args:
            context (dict): OcrMrcnn global information
            inputs (list[np.array]): [scores, masks, boxes,
                                      boxes_cos, boxes_sin], outputs of infer
        Returns:
            context (dict): OcrMrcnn global information
            post_outputs (list[np.array]): [boxes, scores],
                                            outputs of algorithm
        """
        pre_scores, pre_masks, pre_boxes, pre_boxes_cos, pre_boxes_sin, \
            pre_labels = inputs
        scale = context.get('scale', 1.0)
        orig_shape = context.get('orig_shape', [1600, 1600])

        # post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes(pre_boxes, orig_shape)
        full_masks = [
            paste_mask(box, mask, orig_shape)
            for box, mask in zip(pre_boxes, pre_masks)
        ]
        pre_masks = full_masks

        boxes, scores, boxes_cos, boxes_sin, labels = mask_to_bboxes(
            np.asarray(pre_masks),
            pre_scores,
            pre_boxes_cos,
            pre_boxes_sin,
            image_shape=orig_shape,
            geometry='quad',
            labels=pre_labels)

        post_outputs = [
            np.asarray(boxes),
            np.asarray(scores),
            np.asarray(labels)
        ]
        return post_outputs
