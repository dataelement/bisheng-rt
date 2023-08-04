import json
import math

import cv2
import numpy as np
from utils import (compute_angle, order_points, rotate_image_only,
                   rotate_polys_only)

from . import lanms
from .alg import ALGORITHM_REGISTRY, AlgorithmNoGraph, AlgorithmWithGraph


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


def mask_to_bboxes(mask, scores, boxes_cos, boxes_sin, image_shape=None):
    # Minimal shorter side length and area are used for post- filtering and
    # set to 10 and 300 respectively
    min_area = 0
    min_height = 0

    valid_scores = []
    valid_boxes_cos = []
    valid_boxes_sin = []
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
        valid_boxes_cos.append(boxes_cos[bbox_idx])
        valid_boxes_sin.append(boxes_sin[bbox_idx])

    return bboxes, valid_scores, valid_boxes_cos, valid_boxes_sin


def start_point_boxes(boxes, boxes_cos, boxes_sin):
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


def rotate_image_by_bbox(image, boxes):
    """rotate_image_by_bbox

    Args:
        image (np.array): shape is [h, w, c]
        boxes (np.array): shape is [n, 4, 2]
    Returns:
        img_rotate (np.array): shape is [h, w, c]
        old_center (tuple): (center_w, center_h)
        new_center (tuple): (center_w, center_h)
        theta (float): rotate angle
    """
    boxes = [order_points(box) for box in boxes]
    # diff with c++, c++ delete 10% head and tail boxes
    theta = 0
    for box in boxes:
        angle = compute_angle(box)
        theta = (angle / math.pi) * 180 + theta
    theta = theta / len(boxes)
    img_rotate, old_center, new_center = rotate_image_only(image, -theta)
    return img_rotate, old_center, new_center, theta


@ALGORITHM_REGISTRY.register_module()
class OcrMrcnnPreProcess(AlgorithmNoGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrMrcnnPreProcess, self).__init__(alg_params, alg_inputs,
                                                 alg_ouputs, **kwargs)
        if 'scale_list' in alg_params:
            self.scale_list = alg_params['scale_list']['string_value']
            self.scale_list = np.asarray(
                list(map(float, self.scale_list.split(' '))))
        else:
            self.scale_list = np.asarray(
                [200, 400, 600, 800, 1000, 1200, 1600])

    async def infer(self, context, inputs):
        """OcrMrcnn preprocess

        Args:
            context (dict): OcrMrcnn global information
            inputs (list[np.array]): [ori_img], inputs of algorithm
        Returns:
            context (dict): OcrMrcnn global information
            pre_outputs (list[np.array]): [resized_img], outputs of preprocess
        """
        img = inputs[0]
        orig_shape = img.shape[:2]
        h = orig_shape[0]
        w = orig_shape[1]

        if 'longer_edge_size' in context['params']:
            longer_edge_size = context['params']['longer_edge_size']
        else:
            max_side = max(h, w)
            distance = np.abs(self.scale_list - max_side)
            longer_edge_size = self.scale_list[np.argmin(distance)]

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

        padding = context['params'].get('padding', False)
        if padding:
            resized_h, resized_w, channel = resized_img.shape
            pad_img = np.zeros([longer_edge_size, longer_edge_size, channel],
                               dtype=resized_img.dtype)
            pad_img[:resized_h, :resized_w, :] = resized_img
            resized_img = pad_img

        context['params']['scale'] = float(scale)
        context['params']['orig_shape'] = orig_shape
        pre_params = np.array([json.dumps(context['params']).encode('utf-8')
                               ]).astype(np.object_)
        pre_outputs = [resized_img, pre_params]
        return context, pre_outputs


@ALGORITHM_REGISTRY.register_module()
class OcrMrcnnPostProcess(AlgorithmNoGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrMrcnnPostProcess, self).__init__(alg_params, alg_inputs,
                                                  alg_ouputs, **kwargs)

    async def infer(self, context, inputs):
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
        pre_scores, pre_masks, pre_boxes, pre_boxes_cos, pre_boxes_sin = inputs
        scale = context['params'].get('scale', 1.0)
        orig_shape = context['params'].get('orig_shape', [1600, 1600])

        # post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes(pre_boxes, orig_shape)
        full_masks = [
            paste_mask(box, mask, orig_shape)
            for box, mask in zip(pre_boxes, pre_masks)
        ]
        pre_masks = full_masks

        boxes, scores, boxes_cos, boxes_sin = mask_to_bboxes(
            np.asarray(pre_masks), pre_scores, pre_boxes_cos, pre_boxes_sin,
            orig_shape)
        if len(boxes):
            boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
            boxes_scores[:, :8] = boxes
            boxes_scores[:, 8] = scores
            boxes_scores[:, 9] = boxes_cos
            boxes_scores[:, 10] = boxes_sin
            boxes_scores = lanms.merge_quadrangle_standard(
                boxes_scores.astype('float32'), 0.2)
            boxes = boxes_scores[:, :8]
            scores = boxes_scores[:, 8]
            boxes_cos = boxes_scores[:, 9]
            boxes_sin = boxes_scores[:, 10]
            boxes = start_point_boxes(boxes, boxes_cos, boxes_sin)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
        else:
            boxes = np.empty([0, 4, 2], np.float32)
            scores = np.empty([0], np.float32)

        post_outputs = [np.asarray(boxes), np.asarray(scores)]
        return context, post_outputs


@ALGORITHM_REGISTRY.register_module()
class OcrMrcnnTrtInfer(AlgorithmWithGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrMrcnnTrtInfer, self).__init__(alg_params, alg_inputs,
                                               alg_ouputs, **kwargs)
        self.dep_model_inputs = ['image']
        self.dep_model_outputs = ['output_detections', 'output_masks']

    def preprocess(self, context, inputs):
        img = inputs[0]
        img = np.expand_dims(img, axis=0)
        pre_outputs = [img]
        return context, pre_outputs

    def postprocess(self, context, inputs):
        output_detections, output_masks = inputs
        boxes = output_detections[:, :4]
        scores = output_detections[:, 5]
        boxes_cos = output_detections[:, 6]
        boxes_sin = output_detections[:, 7]

        valid_num = 0
        for index in range(len(scores)):
            if scores[index] > 0.01:
                valid_num += 1
            else:
                break
        boxes = boxes[:valid_num, :]
        scores = scores[:valid_num]
        boxes_cos = boxes_cos[:valid_num]
        boxes_sin = boxes_sin[:valid_num]
        output_masks = output_masks[:valid_num, 0, :, :]

        post_outputs = [scores, output_masks, boxes, boxes_cos, boxes_sin]
        return context, post_outputs


@ALGORITHM_REGISTRY.register_module()
class OcrMrcnn(AlgorithmWithGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrMrcnn, self).__init__(alg_params, alg_inputs, alg_ouputs,
                                       **kwargs)
        self.dep_model_inputs = ['image']
        self.dep_model_outputs = [
            'output/scores', 'output/masks', 'output/boxes',
            'output/boxes_cos', 'output/boxes_sin'
        ]
        if 'scale_list' in alg_params:
            self.scale_list = alg_params['scale_list']['string_value']
            self.scale_list = np.asarray(
                list(map(float, self.scale_list.split(' '))))
        else:
            self.scale_list = np.asarray(
                [200, 400, 600, 800, 1000, 1200, 1600])

    def preprocess(self, context, inputs):
        """OcrMrcnn preprocess

        Args:
            context (dict): OcrMrcnn global information
            inputs (list[np.array]): [ori_img], inputs of algorithm
        Returns:
            context (dict): OcrMrcnn global information
            pre_outputs (list[np.array]): [resized_img], outputs of preprocess
        """
        img = inputs[0]
        orig_shape = img.shape[:2]
        h = orig_shape[0]
        w = orig_shape[1]

        if 'longer_edge_size' in context['params']:
            longer_edge_size = context['params'].get('longer_edge_size', 1600)
        else:
            max_side = max(h, w)
            distance = np.abs(self.scale_list - max_side)
            longer_edge_size = self.scale_list[np.argmin(distance)]

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

        padding = context['params'].get('padding', False)
        if padding:
            resized_h, resized_w, channel = resized_img.shape
            pad_img = np.zeros([longer_edge_size, longer_edge_size, channel],
                               dtype=resized_img.dtype)
            pad_img[:resized_h, :resized_w, :] = resized_img
            resized_img = pad_img

        context['scale'] = scale
        context['orig_shape'] = orig_shape
        pre_outputs = [resized_img]
        return context, pre_outputs

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
        pre_scores, pre_masks, pre_boxes, pre_boxes_cos, pre_boxes_sin = inputs
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

        boxes, scores, boxes_cos, boxes_sin = mask_to_bboxes(
            np.asarray(pre_masks), pre_scores, pre_boxes_cos, pre_boxes_sin,
            orig_shape)
        if len(boxes):
            boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
            boxes_scores[:, :8] = boxes
            boxes_scores[:, 8] = scores
            boxes_scores[:, 9] = boxes_cos
            boxes_scores[:, 10] = boxes_sin
            boxes_scores = lanms.merge_quadrangle_standard(
                boxes_scores.astype('float32'), 0.2)
            boxes = boxes_scores[:, :8]
            scores = boxes_scores[:, 8]
            boxes_cos = boxes_scores[:, 9]
            boxes_sin = boxes_scores[:, 10]
            boxes = start_point_boxes(boxes, boxes_cos, boxes_sin)
            boxes = boxes[:, :8].reshape((-1, 4, 2))

        post_outputs = [np.asarray(boxes), np.asarray(scores)]
        return context, post_outputs

    async def predict(self, context):
        """algorithm predict

        Args:
            context (dict): algorithm global information
                            (include alg input tensors)
        Returns:
            context (dict): algorithm global information
                            (include alg output tensors)
        """
        params = context.get('params', [b'{}'])
        params = json.loads(params[0].decode('utf-8'))  # python dict
        context['params'] = params

        input_list = []
        for input_name in self.alg_inputs:
            assert input_name in context, f'{input_name} not in context. ' + \
                                          'Please check request input tensor.'
            input_list.append(context[input_name])

        support_long_rotate_dense = context['params'].get(
            'support_long_rotate_dense', False)
        if support_long_rotate_dense:
            # first infer on origin image
            context, pre_outputs = self.preprocess(context, input_list)
            context, infer_outputs = await self.infer(context, pre_outputs)
            context, post_outputs = self.postprocess(context, infer_outputs)

            if len(post_outputs[0]):
                # middle: compute rotate angle and rotate image
                img_rotate, old_center, new_center, theta = \
                    rotate_image_by_bbox(input_list[0], post_outputs[0])

                # second infer on ratate image
                context, pre_outputs = self.preprocess(context, [img_rotate])
                context, infer_outputs = await self.infer(context, pre_outputs)
                context, post_outputs = self.postprocess(
                    context, infer_outputs)

                # convert boxes to origin image
                post_outputs[0] = rotate_polys_only(old_center, new_center,
                                                    post_outputs[0], theta)
        else:
            context, pre_outputs = self.preprocess(context, input_list)
            context, infer_outputs = await self.infer(context, pre_outputs)
            context, post_outputs = self.postprocess(context, infer_outputs)

        assert len(post_outputs) == len(
            self.alg_ouputs
        ), 'Num of post_outputs not equal to num of outputs in modelconfig.'
        for index, output_name in enumerate(self.alg_ouputs):
            context[output_name] = post_outputs[index]
        return context


@ALGORITHM_REGISTRY.register_module()
class OcrMrcnnTrt(OcrMrcnn):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrMrcnnTrt, self).__init__(alg_params, alg_inputs, alg_ouputs,
                                          **kwargs)
        self.dep_model_inputs = ['image']
        self.dep_model_outputs = ['output_detections', 'output_masks']
        if 'scale_list' in alg_params:
            self.scale_list = alg_params['scale_list']['string_value']
            self.scale_list = np.asarray(
                list(map(float, self.scale_list.split(' '))))
        else:
            self.scale_list = np.asarray(
                [200, 400, 600, 800, 1000, 1200, 1600])

    def preprocess(self, context, inputs):
        """OcrMrcnn preprocess

        Args:
            context (dict): OcrMrcnn global information
            inputs (list[np.array]): [ori_img], inputs of algorithm
        Returns:
            context (dict): OcrMrcnn global information
            pre_outputs (list[np.array]): [resized_img], outputs of preprocess
        """
        img = inputs[0]
        orig_shape = img.shape[:2]
        h = orig_shape[0]
        w = orig_shape[1]

        if 'longer_edge_size' in context['params']:
            longer_edge_size = context['params'].get('longer_edge_size', 1600)
        else:
            max_side = max(h, w)
            distance = np.abs(self.scale_list - max_side)
            longer_edge_size = self.scale_list[np.argmin(distance)]

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

        padding = True
        if padding:
            resized_h, resized_w, channel = resized_img.shape
            pad_img = np.zeros([longer_edge_size, longer_edge_size, channel],
                               dtype=resized_img.dtype)
            pad_img[:resized_h, :resized_w, :] = resized_img
            resized_img = pad_img

        context['scale'] = scale
        context['orig_shape'] = orig_shape
        # resized_img shape is 1, longer_edge_size, longer_edge_size, channel
        pre_outputs = [np.expand_dims(resized_img, axis=0)]
        return context, pre_outputs

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
        output_detections, output_masks = inputs
        boxes = output_detections[:, :4]
        scores = output_detections[:, 5]
        boxes_cos = output_detections[:, 6]
        boxes_sin = output_detections[:, 7]

        valid_num = 0
        for index in range(len(scores)):
            if scores[index] > 0.01:
                valid_num += 1
            else:
                break
        pre_boxes = boxes[:valid_num, :]
        pre_scores = scores[:valid_num]
        pre_boxes_cos = boxes_cos[:valid_num]
        pre_boxes_sin = boxes_sin[:valid_num]
        pre_masks = output_masks[:valid_num, 0, :, :]

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

        boxes, scores, boxes_cos, boxes_sin = mask_to_bboxes(
            np.asarray(pre_masks), pre_scores, pre_boxes_cos, pre_boxes_sin,
            orig_shape)
        if len(boxes):
            boxes_scores = np.zeros((len(boxes), 11), dtype=np.float32)
            boxes_scores[:, :8] = boxes
            boxes_scores[:, 8] = scores
            boxes_scores[:, 9] = boxes_cos
            boxes_scores[:, 10] = boxes_sin
            boxes_scores = lanms.merge_quadrangle_standard(
                boxes_scores.astype('float32'), 0.2)
            boxes = boxes_scores[:, :8]
            scores = boxes_scores[:, 8]
            boxes_cos = boxes_scores[:, 9]
            boxes_sin = boxes_scores[:, 10]
            boxes = start_point_boxes(boxes, boxes_cos, boxes_sin)
            boxes = boxes[:, :8].reshape((-1, 4, 2))

        post_outputs = [np.asarray(boxes), np.asarray(scores)]
        return context, post_outputs
