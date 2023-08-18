import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import requests
import tensorflow as tf
from shapely.geometry import Polygon

CLASSES = ['印章', '图片', '标题', '段落', '表格', '页眉', '页码', '页脚', '其他']
CLASS_MAP = {(k + 1): v for k, v in enumerate(CLASSES)}


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


def ocr_bboxes_layout_analysis(model_pred, ocr_bboxes):
    boxes, scores, labels = model_pred[0], model_pred[1], model_pred[2]
    layout_res = list()
    for box, score, label in zip(boxes, scores, labels):
        tmp_dict = dict()
        category_name = CLASS_MAP[label]
        tmp_dict['category_id'] = label
        tmp_dict['category_name'] = category_name
        tmp_dict['bbox'] = list()
        tmp_dict['bbox'].extend(box.tolist())
        tmp_dict['score'] = score
        layout_res.append(tmp_dict)

    layout_bbox_result = list()
    for ind, ocr_bbox in enumerate(ocr_bboxes):
        tmp_result = list()
        for layout_pred in layout_res:
            layout_bbox = Polygon(np.array(layout_pred['bbox']).reshape(-1, 2))
            intersect = Polygon(ocr_bbox).intersection(layout_bbox)
            if intersect:
                # intersection_area = intersect.area
                # The ratio of the intersection area to ocr bbox area
                intersection_area = intersect.area / Polygon(ocr_bbox).area
                layout_cls = layout_pred['category_name']
                score = layout_pred['score']
                tmp_result.append((ind, layout_cls, intersection_area, score))

        if not len(tmp_result):
            tmp_result.append((ind, '其他', 0, 0))

        _tmp_result = sorted(tmp_result,
                             key=lambda x: (x[2], x[3]),
                             reverse=True)
        # print(_tmp_result)

        layout_bbox_result.append(_tmp_result[0][1])

    return layout_bbox_result


class Mrcnn(object):
    """
    mrcnn infer
    """
    def __init__(self, model_path, with_angle=True):
        self.model_path = model_path
        self.with_angle = with_angle
        self.xs = []
        self.ys = []

    def load_pb(self, variable_scope, device):
        if self.with_angle:
            sig = {
                'inputs': ['image:0'],
                'outputs': [
                    'output/boxes:0',
                    'output/scores:0',
                    'output/boxes_cos:0',
                    'output/boxes_sin:0',
                    'output/masks:0',
                    'output/labels:0',
                ],
            }
        else:
            sig = {
                'inputs': ['image:0'],
                'outputs': [
                    'output/boxes:0',
                    'output/scores:0',
                    'output/masks:0',
                    'output/labels:0',
                ],
            }

        with open(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # refer https://github.com/tensorflow/tensorflow/issues/18861
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=device)
        tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=gpu_options)

        self.sess = tf.Session(config=tfconfig)
        with self.sess.graph.as_default():
            tf.import_graph_def(graph_def, name=variable_scope)

        self.ys = [
            tf.get_default_graph().get_tensor_by_name(
                os.path.join(variable_scope, n)) for n in sig['outputs']
        ]
        self.xs = [
            tf.get_default_graph().get_tensor_by_name(
                os.path.join(variable_scope, n)) for n in sig['inputs']
        ]

    def infer(self, img, longer_edge_size=1600):
        pass


class LayoutMrcnn(Mrcnn):
    """
    Layout
    """
    def __init__(self, **kwargs):
        model_path = kwargs.get('model_path', None)
        if model_path is None:
            model_path = kwargs.get('pretrain_path')

        super().__init__(model_path=model_path)
        devices = kwargs.get('devices')
        used_device = devices.split(',')[0]
        self.precision = kwargs.get('precision', 'float32')
        self.load_pb('table_structure_row_col', device=used_device)
        self.scale_list = np.array([600, 800, 1000, 1200, 1400, 1600])

    def infer(self, img, longer_edge_size=0):
        orig_shape = img.shape[:2]
        # prep
        h = orig_shape[0]
        w = orig_shape[1]
        if longer_edge_size == 0:
            side0 = max(h, w)
            distance = np.abs(self.scale_list - side0)
            longer_edge_size = self.scale_list[np.argmin(distance)]
        scale = longer_edge_size * 1.0 / max(h, w)
        if h > w:
            newh, neww = longer_edge_size, scale * w
        else:
            newh, neww = scale * h, longer_edge_size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        resized_img = cv2.resize(img, dsize=(neww, newh))
        scale = np.sqrt(
            resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] /
            img.shape[1],
            dtype=np.float32,
        )

        # if self.precision == 'fp16':
        #     resized_img = resized_img.astype(np.float16)

        # start = time.time()
        # graph infer
        (
            pre_boxes,
            pre_scores,
            pre_boxes_cos,
            pre_boxes_sin,
            pre_masks,
            pre_labels,
        ) = self.sess.run(self.ys, feed_dict={self.xs[0]: resized_img})

        # end = time.time()
        # print('[Layout Analysis] %d ms per frame' % ((end - start) * 1000))

        # post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes(pre_boxes, orig_shape)

        boxes = pre_boxes.astype(np.int32)
        new_boxes = np.zeros((boxes.shape[0], 8), dtype=np.int32)
        new_boxes[:, 0] = boxes[:, 0]  # x1
        new_boxes[:, 1] = boxes[:, 1]  # y1
        new_boxes[:, 2] = boxes[:, 2]  # x2
        new_boxes[:, 3] = boxes[:, 1]  # y1
        new_boxes[:, 4] = boxes[:, 2]  # x2
        new_boxes[:, 5] = boxes[:, 3]  # y2
        new_boxes[:, 6] = boxes[:, 0]  # x1
        new_boxes[:, 7] = boxes[:, 3]  # y2
        scores = pre_scores.astype(np.float32)
        labels = pre_labels.astype(np.int32)
        return boxes, scores, labels

    def predict(self, inp):
        img = inp.get('b64_image')
        longer_edge_size = inp.get('longer_edge_size', 0)
        img = base64.b64decode(img)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        boxes, scores, labels = self.infer(img, longer_edge_size)
        res = []
        for i, box in enumerate(boxes):
            tmp_dict = {}
            score = float(scores[i])
            label = int(labels[i])
            category_name = CLASS_MAP[label]
            tmp_dict['category_id'] = label
            tmp_dict['category_name'] = category_name
            tmp_dict['bbox'] = []
            tmp_dict['bbox'].extend(box.tolist())
            tmp_dict['score'] = score
            res.append(tmp_dict)
        return res
