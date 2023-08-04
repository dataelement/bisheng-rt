import collections
import json

import cv2
import numpy as np
from utils import crop, dist_euclid, order_points

from .alg import ALGORITHM_REGISTRY, AlgorithmNoGraph


def refine_box_orientation(boxes, process_horizon):
    def box_bias(box):
        # box 4 x 2
        box_original = order_points(box).tolist()
        for i, bb in enumerate(box_original):
            if bb == list(box[0]):
                bias = i
                break
        return bias

    def box_mainbias(box):
        box_order = order_points(box).tolist()
        box_order_mainbias = [
            box_order[main_bais], box_order[(main_bais + 1) % 4],
            box_order[(main_bais + 2) % 4], box_order[(main_bais + 3) % 4]
        ]
        return box_order_mainbias

    boxes = np.array(boxes).reshape(-1, 4, 2)
    biases = list(map(box_bias, boxes))
    main_bais = collections.Counter(biases).most_common()[0][0]
    boxes_refined = []
    for bias, box in zip(biases, boxes):
        box_order = box_mainbias(box)
        order_w = dist_euclid(box_order[0], box_order[1])
        order_h = dist_euclid(box_order[1], box_order[2])
        box_w = dist_euclid(box[0], box[1])
        box_h = dist_euclid(box[1], box[2])
        if order_w * 2 > order_h:
            if not process_horizon and box_w * 2 > box_h:
                boxes_refined.append(box)
            else:
                boxes_refined.append(box_order)
        else:
            boxes_refined.append(box)
    return boxes_refined


def segment_image_v2(img_data, H=32, interval=1, unit_len=800):
    h, w, _ = img_data.shape
    if int(w * H / h) <= unit_len:
        return [img_data]
    resized_w = int(w * H / h)
    img_data2 = cv2.resize(img_data, (resized_w, H))
    img_data2_gray = cv2.cvtColor(img_data2, cv2.COLOR_BGR2GRAY)
    img_mean = np.mean(img_data2_gray)
    img_std = np.std(img_data2_gray)
    if np.abs(img_std) > 1e-6:
        img_data2_gray = (img_data2_gray - img_mean) / img_std
    else:
        img_data2_gray -= img_mean
    seg_width = unit_len - 100
    img_data_list = list()
    end_flag = False
    start_point = 0
    part_num = resized_w // seg_width
    point_list = list()
    for i in range(part_num):
        start_point += seg_width
        end_point = min(start_point + 100, resized_w)
        if end_point - interval > start_point:
            point_var = dict()
            for point in range(start_point, end_point - interval, interval):
                seg = img_data2_gray[0:H,
                                     point - interval:point + interval + 1]
                var_value = np.std(seg)
                point_var[point] = var_value
            seg_point = sorted(point_var.items(), key=lambda x: x[1])[0][0]
            if resized_w - seg_point < 50:
                point_list.append(resized_w)
                end_flag = True
                break
            else:
                point_list.append(seg_point)

    if not end_flag:
        point_list.append(resized_w)
    last_point_ori = 0
    for point in point_list:
        point_ori = min(int(point * h / H), w)
        img_data_list.append(img_data[:, last_point_ori:point_ori, :])
        last_point_ori = point_ori

    return img_data_list


def long_image_segment(crop_images, H=32, W_max=800):
    # TODO: take H and W_max into config yaml
    new_crop_images = []
    groups = []
    for index, crop_image in enumerate(crop_images):
        split_crop_images = segment_image_v2(crop_image, H, unit_len=W_max)
        new_crop_images.extend(split_crop_images)
        groups.extend([index] * len(split_crop_images))
    return new_crop_images, groups


def merge_texts(reg_texts, groups, delimitor=''):
    merge_reg_texts = []
    curr_val = reg_texts[0]
    curr_group = groups[0]
    for index in range(1, len(reg_texts)):
        if groups[index] == curr_group:
            curr_val += delimitor
            curr_val += reg_texts[index]
        else:
            merge_reg_texts.append(curr_val)
            curr_group = groups[index]
            curr_val = reg_texts[index]

    merge_reg_texts.append(curr_val)
    return merge_reg_texts


def convert_to_image_tensor(images, H=32):
    # images: list of np.array
    _, _, channel = images[0].shape
    num_img = len(images)
    shapes = np.array(list(map(lambda x: x.shape[:2], images)))
    widths = np.round(H / shapes[:, 0] * shapes[:, 1]).reshape([num_img, 1])
    heights = np.ones([num_img, 1]) * H
    shapes = np.asarray(np.concatenate([heights, widths], axis=1), np.int32)
    w_max = int(np.max(widths))

    img_canvas = np.zeros([num_img, H, w_max, channel], dtype=np.int32)
    for i, img in enumerate(images):
        h, w = shapes[i]
        img = cv2.resize(img, (w, h))
        img_canvas[i, :, :w, :] = img
    return img_canvas, shapes


@ALGORITHM_REGISTRY.register_module()
class OcrIntermediate(AlgorithmNoGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrIntermediate, self).__init__(alg_params, alg_inputs,
                                              alg_ouputs, **kwargs)

    def intermediate(self,
                     inputs,
                     refine_boxes=False,
                     process_horizon=False,
                     split_long_sentence=False):
        """intermadiate between detection and recognition
        refine box, crop image, split long sentence

        Args:
            inputs (list[np.array]): [image, boxes]
        Returns:
            outputs (list[np.array]): [crop_images, groups]
        """
        image = inputs[0]
        boxes = inputs[1]

        if len(boxes) > 0:
            # huarong refine box can be applied here
            if refine_boxes:
                # boxes: shape is [n, 4, 2]
                boxes = refine_box_orientation(boxes, process_horizon)
            crop_images = crop(image, boxes)  # list[np.array]
        else:
            crop_images = []

        if split_long_sentence:
            new_crop_images, groups = long_image_segment(crop_images)
        else:
            new_crop_images, groups = crop_images, list(range(
                len(crop_images)))

        return new_crop_images, groups

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): OcrIntermediate global information
            inputs (list[np.array]): [image, boxes], inputs of app
        Returns:
            context (dict): OcrIntermediate global information
            outputs (list[np.array]): [crop_images, shapes, groups],
                                      outputs of app
        """
        refine_boxes = context['params'].get('refine_boxes', False)
        process_horizon = context['params'].get('process_horizon', False)
        split_long_sentence = context['params'].get('split_long_sentence',
                                                    False)

        crop_images, groups = self.intermediate(inputs, refine_boxes,
                                                process_horizon,
                                                split_long_sentence)

        if len(crop_images):
            # convert to tensor (num, 32, max_width, channel)
            crop_images, shapes = convert_to_image_tensor(crop_images)
            groups = np.asarray(groups, np.int32)
        else:
            crop_images = np.empty([0, 32, 800, 3], dtype=np.float32)
            shapes = np.empty([0, 2], dtype=np.int32)
            groups = np.empty([0], dtype=np.int32)

        infer_outputs = [crop_images, shapes, groups]
        return context, infer_outputs


@ALGORITHM_REGISTRY.register_module()
class OcrPost(AlgorithmNoGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrPost, self).__init__(alg_params, alg_inputs, alg_ouputs,
                                      **kwargs)

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): OcrPost global information
            inputs (list[np.array]): [image, boxes, box_scores,
                                      reg_texts, groups], inputs of OcrPost
        Returns:
            context (dict): OcrPost global information
            outputs (list[np.array]): [ocr_results], outputs of OcrPost
        """
        split_long_sentence = context['params'].get('split_long_sentence',
                                                    False)
        is_blank_delimitor = context['params'].get('split_long_sentence_blank',
                                                   False)
        delimitor = ' ' if is_blank_delimitor else ''

        ocr_results = {}
        image = inputs[0]
        ocr_results['bboxes'] = inputs[1].tolist()
        ocr_results['bbox_scores'] = inputs[2].tolist()

        reg_texts = inputs[3]
        groups = inputs[4]
        if len(reg_texts):
            reg_texts = list(map(lambda x: x.decode(), reg_texts))
            if split_long_sentence:
                reg_texts = merge_texts(reg_texts, groups, delimitor)
            ocr_results['texts'] = reg_texts
        else:
            ocr_results['texts'] = []

        output = dict()
        output['width'] = image.shape[1]
        output['height'] = image.shape[0]
        output['contents'] = ocr_results

        infer_outputs = [
            np.array([json.dumps(output).encode('utf-8')]).astype(np.object_)
        ]
        return context, infer_outputs
