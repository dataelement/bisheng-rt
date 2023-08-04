# flake8: noqa
import base64
import copy
import json
import math
import os
import time

import cv2
import numpy as np
from utils import (curve2rect_ellipse, fit_ellipse_of_outer_points,
                   perspective_transform, rotate_image, seal_postprocess)

from .app import APP_REGISTRY, BaseApp

color_range = [(255, 0, 0), (0, 255, 0), (0, 0, 0), (255, 255, 0),
               (255, 0, 255), (0, 255, 255)]
class_map = {1: 'round', 2: 'ellipse', 3: 'square'}


def start_point_boxes(boxes, boxes_cos, boxes_sin):
    boxes_with_angle = np.zeros((len(boxes), 9))
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
        boxes_with_angle[box_index] = np.append(box.reshape((-1)), angle)
    return boxes_with_angle


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


@APP_REGISTRY.register_module()
class SealApp(BaseApp):
    def __init__(self, app_params, app_inputs, app_ouputs, **kwargs):
        super(SealApp, self).__init__(app_params, app_inputs, app_ouputs,
                                      **kwargs)
        assert 'seal_det_model_name' in app_params, \
            'Please specify seal_det_model_name in config.pbtxt'
        self.seal_det_model_name = \
            app_params['seal_det_model_name']['string_value']
        self.seal_det_model_version = int(
            app_params['seal_det_model_version']
            ['string_value']) if 'seal_det_model_version' in app_params else -1
        self.seal_det_model_inputs = ['image', 'params']
        self.seal_det_model_outputs = [
            'boxes', 'labels', 'boxes_cos', 'boxes_sin'
        ]

        assert 'seal_curve_text_det_model_name' in app_params, \
            'Please specify seal_curve_text_det_model_name in config.pbtxt'
        self.seal_curve_text_det_model_name = app_params[
            'seal_curve_text_det_model_name']['string_value']
        self.seal_curve_text_det_model_version = int(
            app_params['seal_curve_text_det_model_version']['string_value']
        ) if 'seal_curve_text_det_model_version' in app_params else -1
        self.seal_curve_text_det_model_inputs = ['image', 'params']
        self.seal_curve_text_det_model_outputs = ['boxes']

        assert 'seal_quad_text_det_model_name' in app_params, \
            'Please specify seal_quad_text_det_model_name in config.pbtxt'
        self.seal_quad_text_det_model_name = app_params[
            'seal_quad_text_det_model_name']['string_value']
        self.seal_quad_text_det_model_version = int(
            app_params['seal_quad_text_det_model_version']['string_value']
        ) if 'seal_quad_text_det_model_version' in app_params else -1
        self.seal_quad_text_det_model_inputs = ['image', 'params']
        self.seal_quad_text_det_model_outputs = ['boxes']

        assert 'seal_text_reg_model_name' in app_params, \
            'Please specify seal_text_reg_model_name in config.pbtxt'
        self.seal_text_reg_model_name = app_params['seal_text_reg_model_name'][
            'string_value']
        self.seal_text_reg_model_version = int(
            app_params['seal_text_reg_model_version']['string_value']
        ) if 'seal_text_reg_model_version' in app_params else -1
        self.seal_text_reg_model_inputs = ['image', 'image_shape', 'params']
        self.seal_text_reg_model_outputs = ['while/Exit_1']

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): SealApp global information
            inputs (list[np.array]): [image_b64], inputs of app
        Returns:
            context (dict): SealApp global information
            outputs (list[np.array]): [seal_results], outputs of app
        """
        seal_det_longer_edge_size = context['params'].get(
            'seal_det_longer_edge_size', 0)
        seal_curve_text_det_longer_edge_size = context['params'].get(
            'seal_curve_text_det_longer_edge_size', 0)
        seal_quad_text_det_longer_edge_size = context['params'].get(
            'seal_quad_text_det_longer_edge_size', 0)
        debug = context['params'].get('debug', False)

        seal_results = []
        timer_result = {}
        # phase0: decode image
        start_time = time.time()
        # image_b64 = inputs[0][0]
        image = inputs[0]
        try:
            # image = cv2.imdecode(
            #     np.fromstring(base64.b64decode(image_b64), np.uint8),
            #     cv2.IMREAD_COLOR)
            timer_result['decode_time'] = time.time() - start_time

            # phase1: get seal cnts and labels of whole image
            start_time = time.time()
            seal_det_params = {}
            if seal_det_longer_edge_size != 0:
                seal_det_params['longer_edge_size'] = seal_det_longer_edge_size
            det_params = np.array([
                json.dumps(seal_det_params).encode('utf-8')
            ]).astype(np.object_)
            det_inputs = [image.astype(np.float32), det_params]
            det_outputs = self.alg_infer(det_inputs, self.seal_det_model_name,
                                         self.seal_det_model_version,
                                         self.seal_det_model_inputs,
                                         self.seal_det_model_outputs)
            seal_cnts = det_outputs[0].tolist()
            seal_labels = det_outputs[1].tolist()
            seal_cnts_cos = det_outputs[2].tolist()
            seal_cnts_sin = det_outputs[3].tolist()
            for index, cnt in enumerate(seal_cnts):
                # delete padding
                if -100 in cnt:
                    find_index = cnt.index(-100)
                    seal_cnts[index] = cnt[:find_index]

            if debug:
                save_dir = '/home/public/seal_vis'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if 'image_name' in context['params']:
                    image_name = context['params']['image_name']
                    image_base, image_end = os.path.splitext(
                        image_name.split('/')[-1])
                else:
                    image_base = time.strftime('%Y-%m-%d-%H_%M_%S',
                                               time.localtime(time.time()))
                    image_end = '.png'
                vis_img = copy.deepcopy(image)
                for index, box in enumerate(seal_cnts):
                    box = np.array(box)
                    cv2.polylines(vis_img,
                                  [box.astype(np.int32).reshape((-1, 1, 2))],
                                  True,
                                  color=color_range[seal_labels[index] - 1],
                                  thickness=2)
                    cv2.circle(vis_img,
                               (int(float(box[0])), int(float(box[1]))), 4,
                               (0, 0, 255), 2)
                    cv2.putText(vis_img,
                                str(seal_labels[index]),
                                (int(float(box[0])), int(float(box[1]))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.5,
                                thickness=1,
                                color=(0, 0, 255))
                cv2.imwrite(os.path.join(save_dir, image_base + image_end),
                            vis_img)
            timer_result['seal_det_time'] = time.time() - start_time

            # process each seal
            for seal_index, seal_cnt in enumerate(seal_cnts):
                seal_label = seal_labels[seal_index]
                seal_cnt_cos = seal_cnts_cos[seal_index]
                seal_cnt_sin = seal_cnts_sin[seal_index]
                seal_cnt = np.asarray(seal_cnt).reshape(-1, 2)
                xmin = min(seal_cnt[:, 0])
                ymin = min(seal_cnt[:, 1])
                xmax = max(seal_cnt[:, 0])
                ymax = max(seal_cnt[:, 1])

                crop_text_images = []
                crop_text_boxes = []
                # phase2: crop seal and predict txts in seal
                if class_map[seal_label] == 'round' or class_map[
                        seal_label] == 'ellipse':
                    # process round and ellipse
                    bbox = np.asarray([[xmin, ymin], [xmax, ymin],
                                       [xmax, ymax], [xmin, ymax]])
                    crop_seal_img = perspective_transform(image, bbox)

                    seal_curve_text_det_params = {}
                    if seal_curve_text_det_longer_edge_size != 0:
                        seal_curve_text_det_params['longer_edge_size'] = \
                            seal_curve_text_det_longer_edge_size
                    det_params = np.array([
                        json.dumps(seal_curve_text_det_params).encode('utf-8')
                    ]).astype(np.object_)
                    det_inputs = [crop_seal_img.astype(np.float32), det_params]
                    det_outputs = self.alg_infer(
                        det_inputs, self.seal_curve_text_det_model_name,
                        self.seal_curve_text_det_model_version,
                        self.seal_curve_text_det_model_inputs,
                        self.seal_curve_text_det_model_outputs)
                    text_boxes = det_outputs[0].tolist()
                    for index, text_box in enumerate(text_boxes):
                        # delete padding
                        if -100 in text_box:
                            find_index = text_box.index(-100)
                            text_boxes[index] = text_box[:find_index]

                    if debug:
                        vis_seal_img = copy.deepcopy(crop_seal_img)
                        for index, box in enumerate(text_boxes):
                            box = np.array(box)
                            cv2.polylines(vis_seal_img, [
                                box[:-1].astype(np.int32).reshape((-1, 1, 2))
                            ],
                                          True,
                                          color=color_range[0],
                                          thickness=2)
                        crop_image_name = image_base + '_round_seal_' + str(
                            seal_index) + image_end
                        cv2.imwrite(os.path.join(save_dir, crop_image_name),
                                    vis_seal_img)

                    # crop text image from seal according to text box
                    for text_index, text_box in enumerate(text_boxes):
                        text_box = np.asarray(text_box)
                        text_angle = text_box[-1]
                        text_box = text_box[:-1]
                        is_rotate = ((len(text_box) // 2) == 4)
                        if is_rotate:
                            # rotate box
                            text_image = perspective_transform(
                                crop_seal_img, text_box.reshape(-1, 2))
                            crop_text_images.append(text_image)
                            crop_text_boxes.append(text_box)
                        else:
                            # curve box
                            ellipse = fit_ellipse_of_outer_points(
                                crop_seal_img, text_box)
                            image_rect, _, success = curve2rect_ellipse(
                                crop_seal_img, text_box, ellipse, text_angle)
                            if success:
                                crop_text_images.append(image_rect)
                                crop_text_boxes.append(text_box)

                    if debug:
                        for index, text_image in enumerate(crop_text_images):
                            crop_image_name = image_base + '_round_seal_' + \
                                              str(seal_index) + '_text_' + \
                                              str(index) + image_end
                            cv2.imwrite(
                                os.path.join(save_dir, crop_image_name),
                                text_image)

                elif class_map[seal_label] == 'square':
                    # process square
                    rect = cv2.minAreaRect(seal_cnt.astype(np.int32))
                    rect_box = cv2.boxPoints(rect)
                    rect_box = start_point_boxes([rect_box], [seal_cnt_cos],
                                                 [seal_cnt_sin])[0][:-1]
                    rect_box = rect_box.reshape(-1, 2)

                    crop_seal_img = perspective_transform(image, rect_box)
                    seal_quad_text_det_params = {}
                    if seal_quad_text_det_longer_edge_size != 0:
                        seal_quad_text_det_params['longer_edge_size'] = \
                            seal_quad_text_det_longer_edge_size
                    det_params = np.array([
                        json.dumps(seal_quad_text_det_params).encode('utf-8')
                    ]).astype(np.object_)
                    det_inputs = [crop_seal_img.astype(np.float32), det_params]
                    det_outputs = self.alg_infer(
                        det_inputs, self.seal_quad_text_det_model_name,
                        self.seal_quad_text_det_model_version,
                        self.seal_quad_text_det_model_inputs,
                        self.seal_quad_text_det_model_outputs)
                    text_boxes = det_outputs[0].tolist()
                    text_boxes = np.asarray(text_boxes)

                    # rotate image and boxes to 0 degree
                    crop_seal_img_origin = copy.deepcopy(crop_seal_img)
                    text_boxes_origin = copy.deepcopy(text_boxes)
                    crop_seal_img, text_boxes = self.rotate_image_0_degree(
                        crop_seal_img, text_boxes)

                    if debug:
                        vis_seal_img = copy.deepcopy(crop_seal_img)
                        for index, box in enumerate(text_boxes):
                            cv2.polylines(vis_seal_img, [
                                box[:-1].astype(np.int32).reshape((-1, 1, 2))
                            ],
                                          True,
                                          color=color_range[0],
                                          thickness=2)
                            cv2.circle(
                                vis_seal_img,
                                (int(float(box[0])), int(float(box[1]))), 4,
                                (0, 0, 255), 2)
                        for index, box in enumerate(text_boxes_origin):
                            cv2.polylines(crop_seal_img_origin, [
                                box[:-1].astype(np.int32).reshape((-1, 1, 2))
                            ],
                                          True,
                                          color=color_range[0],
                                          thickness=2)
                            cv2.circle(
                                crop_seal_img_origin,
                                (int(float(box[0])), int(float(box[1]))), 4,
                                (0, 0, 255), 2)
                        crop_image_name = image_base + '_square_seal_' + str(
                            seal_index) + image_end
                        cv2.imwrite(os.path.join(save_dir, crop_image_name),
                                    vis_seal_img)
                        crop_image_name = image_base + \
                            '_square_seal_origin_' + str(
                                seal_index) + image_end
                        cv2.imwrite(os.path.join(save_dir, crop_image_name),
                                    crop_seal_img_origin)

                    # crop text image from seal according to text box
                    for text_index, text_box in enumerate(text_boxes):
                        text_box = text_box[:-1]
                        text_image = perspective_transform(
                            crop_seal_img, text_box.reshape(-1, 2))
                        crop_text_images.append(text_image)
                        crop_text_boxes.append(text_box)

                # phase3: text recog
                if len(crop_text_images):
                    crop_images, shapes = convert_to_image_tensor(
                        crop_text_images)
                    reg_params = np.array([json.dumps({}).encode('utf-8')
                                           ]).astype(np.object_)
                    reg_inputs = [
                        crop_images.astype(np.float32), shapes, reg_params
                    ]
                    reg_outputs = self.alg_infer(
                        reg_inputs, self.seal_text_reg_model_name,
                        self.seal_text_reg_model_version,
                        self.seal_text_reg_model_inputs,
                        self.seal_text_reg_model_outputs)
                    crop_texts = list(map(lambda x: x.decode(),
                                          reg_outputs[0]))

                    # phase4: square postprocess
                    if class_map[seal_label] == 'square':
                        predicts = []
                        for crop_text_box, crop_text in zip(
                                crop_text_boxes, crop_texts):
                            content = dict()
                            content['value'] = crop_text
                            content['points'] = crop_text_box.reshape(
                                -1, 2).tolist()
                            predicts.append(content)
                        crop_texts = seal_postprocess(predicts)

                    seal_res = dict()
                    seal_res['type'] = class_map[seal_label]
                    seal_res['bbox'] = [[xmin, ymin], [xmax, ymin],
                                        [xmax, ymax], [xmin, ymax]]
                    seal_res['texts'] = crop_texts
                    seal_results.append(seal_res)

            output = dict()
            output['code'] = 200
            output['message'] = 'success'
            output['width'] = image.shape[1]
            output['height'] = image.shape[0]
            output['contents'] = seal_results
        except Exception:
            import traceback
            output = dict()
            output['code'] = 300
            output['message'] = traceback.format_exc()

        infer_outputs = [
            np.array([json.dumps(output).encode('utf-8')]).astype(np.object_)
        ]
        return context, infer_outputs

    def rotate_image_0_degree(self, crop_seal_img, text_boxes):
        """
        rotate image to 0 degree
        """
        box_direction = self.compute_boxes_direction(text_boxes)
        mean_angle = self.compute_angle(box_direction)

        angles = np.array([0, 90, 180, 270])
        delta_angle = np.append(np.abs(angles - mean_angle),
                                360 - np.abs(angles - mean_angle))
        image_angle = angles[np.argmin(delta_angle) % 4]

        # unify all text boxes start point to image_angle
        text_polys = text_boxes[:, :8].reshape(-1, 4, 2)
        for index, box in enumerate(text_polys):
            box_angle_vector = box[1] - box[0]
            box_cos_value = box_angle_vector[0] / np.linalg.norm(
                box_angle_vector)
            box_sin_value = box_angle_vector[1] / np.linalg.norm(
                box_angle_vector)
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

            delta_angle = np.append(np.abs(box_angle - image_angle),
                                    360 - np.abs(box_angle - image_angle))
            start_point_index = np.argmin(delta_angle) % 4
            box = box[[
                start_point_index, (start_point_index + 1) % 4,
                (start_point_index + 2) % 4, (start_point_index + 3) % 4
            ]]
            text_polys[index] = box

        # rotate to 0 degree
        crop_seal_img, text_polys = rotate_image(crop_seal_img, text_polys,
                                                 image_angle)
        text_boxes[:, :8] = np.asarray(text_polys).reshape(-1, 8)
        return crop_seal_img, text_boxes

    def compute_boxes_direction(self, text_boxes):
        """
        compute boxes direction by standard_vec
        """
        direction_vectors = []
        for box_index, text_box in enumerate(text_boxes):
            box = text_box[:-1].reshape(-1, 2)
            p1 = (box[0] + box[3]) / 2
            p2 = (box[1] + box[2]) / 2
            d = self.standard_vec(p2 - p1)
            direction_vectors.append(d)

        vec_x = np.mean(np.array(direction_vectors)[:, 0])
        vec_y = np.mean(np.array(direction_vectors)[:, 1])
        box_direction = self.standard_vec([vec_x, vec_y])
        if np.isnan(box_direction[0]) or np.isnan(box_direction[1]):
            box_direction = np.array([1.0, 0.0])

        return box_direction

    @staticmethod
    def standard_vec(line):
        if not isinstance(line, np.ndarray):
            line = np.array(line)
        line_len = np.linalg.norm(line)
        return line / line_len

    @staticmethod
    def compute_angle(angle_vector):
        cos_angle = angle_vector[0] / np.linalg.norm(angle_vector)
        sin_angle = angle_vector[1] / np.linalg.norm(angle_vector)

        cos_angle = math.acos(cos_angle) * 180 / np.pi  # range(0, 180)
        sin_angle = math.asin(sin_angle) * 180 / np.pi  # range(-90, 90)
        # four
        if cos_angle <= 90 and sin_angle <= 0:
            angle = 360 + sin_angle
        # first
        elif cos_angle <= 90 and sin_angle > 0:
            angle = sin_angle
        # second
        elif cos_angle > 90 and sin_angle > 0:
            angle = cos_angle
        # third
        elif cos_angle > 90 and sin_angle <= 0:
            angle = 360 - cos_angle

        return angle
