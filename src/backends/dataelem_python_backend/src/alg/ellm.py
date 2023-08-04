# flake8: noqa
import asyncio
import json
import math
import os
from functools import partial

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from collections import defaultdict

from .alg import ALGORITHM_REGISTRY, AlgorithmWithGraph, pb_tensor_to_numpy
from .ellm_utils.taskflow import Taskflow


def rotate_image_only(im, angle):
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """
    def rotate(src, angle, scale=1.0):  #1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5,
                                             0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rotated_image = cv2.warpAffine(
            src,
            rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    return image, old_center, new_center


@ALGORITHM_REGISTRY.register_module()
class ELLM(AlgorithmWithGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(ELLM, self).__init__(alg_params, alg_inputs, alg_ouputs,
                                   **kwargs)
        self.name = 'ELLM'
        self.task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'ellm_utils/data')
        is_trt = alg_params['is_trt']['string_value'] if 'is_trt' in alg_params else False
        self.is_trt = True if is_trt == "1" else False
        self.ie_task = Taskflow('information_extraction',
                                model='uie-x-base',
                                task_path=self.task_path,
                                ocr_lang='ch',
                                schema_lang='ch',
                                graph_name=self.dep_model_name,
                                is_trt=self.is_trt)
        self.char_bboxes_list = []

    def split_for_ellm(self, bboxes, texts, schema, max_seq_len=512):
        max_prompt_len = max([len(key) for key in schema])
        summary_token_num = 4  # [CLS] + [SEP] + [SEP] + [SEP]
        max_content_len = max_seq_len - max_prompt_len - summary_token_num

        bboxes_start = []
        start_index = 0
        bboxes_start.append(0)
        for text in texts:
            start_index += len(text)
            bboxes_start.append(start_index)
        
        split_texts = []
        split_bboxes = []
        split_range = []
        each_fragment_text = []
        each_fragment_bbox = []
        max_len = max_content_len
        for index, text in enumerate(texts):
            if bboxes_start[index] + len(text) > max_len:
                # 当前box文本累加长度超出max_len，切分到上一个box
                split_texts.append(each_fragment_text)
                split_bboxes.append(each_fragment_bbox)
                fragment_text_len = len(''.join(each_fragment_text))
                split_range.append((max_len - max_content_len,
                                    max_len - max_content_len + fragment_text_len))
                max_len += fragment_text_len
                each_fragment_text = []
                each_fragment_bbox = []
            each_fragment_text.append(text)
            each_fragment_bbox.append(bboxes[index])
        split_texts.append(each_fragment_text)
        split_bboxes.append(each_fragment_bbox)
        fragment_text_len = len(''.join(each_fragment_text))
        split_range.append((max_len - max_content_len,
                            max_len - max_content_len + fragment_text_len))
        return split_bboxes, split_texts, split_range

    def preprocess(self, context, inputs):
        """OcrTransformer preprocess

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [images, images_shape],
                                      inputs of algorithm
        Returns:
            context (dict): OcrTransformer global information
            pre_outputs (list[np.array]): [batch_imgs, batch_imgs_shape],
                                          outputs of preprocess
        """
        image = inputs[0]
        schema = inputs[2][0].decode().split('|')
        self.ie_task.set_schema(schema)
        ocr_results = json.loads(inputs[1][0])
        bboxes = ocr_results['bboxes']
        texts = ocr_results['texts']
        rotate_angle = ocr_results['rotate_angle']
        image, _, _ = rotate_image_only(image, rotate_angle)
        
        split_bboxes, split_texts, _, = self.split_for_ellm(bboxes, texts, schema)
        layout_list = []
        for bboxes, texts in zip(split_bboxes, split_texts):
            layout = []
            char_bboxes = []
            for bbox, text in zip(bboxes, texts):
                bbox = np.array(bbox)
                x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(
                    bbox[:, 0]), max(bbox[:, 1])
                layout.append(([x1, y1, x2, y2], text))
                char_bboxes.extend([bbox.tolist()] * len(text))
            layout_list.append(layout)
            self.char_bboxes_list.append(char_bboxes)

        pre_outputs = self.ie_task.preprocess([{'doc': image, 'layout': layout} for layout in layout_list])
        return context, pre_outputs

    async def infer(self, context, inputs):
        """algorithm infer

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [batch_imgs, batch_imgs_shape],
                                      inputs of infer (outputs of preprocess)
        Returns:
            context (dict): OcrTransformer global information
            infer_outputs (list[np.array]): [img_string], outputs of infer
                                             (inputs of postprocess)
        """
        infer_outputs = self.ie_task.run_model(inputs)
        return context, infer_outputs

    def postprocess(self, context, inputs):
        """OcrTransformer postprocess

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [img_string], outputs of infer
        Returns:
            context (dict): OcrTransformer global information
            post_outputs (list[np.array]): [img_string], outputs of algorithm
        """
        results = self.ie_task.postprocess(inputs)
        convert_res = defaultdict(lambda: defaultdict(list))
        cnt = 0
        for res in results:
            for key, val in res.items():
                val = sorted(val, key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))
                conbine_bbox = []
                combine_text = [elem['text'] for elem in val]
                for elem in val:
                    # conbine_bbox.append(self.char_bboxes_list[cnt][elem['start']])
                    for box in elem['bbox']:
                        x1, y1, x2, y2 = box
                        conbine_bbox.append([[x1, y1], [x2, y1], [x2, y2],
                                            [x1, y2]])

                convert_res[key]['box'].extend(conbine_bbox)
                convert_res[key]['text'].extend(combine_text)
            cnt += 1
        res_byte = json.dumps(convert_res).encode('utf-8')
        post_outputs = [np.array([res_byte]).astype(np.object_)]
        return context, post_outputs
