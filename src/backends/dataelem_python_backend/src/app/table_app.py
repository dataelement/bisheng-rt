#!/usr/bin/env python
# File: train.py

import base64
import copy
import io
import json
import math
from collections import defaultdict

import cv2
import numpy as np
from utils import (intersection, perspective_transform, rotate_image_only,
                   rotate_polys_only)
from utils.html_to_excel import document_to_workbook
from utils.table_cell_post import PostCell, area_to_html, format_html
from utils.table_rowcol_post import objects_to_cells

from .app import APP_REGISTRY, BaseApp

# from utils.visualization import draw_box_on_img, ocr_visual


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def serilize_json(data):
    data = NpEncoder().encode(data)
    data = json.JSONDecoder().decode(data)
    return data


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


def group_ocr_bboxes(ocr_bboxes, ocr_texts, table_bboxes):
    group_bboxes = defaultdict(list)
    group_texts = defaultdict(list)
    group_orders = defaultdict(list)
    for index, bbox in enumerate(ocr_bboxes):
        for group_id, table_box in enumerate(table_bboxes):
            iou, bbox_in = intersection(table_box, bbox)
            if bbox_in > 0.05:
                group_bboxes[group_id].append(copy.deepcopy(bbox))
                group_texts[group_id].append(copy.deepcopy(ocr_texts[index]))
                group_orders[group_id].append(index)

    return group_bboxes, group_texts, group_orders


def resize(img, ocr_bboxes, table_bboxes, long_edge_size=1600):
    h, w, _ = img.shape
    scale = long_edge_size * 1.0 / max(h, w)
    if h > w:
        newh, neww = long_edge_size, scale * w
    else:
        newh, neww = scale * h, long_edge_size
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    resized_img = cv2.resize(img, dsize=(neww, newh))
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] *
                    resized_img.shape[1] / img.shape[1])

    for index, ocr_bbox in enumerate(ocr_bboxes):
        ocr_bboxes[index] = ocr_bbox * scale
    for index, table_bbox in enumerate(table_bboxes):
        table_bboxes[index] = table_bbox * scale

    return resized_img, ocr_bboxes, table_bboxes


def sorted_table_bboxes(table_bboxes):
    """
    sort table boxes from top to bottom
    """
    direction_vectors = []
    for table_bbox in table_bboxes:
        p1 = (table_bbox[0] + table_bbox[3]) / 2
        p2 = (table_bbox[1] + table_bbox[2]) / 2
        d = standard_vec(p2 - p1)
        direction_vectors.append(d)

    vec_x = np.mean(np.array(direction_vectors)[:, 0])
    vec_y = np.mean(np.array(direction_vectors)[:, 1])
    box_direction = standard_vec([vec_x, vec_y])
    if np.isnan(box_direction[0]) or np.isnan(box_direction[1]):
        box_direction = np.array([1.0, 0.0])

    pts_0 = table_bboxes[:, 0]
    projected_pts_0 = np.array(
        [project_pt_to_axis(pt, box_direction) for pt in pts_0])
    # sort y coordinates
    sorted_inds = np.argsort(projected_pts_0[:, 1])

    return sorted_inds


def project_pt_to_axis(pt, axis_x_vec):
    def get_vertical_direction(direction):
        assert (direction[0] != 0) or (direction[1] != 0)
        v_direction = np.array([-direction[1], direction[0]])
        return v_direction

    # axis_x_vec is standard x vec
    axis_y_vec = standard_vec(get_vertical_direction(axis_x_vec))
    projected_pt_x = np.sum(pt * axis_x_vec)
    projected_pt_y = np.sum(pt * axis_y_vec)
    return np.array([projected_pt_x, projected_pt_y])


def compute_boxes_direction(valid_bbox_angle, table_ocr_bboxes):
    """
    compute boxes direction by standard_vec
    """
    long_boxes_exist = True  # w/h >= 4 bbox
    direction_vectors = []
    for bbox_angle, bbox_index in valid_bbox_angle:
        box = table_ocr_bboxes[bbox_index]
        w = np.linalg.norm(box[1] - box[0])
        h = np.linalg.norm(box[3] - box[0])
        # only compute w/h >= 4 bbox angle which angle is more accurate,
        # ignore w/h < 4 bbox
        if w < 4 * h:
            continue
        p1 = (box[0] + box[3]) / 2
        p2 = (box[1] + box[2]) / 2
        d = standard_vec(p2 - p1)
        direction_vectors.append(d)

    if len(direction_vectors) == 0:
        long_boxes_exist = False
        for bbox_angle, bbox_index in valid_bbox_angle:
            box = table_ocr_bboxes[bbox_index]
            p1 = (box[0] + box[3]) / 2
            p2 = (box[1] + box[2]) / 2
            d = standard_vec(p2 - p1)
            direction_vectors.append(d)

    vec_x = np.mean(np.array(direction_vectors)[:, 0])
    vec_y = np.mean(np.array(direction_vectors)[:, 1])
    box_direction = standard_vec([vec_x, vec_y])
    if np.isnan(box_direction[0]) or np.isnan(box_direction[1]):
        box_direction = np.array([1.0, 0.0])

    return box_direction, long_boxes_exist


def standard_vec(line):
    if not isinstance(line, np.ndarray):
        line = np.array(line)
    line_len = np.linalg.norm(line)
    return line / line_len


structure_class_names = [
    'table', 'table column', 'table row', 'table column header',
    'table projected row header', 'table spanning cell', 'no object'
]
structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
structure_class_thresholds = {
    'table': 0.3,
    'table column': 0.3,
    'table row': 0.3,
    'table column header': 0.3,
    'table projected row header': 0.3,
    'table spanning cell': 0.3,
    'no object': 10
}


@APP_REGISTRY.register_module()
class TableRowColApp(BaseApp):
    def __init__(self, app_params, app_inputs, app_ouputs, **kwargs):
        super(TableRowColApp, self).__init__(app_params, app_inputs,
                                             app_ouputs, **kwargs)
        assert 'table_det_model_name' in app_params, \
            'Please specify table_det_model_name in config.pbtxt'
        self.table_det_model_name = \
            app_params['table_det_model_name']['string_value']
        self.table_det_model_version = int(
            app_params['table_det_model_version']['string_value']
        ) if 'table_det_model_version' in app_params else -1
        self.table_det_model_inputs = ['image', 'params']
        self.table_det_model_outputs = ['boxes']

        assert 'table_rowcol_det_model_name' in app_params, \
            'Please specify table_rowcol_det_model_name in config.pbtxt'
        self.table_rowcol_det_model_name = \
            app_params['table_rowcol_det_model_name']['string_value']
        self.table_rowcol_det_model_version = int(
            app_params['table_rowcol_det_model_version']['string_value']
        ) if 'table_rowcol_det_model_version' in app_params else -1
        self.table_rowcol_det_model_inputs = ['image', 'params']
        self.table_rowcol_det_model_outputs = ['boxes', 'scores', 'labels']
        self.padding = 50

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): TableRowColApp global information
            inputs (list[np.array]): [image_b64], inputs of app
        Returns:
            context (dict): TableRowColApp global information
            outputs (list[np.array]): [seal_results], outputs of app
        """
        table_det_longer_edge_size = context['params'].get(
            'table_det_longer_edge_size', 0)
        table_rowcol_det_longer_edge_size = context['params'].get(
            'table_rowcol_det_longer_edge_size', 0)
        sep_char = context['params'].get('sep_char', '')
        # debug = context['params'].get('debug', False)

        image = inputs[0]
        image = image.astype(np.float32)

        # phase1: ocr results
        ocr_result_byte = inputs[1][0]
        ocr_result = json.loads(ocr_result_byte.decode('utf-8'))
        ocr_bboxes = np.asarray(ocr_result['bboxes'])
        ocr_bboxes_origin = copy.deepcopy(ocr_bboxes)
        ocr_texts = ocr_result['texts']

        # phase2: table detect
        table_det_params = {}
        if table_det_longer_edge_size != 0:
            table_det_params['longer_edge_size'] = table_det_longer_edge_size
        det_params = np.array([json.dumps(table_det_params).encode('utf-8')
                               ]).astype(np.object_)
        det_inputs = [image.astype(np.float32), det_params]
        det_outputs = self.alg_infer(det_inputs, self.table_det_model_name,
                                     self.table_det_model_version,
                                     self.table_det_model_inputs,
                                     self.table_det_model_outputs)
        table_bboxes = det_outputs[0].tolist()

        output_res = dict()
        if len(table_bboxes) != 0:
            table_bboxes = np.asarray(table_bboxes).reshape(-1, 4, 2)
            # sort table bboxes from top to bottom
            sorted_inds = sorted_table_bboxes(table_bboxes)
            table_bboxes = table_bboxes[sorted_inds]

            # phase3: resize origin image egde size to 1600 and split
            #         ocr_bboxes to diffenert group according table_bboxes
            resize_img, ocr_bboxes, table_bboxes = resize(
                image, ocr_bboxes, table_bboxes)
            ocr_group_bboxes, ocr_group_texts, ocr_group_orders = \
                group_ocr_bboxes(ocr_bboxes, ocr_texts, table_bboxes)

            b64_vis_table_img = None
            raw_table_result = []
            raw_table_html = ''
            # process each table
            for table_index, table_bbox in enumerate(table_bboxes):
                table_ocr_bboxes = ocr_group_bboxes[table_index]
                table_ocr_texts = ocr_group_texts[table_index]
                table_ocr_orders = ocr_group_orders[table_index]

                # no ocr results in table, no need to process
                if len(table_ocr_bboxes) == 0:
                    print(f'There are no ocr results in table {table_index}.')
                    continue

                # phase4: rotate image by ocr_bboxes angle and crop table
                table_angle = compute_angle(table_bbox[1] - table_bbox[0])
                valid_bbox_angle = []
                for bbox_index, ocr_bbox in enumerate(table_ocr_bboxes):
                    bbox_angle = compute_angle(ocr_bbox[1] -
                                               ocr_bbox[0])  # range(0, 360)
                    # assume the first point and angle of table_bbox is right
                    delta = min(abs(table_angle - bbox_angle),
                                360 - abs(table_angle - bbox_angle))
                    # filter ocr_bbox which delta angle greater than 10
                    # (filter some wrong ocr bbox angles)
                    if delta < 10:
                        valid_bbox_angle.append([bbox_angle, bbox_index])

                valid_bbox_num = len(valid_bbox_angle)
                if valid_bbox_num > 0:
                    if valid_bbox_num < len(table_ocr_bboxes) // 2:
                        print(
                            'More than half ocr_bbox angles diff with table ' +
                            'angle. Please check table or ocr_bbox angle.')
                    # rotate angle is computed by valid ocr bbox angle
                    box_direction, long_boxes_exist = compute_boxes_direction(
                        valid_bbox_angle, table_ocr_bboxes)
                    if not long_boxes_exist:
                        print(f'All ocr boxes w/h in table {table_index} < 4.')
                    mean_angle = compute_angle(box_direction)
                else:
                    # rotate angle is table_angle
                    mean_angle = table_angle

                img_rotate, old_center, new_center = rotate_image_only(
                    resize_img, mean_angle)
                img_h, img_w, _ = img_rotate.shape
                table_bbox = rotate_polys_only(new_center, old_center,
                                               [table_bbox], mean_angle)[0]
                table_ocr_bboxes = rotate_polys_only(new_center, old_center,
                                                     table_ocr_bboxes,
                                                     mean_angle)
                x1, y1 = min(table_bbox[:, 0]), min(table_bbox[:, 1])
                x2, y2 = max(table_bbox[:, 0]), max(table_bbox[:, 1])
                crop_x1, crop_y1 = max(0, x1 - self.padding), max(
                    0, y1 - self.padding)
                crop_x2, crop_y2 = min(img_w, x2 + self.padding), min(
                    img_h, y2 + self.padding)
                pts = [[crop_x1, crop_y1], [crop_x2, crop_y1],
                       [crop_x2, crop_y2], [crop_x1, crop_y2]]
                crop_table_img = perspective_transform(img_rotate, pts)

                table_bbox[:, 0] = table_bbox[:, 0] - crop_x1
                table_bbox[:, 1] = table_bbox[:, 1] - crop_y1
                table_ocr_bboxes[:, :, 0] = table_ocr_bboxes[:, :, 0] - crop_x1
                table_ocr_bboxes[:, :, 1] = table_ocr_bboxes[:, :, 1] - crop_y1

                # phase5: predict table, row, col, spannig cell bouding box
                table_rowcol_det_params = {}
                if table_rowcol_det_longer_edge_size != 0:
                    table_rowcol_det_params[
                        'longer_edge_size'] = table_rowcol_det_longer_edge_size
                det_params = np.array([
                    json.dumps(table_rowcol_det_params).encode('utf-8')
                ]).astype(np.object_)
                det_inputs = [crop_table_img.astype(np.float32), det_params]
                det_outputs = self.alg_infer(
                    det_inputs, self.table_rowcol_det_model_name,
                    self.table_rowcol_det_model_version,
                    self.table_rowcol_det_model_inputs,
                    self.table_rowcol_det_model_outputs)
                row_col_bboxes = det_outputs[0].tolist()
                scores = det_outputs[1].tolist()
                labels = det_outputs[2].tolist()

                if len(row_col_bboxes) > 0:
                    row_col_bboxes = np.asarray(row_col_bboxes).reshape(
                        -1, 4, 2)
                    scores = np.asarray(scores).reshape(-1)
                    labels = np.asarray(labels).reshape(-1)

                # todo: only support horizontal box, future support rotate box
                page_tokens = []
                for index, bbox in enumerate(table_ocr_bboxes):
                    x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(
                        bbox[:, 0]), max(bbox[:, 1])
                    ocr_res = dict()
                    ocr_res['bbox'] = [x1, y1, x2, y2]
                    ocr_res['text'] = table_ocr_texts[index]
                    ocr_res['flags'] = 0
                    ocr_res['line_num'] = 0
                    ocr_res['block_num'] = 0
                    ocr_res['span_num'] = index
                    ocr_res['order'] = table_ocr_orders[index]
                    page_tokens.append(ocr_res)

                # todo: only support horizontal box, future support rotate box
                row_col_bboxes_rect = []
                for index, bbox in enumerate(row_col_bboxes):
                    x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(
                        bbox[:, 0]), max(bbox[:, 1])
                    row_col_bboxes_rect.append([x1, y1, x2, y2])
                    # start from 0
                    labels[index] = labels[index] - 1

                # phase6: convert row, column, span cell boxes to cell boxes
                _, cells, _, rows_columns_cells = objects_to_cells(
                    row_col_bboxes_rect,
                    labels,
                    scores,
                    page_tokens,
                    structure_class_names,
                    structure_class_thresholds,
                    structure_class_map,
                    row_col_refine=False,
                    row_col_align=True,
                    adjust_by_ocr_boxes=False,
                    sep_char=sep_char)

                if ('rows' not in rows_columns_cells
                        or 'columns' not in rows_columns_cells
                        or 'cells' not in rows_columns_cells):
                    print(f'No rows or cols or cells in table {table_index}.')
                    continue

                # phase7: generate html of each table and convert to excel
                # Searching empty cells and recording them through arearec
                row_nums = len(rows_columns_cells['rows'])
                column_nums = len(rows_columns_cells['columns'])
                table = dict()
                table['rows'] = row_nums
                table['cols'] = column_nums
                arearec = np.zeros([row_nums, column_nums])
                empty_index = -1  # deal with empty cell
                no_empty_index = 0
                no_empty_cell_labels = []
                no_empty_cell_texts = []
                cell_infos = []
                for cellid, cell in enumerate(rows_columns_cells['cells']):
                    rows = cell['row_nums']
                    columns = cell['column_nums']
                    srow, scol, erow, ecol = min(rows), min(columns), max(
                        rows), max(columns)
                    if 'cell_text' in cell and cell['cell_text']:
                        # no empty cell
                        arearec[srow:erow + 1,
                                scol:ecol + 1] = no_empty_index + 1
                        no_empty_cell_labels.append(
                            [1])  # assume no head, same with cell method
                        no_empty_cell_texts.append(cell['cell_text'])
                        no_empty_index += 1
                    else:
                        # empty cell
                        arearec[srow:erow + 1, scol:ecol + 1] = empty_index
                        empty_index -= 1
                    cell_info = dict()
                    x1, y1, x2, y2 = cell['bbox']
                    cell_info['box'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    if 'spans' in cell:
                        cell_info['text'] = [
                            ocr_res['text'] for ocr_res in cell['spans']
                        ]
                        cell_info['text_box'] = [
                            ocr_bboxes_origin[ocr_res['order']]
                            for ocr_res in cell['spans']
                        ]
                    else:
                        cell_info['text'] = ['']
                        cell_info['text_box'] = []
                    cell_info['row'] = srow
                    cell_info['col'] = scol
                    cell_info['rows'] = rows
                    cell_info['cols'] = columns
                    cell_infos.append(cell_info)
                table['cell_infos'] = cell_infos
                raw_table_result.append(table)

                html_str_rec, html_text_rec = area_to_html(
                    arearec, no_empty_cell_labels, no_empty_cell_texts)
                table_html = format_html(html_str_rec, html_text_rec)
                raw_table_html += table_html

                # if b64_vis_table_img is None:
                #     # todo: vis table need remove
                #     b64_vis_table_img = ocr_visual(image=crop_table_img,
                #                                    res={
                #                                        'bboxes':
                #                                        table_ocr_bboxes,
                #                                        'texts':
                #                                        table_ocr_texts
                #                                    },
                #                                    draw_number=False)
                #     # show table cell results
                #     table_cell_bboxes = []
                #     for cell_info in cell_infos:
                #         table_cell_bboxes.append(np.array(cell_info['box']))
                #     b64_vis_table_img = draw_box_on_img(
                #         b64_vis_table_img, table_cell_bboxes)

            if raw_table_result:
                fid = io.BytesIO()
                workbook = document_to_workbook(raw_table_html)
                fid.seek(0)
                workbook.save(fid)
                output_res['resultFile'] = base64.b64encode(
                    fid.getvalue()).decode('ascii').replace('\n', '')
                output_res['raw_result'] = raw_table_result
                output_res['resultImg'] = b64_vis_table_img
                output_res = serilize_json(output_res)

        infer_outputs = [
            np.array([json.dumps(output_res).encode('utf-8')
                      ]).astype(np.object_)
        ]
        return context, infer_outputs


@APP_REGISTRY.register_module()
class TableCellApp(BaseApp):
    def __init__(self, app_params, app_inputs, app_ouputs, **kwargs):
        super(TableCellApp, self).__init__(app_params, app_inputs, app_ouputs,
                                           **kwargs)
        assert 'table_det_model_name' in app_params, \
            'Please specify table_det_model_name in config.pbtxt'
        self.table_det_model_name = \
            app_params['table_det_model_name']['string_value']
        self.table_det_model_version = int(
            app_params['table_det_model_version']['string_value']
        ) if 'table_det_model_version' in app_params else -1
        self.table_det_model_inputs = ['image', 'params']
        self.table_det_model_outputs = ['boxes']

        assert 'table_cell_det_model_name' in app_params, \
            'Please specify table_cell_det_model_name in config.pbtxt'
        self.table_cell_det_model_name = \
            app_params['table_cell_det_model_name']['string_value']
        self.table_cell_det_model_version = int(
            app_params['table_cell_det_model_version']['string_value']
        ) if 'table_cell_det_model_version' in app_params else -1
        self.table_cell_det_model_inputs = ['image', 'params']
        self.table_cell_det_model_outputs = ['boxes']
        self.post_cell = PostCell()
        self.padding = 50

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): TableRowColApp global information
            inputs (list[np.array]): [image_b64], inputs of app
        Returns:
            context (dict): TableRowColApp global information
            outputs (list[np.array]): [seal_results], outputs of app
        """
        table_det_longer_edge_size = context['params'].get(
            'table_det_longer_edge_size', 0)
        table_cell_det_longer_edge_size = context['params'].get(
            'table_cell_det_longer_edge_size', 0)
        sep_char = context['params'].get('sep_char', '')
        # debug = context['params'].get('debug', False)

        image = inputs[0]
        image = image.astype(np.float32)

        # phase1: ocr results
        ocr_result_byte = inputs[1][0]
        ocr_result = json.loads(ocr_result_byte.decode('utf-8'))
        ocr_bboxes = np.asarray(ocr_result['bboxes'])
        ocr_bboxes_origin = copy.deepcopy(ocr_bboxes)
        ocr_texts = ocr_result['texts']

        # phase2: table detect
        table_det_params = {}
        if table_det_longer_edge_size != 0:
            table_det_params['longer_edge_size'] = table_det_longer_edge_size
        det_params = np.array([json.dumps(table_det_params).encode('utf-8')
                               ]).astype(np.object_)
        det_inputs = [image.astype(np.float32), det_params]
        det_outputs = self.alg_infer(det_inputs, self.table_det_model_name,
                                     self.table_det_model_version,
                                     self.table_det_model_inputs,
                                     self.table_det_model_outputs)
        table_bboxes = det_outputs[0].tolist()

        output_res = dict()
        if len(table_bboxes) != 0:
            table_bboxes = np.asarray(table_bboxes).reshape(-1, 4, 2)
            # sort table bboxes from top to bottom
            sorted_inds = sorted_table_bboxes(table_bboxes)
            table_bboxes = table_bboxes[sorted_inds]

            # phase3: resize origin image egde size to 1600 and split
            #         ocr_bboxes to diffenert group according table_bboxes
            resize_img, ocr_bboxes, table_bboxes = resize(
                image, ocr_bboxes, table_bboxes)
            ocr_group_bboxes, ocr_group_texts, ocr_group_orders = \
                group_ocr_bboxes(ocr_bboxes, ocr_texts, table_bboxes)

            b64_vis_table_img = None
            raw_table_result = []
            raw_table_html = ''
            # process each table
            for table_index, table_bbox in enumerate(table_bboxes):
                table_ocr_bboxes = ocr_group_bboxes[table_index]
                table_ocr_texts = ocr_group_texts[table_index]
                table_ocr_orders = ocr_group_orders[table_index]

                # no ocr results in table, no need to process
                if len(table_ocr_bboxes) == 0:
                    print(f'There are no ocr results in table {table_index}.')
                    continue

                # phase4: rotate image by ocr_bboxes angle and crop table
                table_angle = compute_angle(table_bbox[1] - table_bbox[0])
                valid_bbox_angle = []
                for bbox_index, ocr_bbox in enumerate(table_ocr_bboxes):
                    bbox_angle = compute_angle(ocr_bbox[1] -
                                               ocr_bbox[0])  # range(0, 360)
                    # assume the first point and angle of table_bbox is right
                    delta = min(abs(table_angle - bbox_angle),
                                360 - abs(table_angle - bbox_angle))
                    # filter ocr_bbox which delta angle greater than 10
                    # (filter some wrong ocr bbox angles)
                    if delta < 10:
                        valid_bbox_angle.append([bbox_angle, bbox_index])

                valid_bbox_num = len(valid_bbox_angle)
                if valid_bbox_num > 0:
                    if valid_bbox_num < len(table_ocr_bboxes) // 2:
                        print(
                            'More than half ocr_bbox angles diff with table ' +
                            'angle. Please check table or ocr_bbox angle.')
                    # rotate angle is computed by valid ocr bbox angle
                    box_direction, long_boxes_exist = compute_boxes_direction(
                        valid_bbox_angle, table_ocr_bboxes)
                    if not long_boxes_exist:
                        print(f'All ocr boxes w/h in table {table_index} < 4.')
                    mean_angle = compute_angle(box_direction)
                else:
                    # rotate angle is table_angle
                    mean_angle = table_angle

                img_rotate, old_center, new_center = rotate_image_only(
                    resize_img, mean_angle)
                img_h, img_w, _ = img_rotate.shape
                table_bbox = rotate_polys_only(new_center, old_center,
                                               [table_bbox], mean_angle)[0]
                table_ocr_bboxes = rotate_polys_only(new_center, old_center,
                                                     table_ocr_bboxes,
                                                     mean_angle)
                x1, y1 = min(table_bbox[:, 0]), min(table_bbox[:, 1])
                x2, y2 = max(table_bbox[:, 0]), max(table_bbox[:, 1])
                crop_x1, crop_y1 = max(0, x1 - self.padding), max(
                    0, y1 - self.padding)
                crop_x2, crop_y2 = min(img_w, x2 + self.padding), min(
                    img_h, y2 + self.padding)
                pts = [[crop_x1, crop_y1], [crop_x2, crop_y1],
                       [crop_x2, crop_y2], [crop_x1, crop_y2]]
                crop_table_img = perspective_transform(img_rotate, pts)

                table_bbox[:, 0] = table_bbox[:, 0] - crop_x1
                table_bbox[:, 1] = table_bbox[:, 1] - crop_y1
                table_ocr_bboxes[:, :, 0] = table_ocr_bboxes[:, :, 0] - crop_x1
                table_ocr_bboxes[:, :, 1] = table_ocr_bboxes[:, :, 1] - crop_y1

                # phase5: predict table cell bouding box
                table_cell_det_params = {}
                if table_cell_det_longer_edge_size != 0:
                    table_cell_det_params[
                        'longer_edge_size'] = table_cell_det_longer_edge_size
                det_params = np.array([
                    json.dumps(table_cell_det_params).encode('utf-8')
                ]).astype(np.object_)
                det_inputs = [crop_table_img.astype(np.float32), det_params]
                det_outputs = self.alg_infer(det_inputs,
                                             self.table_cell_det_model_name,
                                             self.table_cell_det_model_version,
                                             self.table_cell_det_model_inputs,
                                             self.table_cell_det_model_outputs)
                table_cell_bboxes = det_outputs[0].tolist()

                if len(table_cell_bboxes) > 0:
                    table_cell_bboxes = np.asarray(table_cell_bboxes).reshape(
                        -1, 4, 2)

                # todo: only support horizontal box, future support rotate box
                table_ocr_bboxes_rect = []
                for index, bbox in enumerate(table_ocr_bboxes):
                    x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(
                        bbox[:, 0]), max(bbox[:, 1])
                    table_ocr_bboxes_rect.append([x1, y1, x2, y2])

                # todo: only support horizontal box, future support rotate box
                table_cell_bboxes_rect = []
                for index, bbox in enumerate(table_cell_bboxes):
                    x1, y1, x2, y2 = min(bbox[:, 0]), min(bbox[:, 1]), max(
                        bbox[:, 0]), max(bbox[:, 1])
                    table_cell_bboxes_rect.append([x1, y1, x2, y2])

                # phase6: table cell box postprocess and convert to excel
                bboxes_results = []
                # head
                bboxes_results.append(np.empty((0, 4), dtype=float))
                # body
                bboxes_results.append(np.asarray(table_cell_bboxes_rect))
                if bboxes_results[1].shape == (0, ):
                    continue
                # batch=1
                table_results = self.post_cell.post_processing(
                    [bboxes_results],
                    ocr_results=[{
                        'bboxes': table_ocr_bboxes_rect,
                        'texts': table_ocr_texts,
                        'orders': table_ocr_orders
                    }],
                    sep_char=sep_char)
                content_ann = table_results[0]['content_ann']
                bboxes = content_ann['bboxes']
                cells = content_ann['cells']
                cells_matched_texts = content_ann['cells_matched_texts']
                cells_matched_orders = content_ann['cells_matched_orders']
                table = dict()
                table['rows'] = np.array(cells)[:, 2].max() + 1
                table['cols'] = np.array(cells)[:, 3].max() + 1
                cell_infos = []
                for cellid, cell in enumerate(cells):
                    srow, scol, erow, ecol = cell
                    cell_info = dict()
                    if bboxes[cellid]:
                        x1, y1, x2, y2 = bboxes[cellid]
                        cell_info['box'] = [[x1, y1], [x2, y1], [x2, y2],
                                            [x1, y2]]
                        cell_info['text'] = cells_matched_texts[cellid]
                        cell_info['text_box'] = [
                            ocr_bboxes_origin[order]
                            for order in cells_matched_orders[cellid]
                        ]
                    else:
                        cell_info['box'] = [[-1, -1], [-1, -1], [-1, -1],
                                            [-1, -1]]
                        cell_info['text'] = ['']
                        cell_info['text_box'] = []
                    cell_info['row'] = srow
                    cell_info['col'] = scol
                    cell_info['rows'] = list(range(srow, erow + 1))
                    cell_info['cols'] = list(range(scol, ecol + 1))
                    cell_infos.append(cell_info)
                table['cell_infos'] = cell_infos
                raw_table_result.append(table)
                raw_table_html += table_results[0]['html']

                # if b64_vis_table_img is None:
                #     # todo: vis table need remove
                #     b64_vis_table_img = ocr_visual(image=crop_table_img,
                #                                    res={
                #                                        'bboxes':
                #                                        table_ocr_bboxes,
                #                                        'texts':
                #                                        table_ocr_texts
                #                                    },
                #                                    draw_number=False)
                #     # show table cell results
                #     table_cell_bboxes = []
                #     for cell_info in cell_infos:
                #         if cell_info['box'] != [[-1, -1], [-1, -1], [-1, -1],
                #                                 [-1, -1]]:
                #             table_cell_bboxes.append(np.array(
                #                 cell_info['box']))
                #     b64_vis_table_img = draw_box_on_img(
                #         b64_vis_table_img, table_cell_bboxes)

            if raw_table_result:
                fid = io.BytesIO()
                workbook = document_to_workbook(raw_table_html)
                fid.seek(0)
                workbook.save(fid)
                output_res['resultFile'] = base64.b64encode(
                    fid.getvalue()).decode('ascii').replace('\n', '')
                output_res['raw_result'] = raw_table_result
                output_res['resultImg'] = b64_vis_table_img
                output_res = serilize_json(output_res)

        infer_outputs = [
            np.array([json.dumps(output_res).encode('utf-8')
                      ]).astype(np.object_)
        ]
        return context, infer_outputs
