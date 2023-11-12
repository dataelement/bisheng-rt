# flake8: noqa
import copy
import json
import math

import numpy as np
import triton_python_backend_utils as pb_utils
from shapely.geometry import Polygon


def pt_in_box(pt, box, epsilon=0.01):
    pt = np.array(pt)
    box = np.array(box)
    vec0 = box[0] - pt
    vec1 = box[1] - pt
    vec2 = box[2] - pt
    vec3 = box[3] - pt

    if (np.linalg.norm(vec0) == 0 or np.linalg.norm(vec1) == 0
            or np.linalg.norm(vec2) == 0 or np.linalg.norm(vec3) == 0):
        return True

    cos0 = np.inner(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1))
    cos1 = np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos2 = np.inner(vec2, vec3) / (np.linalg.norm(vec2) * np.linalg.norm(vec3))
    cos3 = np.inner(vec0, vec3) / (np.linalg.norm(vec3) * np.linalg.norm(vec0))

    cos0 = min(1, cos0) if cos0 > 0 else max(-1, cos0)
    cos1 = min(1, cos1) if cos1 > 0 else max(-1, cos1)
    cos2 = min(1, cos2) if cos2 > 0 else max(-1, cos2)
    cos3 = min(1, cos3) if cos3 > 0 else max(-1, cos3)

    theta0 = math.degrees(math.acos(cos0))
    theta1 = math.degrees(math.acos(cos1))
    theta2 = math.degrees(math.acos(cos2))
    theta3 = math.degrees(math.acos(cos3))

    if abs(theta0 + theta1 + theta2 + theta3 - 360) < epsilon:
        return True
    else:
        return False


class CheckBoxModel(object):
    """Split bboxes by another bboxes.
      author: sunjun@dataelem.com
      reviewer: hanfeng@dataelem.com
    """
    def __init__(self, params={}):
        self.checkbox_iou = params.get('checkbox_iou', 0.3)

    def execute(self, text_bboxes, check_bboxes, text_scores, params):
        """
        Returns:
            ndarray: splited_boxes (n,4,2)
            ndarray: reservered_indexes (n,)
        """
        checkbox_iou = params.get('checkbox_iou', self.checkbox_iou)
        if not check_bboxes.size:
            return text_bboxes, text_scores

        splited_boxes = self.split_bboxes_bycheckbox(text_bboxes, check_bboxes,
                                                     checkbox_iou)

        rearrange_box_idx = self.find_overlap(text_bboxes, check_bboxes,
                                              checkbox_iou)

        rearrange_box_idx = list(set(rearrange_box_idx))
        if splited_boxes.size:
            return self.merge_results(text_bboxes, splited_boxes,
                                      rearrange_box_idx, text_scores)
        else:
            return text_bboxes, text_scores

    def merge_results(self, text_bboxes, splitted_box, rearrange_box_idx,
                      bboxes_scores):
        bboxes_scores = [
            n for i, n in enumerate(bboxes_scores.tolist())
            if i not in rearrange_box_idx
        ]
        #bboxes_scores += [-1.0]*len(rearrange_box_idx)

        text_bboxes = [
            n for i, n in enumerate(text_bboxes.tolist())
            if i not in rearrange_box_idx
        ]
        text_bboxes += [box for box in splitted_box]

        N = len(bboxes_scores)
        bboxes_scores += [-1.0] * (len(text_bboxes) - N)
        return np.array(text_bboxes), np.array(bboxes_scores)

    def calc_abc_from_line_2d(self, point0, point1):
        x0, y0 = point0
        x1, y1 = point1
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    def get_line_cross_point(self, point0, point1, checkbox_point0,
                             checkbox_point1):
        """
        计算两条直线的交点
        """
        a0, b0, c0 = self.calc_abc_from_line_2d(point0, point1)
        a1, b1, c1 = self.calc_abc_from_line_2d(checkbox_point0,
                                                checkbox_point1)
        D = a0 * b1 - a1 * b0
        if D == 0:  # 两条直线平行
            return None
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return x, y

    def two_points_distance(self, point0, point1):
        """
        两点距离
        """
        x1, y1 = point0
        x2, y2 = point1
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_foot(self, p1, p2, p3):
        """
        过p3作p1和p2相连直线的垂线, 计算垂足的坐标
        直线1: 垂足坐标和p3连线
        直线2: p1和p2连线
        两条直线垂直, 且交点为垂足
        """
        if p2[0] != p1[0]:
            # 根据点x1和x2计算线性方程的k, b
            k, b = np.linalg.solve([[p1[0], 1], [p2[0], 1]], [p1[1], p2[1]])
            # 垂直向量，数量积为0
            x = np.divide(((p2[0] - p1[0]) * p3[0] +
                           (p2[1] - p1[1]) * p3[1] - b * (p2[1] - p1[1])),
                          (p2[0] - p1[0] + k * (p2[1] - p1[1])))
            y = k * x + b
        else:  # 点p1和p2的连线垂直于x轴时
            x = p1[0]
            y = p3[1]
        return x, y

    def change_list_shape(self, bbox):
        bbox = bbox.ravel()
        bbox = bbox.reshape(-1, 4, 2)
        return bbox

    def prepare_boxes(self, text_bboxes, check_bboxes):
        ocr_res_bboxes = np.array(text_bboxes)
        checkbox_bboxes = np.array(check_bboxes)
        ocr_res_bboxes = self.change_list_shape(ocr_res_bboxes)
        checkbox_bboxes = self.change_list_shape(checkbox_bboxes)
        return ocr_res_bboxes, checkbox_bboxes

    def find_overlap(self, text_bboxes, check_bboxes, checkbox_iou):
        ocr_res_bboxes, checkbox_bboxes = self.prepare_boxes(
            text_bboxes, check_bboxes)
        rearrange_box_idx = []
        for k, cbox in enumerate(checkbox_bboxes):
            C = Polygon(cbox)
            for i, obox in enumerate(ocr_res_bboxes):
                O = Polygon(obox)
                if not C.intersects(O): continue
                inter = C.intersection(O).area
                scale = inter / C.area
                if scale < checkbox_iou: continue
                rearrange_box_idx.append(i)
        return rearrange_box_idx

    def split_bboxes_bycheckbox(self, text_bboxes, check_bboxes, checkbox_iou):
        ocr_res_bboxes, checkbox_bboxes = self.prepare_boxes(
            text_bboxes, check_bboxes)
        splited_boxes = []
        for k, cbox in enumerate(checkbox_bboxes):
            C = Polygon(cbox)
            overlap_boxes_idx = list()
            for i, obox in enumerate(ocr_res_bboxes):
                O = Polygon(obox)
                if not C.intersects(O): continue
                inter = C.intersection(O).area
                scale = inter / C.area
                if inter / O.area == 1.0: continue
                if scale < checkbox_iou: continue
                overlap_boxes_idx.append({
                    'ocrbox_idx': i,
                    'scale': scale,
                    'obox': obox
                })

            for map_info in overlap_boxes_idx:
                ocrbox_idx = map_info['ocrbox_idx']
                match_obox = map_info['obox']
                splited_box = self.split_box(cbox, match_obox)

                delete_cos2splitbox = ocr_res_bboxes[ocrbox_idx].tolist()
                if delete_cos2splitbox in splited_boxes:
                    # print("delete :",delete_cos2splitbox)
                    splited_boxes.remove(delete_cos2splitbox)

                ocr_res_bboxes = np.delete(ocr_res_bboxes, ocrbox_idx, axis=0)

                if splited_box:
                    splited_boxes += splited_box
                    ocr_res_bboxes = np.vstack(
                        (ocr_res_bboxes, np.array(splited_box)))
        return np.array(splited_boxes)

    def order_points(self, pts):
        # pts = np.array(pts)
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        if xSorted[1][0] == xSorted[2][0] and xSorted[1][1] >= xSorted[2][1]:
            xSorted = xSorted[[0, 2, 1, 3], :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl]).tolist()

    def align_box_direction(self, pairs, splited_box):
        aligned_boxes = []
        for box in splited_box:
            aligned_boxes.append(
                [box[int(pairs[i])] for i in ['0', '1', '2', '3']])
        return aligned_boxes

    def record_pairs(self, ocr_box_ori, ocr_box):
        pairs = dict()
        for i in range(len(ocr_box_ori)):
            pairs[str(i)] = None
            for j in range(len(ocr_box)):
                if ocr_box[j] == ocr_box_ori[i]:
                    pairs[str(i)] = str(j)
        return pairs

    def split_box(self, checkbox, ocr_box):
        """
        根据checkbox 切割全文识别检测框
        """
        checkbox_ori = copy.deepcopy(checkbox).tolist()
        ocr_box_ori = copy.deepcopy(ocr_box).tolist()

        checkbox = self.order_points(checkbox)
        ocr_box = self.order_points(ocr_box)

        pairs = self.record_pairs(ocr_box_ori, ocr_box)
        if None in pairs.values():
            print('Error: ocr text box can not match box afetr order_points()')
            return []

        p0 = list(
            self.get_line_cross_point(ocr_box[0], ocr_box[1], checkbox[0],
                                      checkbox[3]))
        p1 = list(
            self.get_line_cross_point(ocr_box[0], ocr_box[1], checkbox[1],
                                      checkbox[2]))
        p2 = list(
            self.get_line_cross_point(ocr_box[3], ocr_box[2], checkbox[1],
                                      checkbox[2]))
        p3 = list(
            self.get_line_cross_point(ocr_box[3], ocr_box[2], checkbox[0],
                                      checkbox[3]))
        dis_left_0 = self.two_points_distance(checkbox[0], p0)
        dis_left_1 = self.two_points_distance(checkbox[3], p3)
        dis_right_0 = self.two_points_distance(checkbox[1], p1)
        dis_right_1 = self.two_points_distance(checkbox[2], p2)

        # 找出梯形的短边，并向长边投影，算出垂足
        if dis_left_0 > dis_left_1:
            left_foot_point = list(self.get_foot(ocr_box[0], p0, p3))
            split_left_box = [ocr_box[0], left_foot_point, p3, ocr_box[3]]
        else:
            left_foot_point = list(self.get_foot(ocr_box[3], p3, p0))
            split_left_box = [ocr_box[0], p0, left_foot_point, ocr_box[3]]

        if dis_right_0 > dis_right_1:
            right_foot_point = list(self.get_foot(ocr_box[1], p1, p2))
            split_right_box = [right_foot_point, ocr_box[1], ocr_box[2], p2]
        else:
            right_foot_point = list(self.get_foot(ocr_box[2], p2, p1))
            split_right_box = [p1, ocr_box[1], ocr_box[2], right_foot_point]

        splited_box = []
        # checkbox 位于检测框的左侧
        if (pt_in_box(p0,checkbox) and not pt_in_box(p0,ocr_box)) or \
            pt_in_box(p3,checkbox) and not pt_in_box(p3,ocr_box):
            splited_box = [split_right_box]

        # checkbox 位于检测框的右侧
        elif (pt_in_box(p1,checkbox) and not pt_in_box(p1,ocr_box)) or \
            pt_in_box(p2,checkbox) and not pt_in_box(p2,ocr_box):
            splited_box = [split_left_box]

        # checkbox 位于检测框的中间
        else:
            # 过滤 w< h/3 的子框
            if not self.two_points_distance(
                    split_left_box[0],
                    split_left_box[1]) <= 1.0 * self.two_points_distance(
                        split_left_box[0], split_left_box[3]) / 3:
                splited_box.append(split_left_box)

            if not self.two_points_distance(
                    split_right_box[0],
                    split_right_box[1]) <= 1.0 * self.two_points_distance(
                        split_right_box[0], split_right_box[3]) / 3:
                splited_box.append(split_right_box)

        if splited_box and splited_box[0]:
            # 切割完之后，保持子框与原ocr box框同顺序
            splited_box = self.align_box_direction(pairs, splited_box)
        return splited_box


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        params = self.model_config['parameters']
        self.cb_model = CheckBoxModel(params)

    async def execute(self, requests):
        def _get_np_input(request, name, has_batch=True):
            return pb_utils.get_input_tensor_by_name(request, name).as_numpy()

        def _get_optional_params(request, name):
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            return json.loads(tensor.as_numpy[0]) if tensor else {}

        responses = []
        for request in requests:
            text_bboxes = _get_np_input(request, 'boxes')
            checkbox_bboxes = _get_np_input(request, 'checkbox_boxes')
            text_scores = _get_np_input(request, 'scores')
            # params = _get_optional_params(request, 'params')
            params = {}

            text_bboxes, text_scores = self.cb_model.execute(
                text_bboxes, checkbox_bboxes, text_scores, params)

            text_bboxes = text_bboxes.astype(np.float32)
            text_scores = text_scores.astype(np.float32)
            out_tensor0 = pb_utils.Tensor('text_boxes', text_bboxes)
            out_tensor1 = pb_utils.Tensor('text_scores', text_scores)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor0, out_tensor1])
            responses.append(inference_response)

        return responses


def test_checkbox():
    text_bboxes = np.array([0, 0, 100, 0, 100, 10, 0, 10]).reshape((-1, 4, 2))
    cb_bboxes = np.array([50, 0, 60, 0, 60, 10, 50, 10]).reshape((-1, 4, 2))
    cbm = CheckBoxModel()
    r0, r1 = cbm.execute(text_bboxes, cb_bboxes, {'checkbox_iou': 0.1})
    print(r0)
    print(r1)
