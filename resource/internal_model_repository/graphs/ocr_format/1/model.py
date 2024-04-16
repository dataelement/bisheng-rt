# flake8: noqa
import json
from collections import Counter
from typing import List, Optional, Union

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from pydantic import BaseModel, RootModel
from shapely.geometry import Polygon


def compute_point_line_distance(p1, p2, direction):
    # """
    # compute distance between a point and a line
    # :param p1: point in the line
    # :param p2: point out of line
    # :param direction: line direciton
    # :return: distance
    # """
    p1 = np.float32(p1)
    p2 = np.float32(p2)
    direction = np.float32(direction)

    a = p2 - p1
    b = direction
    m = np.dot(a, b)
    n = np.dot(b, b)

    c = b * (m / n)
    p = a - c
    q = np.dot(p, p)
    e = np.sqrt(q)

    tmp = float(a[1] * b[0] - b[1] * a[0])
    if tmp < 0:
        return e
    else:
        return -e


def inside_ratio(section1: list, section2: list):
    tmp = section1 + section2
    inds = np.argsort(tmp)
    if sum(inds[:2]) == 1 or sum(inds[:2]) == 5:
        return 0
    else:
        iou = tmp[inds[2]] - tmp[inds[1]]
        return max(iou / abs(section1[1] - section1[0]),
                   iou / abs(section2[1] - section2[0]))


def get_vertical_direction(direction):
    assert (direction[0] != 0) or (direction[1] != 0)
    if (direction[0] == 0) and (direction[1] != 0):
        return np.array([1, 0])
    if (direction[0] != 0) and (direction[1] == 0):
        return np.array([0, 1])
    v_direction = np.array([1, -(direction[0] / direction[1])])
    if v_direction[1] < 0:
        return -v_direction
    else:
        return v_direction


def standard_vec(line):
    if not isinstance(line, np.ndarray):
        line = np.array(line)
    line_len = np.linalg.norm(line)
    return line / line_len


def dist_euclid(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def project_pt_to_axis(pt, axis_x_vec):
    # axis_x_vec is standard x vec
    axis_y_vec = standard_vec(get_vertical_direction(axis_x_vec))
    projected_pt_x = np.sum(pt * axis_x_vec)
    projected_pt_y = np.sum(pt * axis_y_vec)
    return np.array([projected_pt_x, projected_pt_y])


def perspective_transform(img, pts):
    # pts should be the result of order_points
    # debug w, h
    #W = int(dist_euclid(pts[0], pts[1]))+1
    #H = int(dist_euclid(pts[1], pts[2]))+1
    W = np.round(dist_euclid(pts[0], pts[1]))
    H = np.round(dist_euclid(pts[1], pts[2]))
    pts = np.array(pts, 'float32')
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], 'float32')
    M0 = cv2.getPerspectiveTransform(pts, dst)
    image = cv2.warpPerspective(img, M0, (int(W), int(H)))
    return image


def affine_transform(img, pts):
    W = np.round(dist_euclid(pts[0], pts[1]))
    H = np.round(dist_euclid(pts[1], pts[2]))
    src_3points = np.float32([pts[0], pts[1], pts[2]])
    dest_3points = np.float32([[0, 0], [W, 0], [W, H]])
    M = cv2.getAffineTransform(src_3points, dest_3points)
    m = cv2.warpAffine(img, M, (int(W), int(H)))
    return m


def crop(img, boxes):
    # img is the original image canvas
    # boxes is with dim [N, 8]
    boxes = np.array(boxes).reshape(-1, 4, 2)
    ans = []
    for box in boxes:
        #img0 = perspective_transform(img, box)
        # from v2.0.1, we use affine transform instead
        img0 = affine_transform(img, box)
        ans.append(img0)
    return ans


def crop_online(img, boxes):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = np.array(boxes).reshape(-1, 4, 2)
    ans = []
    for pts in boxes:
        ori_w = np.round(dist_euclid(pts[0], pts[1]))
        ori_h = np.round(dist_euclid(pts[1], pts[2]))
        # online ugly setting, need to do the
        # resize prioir to the preprocessing
        fixed_h = 32
        new_w = CvRound(fixed_h / ori_h * ori_w)
        new_h = np.float32(fixed_h)
        src_3points = np.float32([pts[0], pts[1], pts[2]])
        dest_3points = np.float32([[0, 0], [new_w, 0], [new_w, new_h]])
        M = cv2.getAffineTransform(src_3points, dest_3points)
        m = cv2.warpAffine(img, M, (int(new_w), int(new_h)))
        ans.append(m)
    return ans


def CvRound(v):
    fv = np.round(v)
    return int(fv if abs(fv - v) < 1e-4 else np.floor(v) + 1.0)


def listdir(path):
    names = os.listdir(path)
    return [os.path.join(path, xx) for xx in names if not xx.startswith('.')]


def order_points(pts):
    # pts 4 x 2
    pts = np.array(pts)
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
    return np.array([tl, tr, br, bl], dtype='int32')


def nique_box(box):
    l1 = dist_euclid(box[0], box[1])
    l2 = dist_euclid(box[1], box[2])
    if l1 / l2 >= 0.4:
        b0 = box
        b1 = [box[2], box[3], box[0], box[1]]
    else:
        b0 = [box[1], box[2], box[3], box[0]]
        b1 = [box[3], box[0], box[1], box[2]]
    return b0


def get_image_direction(boxes):
    boxes = np.array(boxes, dtype=np.int32)
    boxes_order = [order_points(box) for box in boxes]
    rotate_static = []
    for box, box_o in zip(boxes, boxes_order):
        _box = box.copy()
        for angle in [0, 90, 180, -90]:
            if (_box == box_o).all():
                rotate_static.append(angle)
                break
            _box = np.roll(_box, 1, axis=0)

    rotate_angle = Counter(rotate_static).most_common(1)[0][0]
    return rotate_angle


def rotate_process(boxes, image_size, rotate_angle):
    if rotate_angle == 0:
        return boxes

    # 计算旋转矩阵 R
    w, h = image_size
    origin = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    R = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1)
    _origin = np.concatenate([origin, np.array([[1], [1], [1], [1]])], axis=1)
    origin_rotate = _origin.dot(R.transpose((1, 0)))

    R = cv2.getPerspectiveTransform(np.float32(origin),
                                    np.float32(origin_rotate))
    R[0, 2] -= np.min(origin_rotate[:, 0])
    R[1, 2] -= np.min(origin_rotate[:, 1])

    # 图片旋转后，boxes的坐标也相应地做下变换
    boxes = np.array(boxes)
    boxes = cv2.perspectiveTransform(
        boxes.reshape(1, -1, 2).astype(np.float32), R).reshape(-1, 4, 2)
    return boxes


class Box(object):
    def __init__(self, box):
        if isinstance(box, list):
            box = np.array(box, dtype=np.int32)
        self.box = box

    def get_box_size(self):
        w1 = np.sqrt(np.sum(np.square(self.box[1] - self.box[0])))
        h1 = np.sqrt(np.sum(np.square(self.box[3] - self.box[0])))
        return w1, h1

    def get_box_center(self):
        return (self.box[0] + self.box[2]) / 2


# Simple version of Boxes from ocr-engine
class BoxesV2(object):
    def __init__(self, boxes):
        if isinstance(boxes, list):
            boxes = np.array(boxes, dtype=np.int32)
        self.boxes = boxes

    def compute_boxes_direction(self):
        direction_vectors = []
        wh_ratio = []
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            w_square, h_square = Box(box).get_box_size()
            wh_ratio.append(w_square / h_square)
            if w_square < 4 * h_square:
                continue

            p1 = (box[0] + box[3]) / 2
            p2 = (box[1] + box[2]) / 2
            direction_vectors.append(standard_vec(p2 - p1))

        if len(direction_vectors) == 0:
            if len(self.boxes) < 6:
                best_boxes = self.boxes
            else:
                best_boxes = self.boxes[np.argpartition(
                    -np.array(wh_ratio), 5)[:5]]
            for i in range(len(best_boxes)):
                box = best_boxes[i]
                p1 = (box[0] + box[3]) / 2
                p2 = (box[1] + box[2]) / 2
                direction_vectors.append(standard_vec(p2 - p1))

        start_ind = int(0.25 * len(direction_vectors))
        end_ind = max(int(0.75 * len(direction_vectors)), start_ind + 1)
        vec_x = np.mean(
            np.sort(np.array(direction_vectors)[:, 0])[start_ind:end_ind])
        vec_y = np.mean(
            np.sort(np.array(direction_vectors)[:, 1])[start_ind:end_ind])
        box_direction = standard_vec([vec_x, vec_y])

        return box_direction

    def sort_rotated_boxes(self):
        axis_x_vec = self.compute_boxes_direction()
        pts_0 = self.boxes[:, 0]
        projected_pts_0 = np.array(
            [project_pt_to_axis(pt, axis_x_vec) for pt in pts_0])
        sorted_inds = np.argsort(projected_pts_0[:, 1])
        return sorted_inds


# Original Boxes implemented from socr module
class Boxes(object):
    def __init__(self, boxes):
        if isinstance(boxes, list):
            boxes = np.array(boxes, dtype=np.int32)
        self.boxes = boxes

    def compute_boxes_direction(self):
        if len(self.boxes) == 0:
            return np.array([1, 0])

        direction_vectors = []
        wh_ratio = []
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            w_square, h_square = Box(box).get_box_size()
            wh_ratio.append(w_square / h_square)
            if w_square < 4 * h_square:
                continue

            p1 = (box[0] + box[3]) / 2
            p2 = (box[1] + box[2]) / 2
            d = standard_vec(standard_vec(p2 - p1))
            direction_vectors.append(d)

        if len(direction_vectors) == 0:
            if len(self.boxes) < 6:
                best_boxes = self.boxes
            else:
                best_boxes = self.boxes[np.argpartition(
                    -np.array(wh_ratio), 5)[:5]]
            for i in range(len(best_boxes)):
                box = best_boxes[i]
                p1 = (box[0] + box[3]) / 2
                p2 = (box[1] + box[2]) / 2
                d = standard_vec(standard_vec(p2 - p1))
                direction_vectors.append(d)

        start_ind = int(0.25 * len(direction_vectors))
        end_ind = max(int(0.75 * len(direction_vectors)), start_ind + 1)
        vec_x = np.mean(
            np.sort(np.array(direction_vectors)[:, 0])[start_ind:end_ind])
        vec_y = np.mean(
            np.sort(np.array(direction_vectors)[:, 1])[start_ind:end_ind])
        box_direction = standard_vec([vec_x, vec_y])
        if np.isnan(box_direction[0]) or np.isnan(box_direction[1]):
            box_direction = np.array([1.0, 0.0])

        return box_direction

    def get_vertical_direction(self, direction):
        assert (direction[0] != 0) or (direction[1] != 0)
        if (direction[0] == 0) and (direction[1] != 0):
            return np.array([1, 0])
        if (direction[0] != 0) and (direction[1] == 0):
            return np.array([0, 1])
        v_direction = np.array([1, -(direction[0] / direction[1])])
        if v_direction[1] < 0:
            return -v_direction
        else:
            return v_direction

    def get_boxes_center(self):
        if len(self.boxes) == 0:
            return None
        centers = []
        for box in self.boxes:
            center = Box(box).get_box_center()
            centers.append(center)
        return np.asarray(centers)

    def get_mean_box_height(self):
        if len(self.boxes) == 0:
            return None

        h1 = np.sum((self.boxes[:, 0] - self.boxes[:, 3])**2, axis=1)
        h1 = (np.mean(h1, axis=0))**0.5

        h2 = np.sum((self.boxes[:, 1] - self.boxes[:, 2])**2, axis=1)
        h2 = (np.mean(h2, axis=0))**0.5

        return np.mean([h1, h2])

    def calc_mean_size(self):
        if len(self.boxes) > 4:
            w_list = np.sort(
                np.linalg.norm(self.boxes[:, 1, :] - self.boxes[:, 0, :],
                               axis=-1))
            h_list = np.sort(
                np.linalg.norm(self.boxes[:, 3, :] - self.boxes[:, 0, :],
                               axis=-1))

            start_pos = int(len(self.boxes) / 4)
            end_pos = int(len(self.boxes) * 3 / 4)

            mean_w = np.mean(w_list[start_pos:end_pos])
            mean_h = np.mean(h_list[start_pos:end_pos])
        else:
            mean_w = np.mean(
                np.linalg.norm(self.boxes[:, 1, :] - self.boxes[:, 0, :],
                               axis=-1))
            mean_h = np.mean(
                np.linalg.norm(self.boxes[:, 3, :] - self.boxes[:, 0, :],
                               axis=-1))

        return mean_w, mean_h

    def sort_rotated_boxes(self):
        if len(self.boxes) == 0:
            return np.array([], dtype=np.int64)

        box_direction = self.compute_boxes_direction()
        _boxes = np.array([order_points(box) for box in self.boxes],
                          dtype=np.int32)
        _center01 = np.mean(_boxes[:, :2], axis=1)
        _ys = np.array([
            compute_point_line_distance(np.array([0, 0]), c, box_direction)
            for c in _center01
        ])
        re_inds = np.argsort(-_ys)
        return re_inds

    def compute_column_index(self, box, anchor_sections, v_box_direction):
        x1 = compute_point_line_distance(np.array([0, 0]), box[0],
                                         v_box_direction)
        x2 = compute_point_line_distance(np.array([0, 0]), box[1],
                                         v_box_direction)
        align_info = []
        for section in anchor_sections:
            left = abs(x1 - section[0])
            center = abs((x1 + x2) * 0.5 - (section[0] + section[1]) * 0.5)
            right = abs(x2 - section[1])
            align_info.append(min([left, center, right]))

        return int(np.argmin(align_info))

    def rearrange_boxes(self, box_direction=None):
        if len(self.boxes) == 0:
            return np.array([], dtype=np.int64), []

        mean_box_height = self.get_mean_box_height()
        box_direction = (self.compute_boxes_direction()
                         if box_direction is None else box_direction)
        v_box_direction = get_vertical_direction(box_direction)

        boxes = self.boxes.copy()
        _boxes = np.array([order_points(box) for box in boxes], dtype=np.int32)
        _center01 = np.mean(_boxes[:, :2], axis=1)
        _ys = np.array([
            compute_point_line_distance(np.array([0, 0]), c, box_direction)
            for c in _center01
        ])
        re_inds1 = np.argsort(-_ys)
        _ys = _ys[re_inds1]
        _boxes = _boxes[re_inds1]

        x_taken = []
        y_last = float('-inf')
        row_info = []
        for box, y in zip(_boxes, _ys):
            section = [box[0, 0], box[1, 0]]
            if abs(y - y_last) > 0.5 * mean_box_height:
                row_info.append(row_info[-1] + 1 if len(row_info) > 0 else 0)
                x_taken = [section]
                y_last = y
                continue

            another_line = False
            for t in x_taken:
                if inside_ratio(t, section) > 0.5:
                    another_line = True
                    break
            if another_line:
                x_taken = [section]
                y_last = y
                row_info.append(row_info[-1] + 1)
            else:
                x_taken.append(section)
                y_last = y
                row_info.append(row_info[-1])
        row_info = np.array(row_info, dtype=np.int32)

        re_inds2 = []
        num_most_row = (0, 0)
        for row_ind in range(max(row_info) + 1):
            box_inds = np.where(np.array(row_info) == row_ind)[0]
            box_left_x = _boxes[box_inds, 0, 0]
            argsort = np.argsort(box_left_x)
            vv = box_inds[argsort].tolist()
            re_inds2 += vv
            if len(box_inds) > num_most_row[1]:
                num_most_row = (row_ind, len(box_inds))

        _boxes = _boxes[re_inds2]
        row_info = row_info[re_inds2]

        anchor_inds = np.where(row_info == num_most_row[0])[0].tolist()
        anchor_sections = [_boxes[ind] for ind in anchor_inds]
        anchor_sections = [[
            compute_point_line_distance(np.array([0, 0]), box[0],
                                        v_box_direction),
            compute_point_line_distance(np.array([0, 0]), box[1],
                                        v_box_direction)
        ] for box in anchor_sections]
        col_info = []
        for i, box in enumerate(_boxes):
            if i in anchor_inds:
                col_info.append(anchor_inds.index(i))
            else:
                col_info.append(
                    self.compute_column_index(box, anchor_sections,
                                              v_box_direction))

        row_info = row_info.tolist()
        row_col_info = [[r, c] for r, c in zip(row_info, col_info)]

        re_inds = re_inds1[re_inds2]

        return re_inds, row_col_info


def compute_text_direction(boxes):
    return Boxes(boxes).compute_boxes_direction()


def filter_polys(pts):
    '''
    去除完全包含在大框里面的小框
    '''
    pts = np.array(pts)
    x0 = np.min(pts[:, :, 0], axis=1)
    y0 = np.min(pts[:, :, 1], axis=1)
    x1 = np.max(pts[:, :, 0], axis=1)
    y1 = np.max(pts[:, :, 1], axis=1)

    choose_inds = [True] * len(pts)
    valid_pts = list(np.array(pts))
    for i in range(len(valid_pts) - 1, -1, -1):
        A = Polygon(valid_pts[i])
        if A.area == 0:
            valid_pts.pop(i)
            choose_inds[i] = False
            continue
        for j in range(min(i + 10, len(valid_pts) - 1), max(-1, i - 10), -1):
            if i == j:
                continue

            # judge is overlap roughly
            x0_i, y0_i = x0[i], y0[i]
            x0_j, y0_j = x0[j], y0[j]
            x1_i, y1_i = x1[i], y1[i]
            x1_j, y1_j = x1[j], y1[j]
            iou_bb_x0, iou_bb_y0 = max(x0_i, x0_j), max(y0_i, y0_j)
            iou_bb_x1, iou_bb_y1 = min(x1_i, x1_j), min(y1_i, y1_j)
            if not (iou_bb_x1 - iou_bb_x0 > 0 and iou_bb_y1 - iou_bb_y0 > 0):
                continue

            B = Polygon(valid_pts[j])
            if not B.is_valid or not A.is_valid:
                continue
            if A.intersects(B):
                inter = A.intersection(B).area
                if inter / A.area > 0.8:
                    valid_pts.pop(i)
                    choose_inds[i] = False
                    break

    return choose_inds


def reassign_bbox_height(bboxes, mean_h):
    """Reassign bbox by the same height.
    Args:
      mean_h: mean height
      text_direction: [0, 90, -90, 180]
    """
    new_bboxes = []
    for bbox in bboxes:
        # p0, p1, p2, p3 = order_points(bbox)
        p0, p1, p2, p3 = np.array(bbox)
        new_p0 = p3 + standard_vec(p0 - p3) * mean_h
        new_p1 = p2 + standard_vec(p1 - p2) * mean_h
        new_bboxes.append([new_p0, new_p1, p2, p3])

    return np.array(new_bboxes).tolist()


def get_sorted_inds(boxes, filter_polygon, row_col_align_mode):
    """
      logic1: add filter box in ocr-engine (merge from ocr-engine)
      logic2: 计算boxes方向，对boxes重排序，过滤重合box，获取行列信息 (ocrfunc)
      :param boxes:
      :param filter_polygon:
      :return:
    """

    bboxes = boxes
    c0_index = None
    if filter_polygon and len(boxes) > 0:
        reind0 = BoxesV2(boxes).sort_rotated_boxes()
        choose_inds0 = filter_polys(boxes[reind0])
        c0_index = reind0[choose_inds0]
        bboxes = boxes[reind0][choose_inds0]
    else:
        c0_index = np.arange(len(boxes))

    boxes = np.array(bboxes, dtype=np.int32)
    if len(boxes) != 0:
        text_direction = Boxes(boxes).compute_boxes_direction()
    else:
        text_direction = np.array([1, 0])

    choose_inds = None
    if filter_polygon:
        choose_inds = filter_polys(boxes)
        boxes = boxes[choose_inds]
    else:
        choose_inds = np.arange(len(boxes))

    # added by hanfeng
    if row_col_align_mode == 'bottom':
        mean_h = Boxes(boxes).get_mean_box_height()
        boxes = reassign_bbox_height(boxes, mean_h)

    re_inds, row_col_info = (Boxes(boxes).rearrange_boxes(
        box_direction=text_direction))

    c2_index = c0_index[choose_inds][re_inds]
    return c2_index, row_col_info, text_direction



class TensorData(RootModel):
    root: List[Union['TensorData', float, str, bool]]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, dict):
            return v.get('root', [])
        return v


class OcrEngineResult(BaseModel):
    code: Optional[int] = None
    message: Optional[str] = None
    server: Optional[str] = None
    mode: Optional[str] = None
    det: Optional[str] = None
    recog: Optional[str] = None
    rotateupright: Optional[bool] = None
    rotate_angle: Optional[int] = None
    image_size: Optional[List[int]] = None
    text_direction: Optional[List[float]] = None
    bboxes: Optional['TensorData'] = None
    texts: Optional['TensorData'] = None
    row_col_info: Optional[List[List[int]]] = None
    scores: Optional['TensorData'] = None
    bbox_scores: Optional['TensorData'] = None
    scores_norm: Optional['TensorData'] = None
    score_threshold: Optional['TensorData'] = None
    row_col_align_mode: Optional[str] = None


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        params = self.model_config['parameters']
        self.is_elem_ocr = 'is_elem_ocr' in params

    async def execute(self, requests):
        def _get_np_input(request, name, has_batch=True):
            data = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
            return data[0] if has_batch else data

        def _get_optional_input(request, name):
            return pb_utils.get_input_tensor_by_name(request, name)

        def _parse_params(input_params):
            params_default_def = [('refine_boxes', False),
                                  ('process_horizon', False),
                                  ('support_long_image_segment', False),
                                  ('support_long_rotate_dense', False),
                                  ('split_long_sentence_blank', False),
                                  ('ensemble', False),
                                  ('print_hand_cls', False),
                                  ('support_long_rotate_dense', False),
                                  ('unify_text_direction', False),
                                  ('normalize_image_orientation', False),
                                  ('remove_buddhism', True),
                                  ('rotateupright', False),
                                  ('sort_filter_boxes', False),
                                  ('score_threshold', 0.0),
                                  ('row_col_align_mode', 'center'),
                                  ('det', ''), ('recog', ''),
                                  ('enable_merge_seal_result', False)]

            params = dict([(k, input_params.get(k, def_v))
                           for k, def_v in params_default_def])
            return params

        def _parse_ppocr(request):
            bboxes = _get_np_input(request, 'DET_BBOX_NEW')
            bbox_scores = _get_np_input(request, 'DET_BBOX_SCORE_NEW')
            texts = _get_np_input(request, 'REC_TEXT_CH')
            texts_score = _get_np_input(request, 'REC_PROB_CH')
            shape_list = _get_np_input(request, 'SHAPE_LIST')

            angle = _get_np_input(request, 'ANGLE')
            angle_prob = _get_np_input(request, 'ANGLE_PROB')

            h0, w0 = int(shape_list[0]), int(shape_list[1])
            image_size = (w0, h0)

            # reorder by start point
            indexes = []
            for ang, ang_prob in zip(angle, angle_prob):
                if ang == 0:
                    indexes.append([0, 1, 2, 3])
                elif ang == 1:
                    indexes.append([1, 2, 3, 0])
                elif ang == 2 and ang_prob >= 0.90:
                    indexes.append([2, 3, 0, 1])
                elif ang == 2 and ang_prob < 0.90:
                    indexes.append([0, 1, 2, 3])
                elif ang == 3 and ang_prob >= 0.90:
                    indexes.append([3, 0, 1, 2])
                elif ang == 3 and ang_prob < 0.90:
                    indexes.append([1, 2, 3, 0])

            indexes = np.array(indexes)
            bbs = []
            for bb, reind in zip(bboxes, indexes):
                bbs.append(bb[reind])
            bboxes = np.array(bbs).reshape(-1, 4, 2)
            return (bboxes, bbox_scores, texts, texts_score, image_size, None)

        def _parse_elemocr(request):
            bboxes = _get_np_input(request, 'boxes', False)
            bbox_scores = _get_np_input(request, 'boxes_score', False)
            texts = _get_np_input(request, 'texts', False)
            texts_score = _get_np_input(request, 'texts_score', False)
            src_scale = _get_np_input(request, 'src_scale').reshape(-1)
            other_elems = _get_optional_input(request, 'other_elems')

            h0, w0 = int(src_scale[0]), int(src_scale[1])
            image_size = (w0, h0)

            return (bboxes, bbox_scores, texts, texts_score, image_size,
                    other_elems)

        parse_func = _parse_elemocr if self.is_elem_ocr else _parse_ppocr

        responses = []
        for request in requests:
            (bboxes, bbox_scores, texts, texts_score, image_size,
             other_elems) = parse_func(request)

            params_tensor = _get_optional_input(request, 'params')
            if params_tensor:
                if not self.is_elem_ocr:
                    json_str = params_tensor.as_numpy()[0][0]
                else:
                    json_str = params_tensor.as_numpy()[0]
                params = _parse_params(json.loads(json_str))
            else:
                params = _parse_params({})

            # step 1, rotate boxes if nessecary
            rotate_angle = 0
            text_direction = np.array([0, 0])

            if len(bboxes) > 0:
                rotate_angle = get_image_direction(bboxes)
                if params['rotateupright']:
                    w0, h0 = image_size
                    bboxes = rotate_process(bboxes, (w0, h0), rotate_angle)
                    if abs(rotate_angle) == 90:
                        # socr needs (width, height) of upright image
                        image_size = (h0, w0)
                    else:
                        image_size = (w0, h0)

            # step 2. replace text
            if params['remove_buddhism']:
                texts = np.asarray(
                    [text.decode().replace('卍', '') for text in texts])

            # step 3. sort boxes and get row,col information
            scores = texts_score
            scores_norm = texts_score
            c2_index, row_col_info, text_direction = (get_sorted_inds(
                bboxes, params['sort_filter_boxes'],
                params['row_col_align_mode']))

            bboxes = bboxes[c2_index].astype(np.int32)
            bbox_scores = bbox_scores[c2_index]
            texts = texts[c2_index]
            scores = scores[c2_index]
            scores_norm = scores_norm[c2_index]

            # merge ocr results with seal results
            if other_elems is not None and params['enable_merge_seal_result']:
                seal_result = json.loads(other_elems.as_numpy()[0])
                if seal_result['code'] == 200:
                    seal_texts = []
                    seal_boxes = []
                    for seal in seal_result['contents']:
                        for text in seal['texts']:
                            seal_texts.append(text)
                            # todo: improve the true bboxes if needed
                            # share bbox for each text in one seal
                            seal_boxes.append(seal['bbox'])

                    n = len(seal_texts)
                    if n > 0:
                        center = np.mean(np.array(seal_boxes).reshape(n, -1),
                                         axis=1)
                        reind0 = np.argsort(center)
                        seal_boxes = np.array(seal_boxes,
                                              dtype=np.float32)[reind0]
                        seal_texts = np.array(seal_texts)[reind0]

                        texts = np.concatenate((texts, seal_texts),
                                               axis=0).reshape(-1)
                        bboxes = np.concatenate((bboxes, seal_boxes), axis=0)

                        bbox_scores = np.concatenate(
                            (bbox_scores, np.array([0.99] * n,
                                                   dtype=np.float32)),
                            axis=0).reshape(-1)

                        texts_score = np.concatenate(
                            (texts_score, np.array([0.99] * n,
                                                   dtype=np.float32)),
                            axis=0).reshape(-1)

                        scores_norm = texts_score

                        row_col_info += [[1000, 1000 + i] for i in range(n)]

            output1 = OcrEngineResult(
                code=200,
                texts=texts.tolist(),
                scores=scores.tolist(),
                scores_norm=scores_norm.tolist(),
                bboxes=bboxes.tolist(),
                bbox_scores=bbox_scores.tolist(),
                text_direction=text_direction.tolist(),
                rotate_angle=rotate_angle,
                image_size=image_size,
                rotateupright=params['rotateupright'],
                row_col_info=row_col_info,
                row_col_align_mode=params['row_col_align_mode'],
                det=params['det'],
                recog=params['recog'])

            output = np.array([output1.json()])
            out_tensor = pb_utils.Tensor('RESULTS', output.astype(np.bytes_))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses
