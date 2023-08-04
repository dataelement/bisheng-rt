# flake8: noqa
import os
import re
from itertools import chain

import Levenshtein
import numpy as np
from shapely.geometry import Polygon


class EditDistance:
    def __init__(self, str1, str2):
        self.str1 = str1
        self.str2 = str2
        self.paths = []
        self.dp = self.minDistance()
        self.dfs_flag = True

    def minDistance(self):
        n = len(self.str1)
        m = len(self.str2)

        # 有一个字符串为空串
        if n * m == 0:
            return n + m

        # DP 数组
        D = [[0] * (m + 1) for _ in range(n + 1)]

        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j

        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1]
                if self.str1[i - 1] != self.str2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)
        return D

    def dfs(self, i, j, path):
        if self.dfs_flag:
            for next_i, next_j, ops in ([[i - 1, j - 1, 'replace'],
                                         [i - 1, j, 'delete'],
                                         [i, j - 1, 'add']]):
                if 0 <= next_i and 0 <= next_j:
                    if self.dp[next_i][next_j] < self.dp[i][j]:
                        self.dfs(next_i, next_j,
                                 path + [[next_i, next_j, ops]])
                    elif (self.dp[next_i][next_j] == self.dp[i][j]
                          and next_i == i - 1 and next_j == j - 1):
                        if self.str1[next_i] == self.str2[next_j]:
                            ops = ''
                            self.dfs(next_i, next_j,
                                     path + [[next_i, next_j, ops]])
                else:
                    if path not in self.paths:
                        x, y, ops = path[-1]
                        if self.dp[x][y] == 0:
                            self.paths.append(path)
                        if len(self.paths) >= 1000:
                            self.dfs_flag = False

    def editops(self):
        i, j = len(self.dp) - 1, len(self.dp[0]) - 1
        path = [[i, j, '']]

        self.dfs(i, j, path)
        paths_distance = []
        for path in self.paths:
            path_distance = []
            for index in range(len(path) - 1):
                ops = path[index + 1][-1]
                path_distance.append([ops, path[index][0] - 1])
            paths_distance.append(path_distance)
        # 所有path的valid pos
        valid_pos_list = list()
        for path_distance in paths_distance:
            # 一条path的valid pos（去掉delete部分）
            valid_pos = self.get_valid_pos(path_distance)
            valid_pos_list.append(valid_pos)

        # 每一条path的valid pos分段数，找到分段数最少的path
        valid_pos_part_num = np.asarray(
            [len(valid_pos) for valid_pos in valid_pos_list])
        min_part = np.min(valid_pos_part_num)
        candidates_pos = np.where(valid_pos_part_num == min_part)[0]
        # 分段数最少的path进行内部展平，valid_pos_list可能有多个分段数最少的path
        valid_pos_list = [
            list(chain(*valid_pos))
            for valid_pos in np.asarray(valid_pos_list)[candidates_pos]
        ]
        valid_pos_var = []
        # 如果存在多条分段数最少的path，统计方差最少的path，希望得到的结果replace紧凑一点
        for valid_pos in valid_pos_list:
            valid_pos_var.append(np.var(np.asarray(valid_pos)))
        best_index = candidates_pos[np.argmin(valid_pos_var)]
        # best_path对应的分段数最少，且valid_pos之间的方差最小
        best_path = paths_distance[best_index]
        return best_path[::-1], min_part

    def get_valid_pos(self, path_distance):
        valid_pos = []
        part_pos = []
        index = 0
        # op type: replace, delete, insert
        while index < len(path_distance):
            ops, ops_index = path_distance[index][:2]
            if ops != 'delete':
                while index < len(path_distance) and ops != 'delete':
                    ops, ops_index = path_distance[index][:2]
                    part_pos.append(index)
                    index += 1
            if part_pos:
                valid_pos.append(part_pos)
                part_pos = []
                index -= 1
            index += 1
        return valid_pos


def filter_box(query_box, query_text, ocr_boxes, ocr_texts):
    # 框粗匹配，iou > 0.5
    mask_v2 = filter_boxes_v2(query_box, query_text, ocr_boxes, ocr_texts)
    candidates = []
    for index in np.where(mask_v2 > 0)[0]:
        ops = Levenshtein.editops(ocr_texts[index], query_text)
        # 文本精确匹配，只匹配一个ocr box
        if not len(ops):
            mask = np.zeros_like(mask_v2)
            mask[index] = 1
            return mask
        # ocr_text长度大于等于gt_text（没有insert）且两者字符有一定相关性，放到候选集合中
        if not insert_in_ops(ops, ocr_texts[index], query_text):
            candidates.append([index, len(ops)])
    mask = np.zeros_like(mask_v2)
    if len(candidates) > 0:
        # 从候选集合中选择最高相似的匹配，只匹配一个ocr box
        if len(candidates) > 1:
            candidates = sorted(candidates, key=lambda x: x[1])
        select_index = candidates[0][0]
        mask[select_index] = 1
    else:
        # 可能匹配上多个ocr box
        mask = mask_v2
        # print('query_text:', query_text)
        # for index in np.where(mask_v2 > 0)[0]:
        #     ops = Levenshtein.editops(ocr_texts[index], query_text)
        #     print(ops, insert_in_ops(ops, ocr_texts[index], query_text))
        #     print('ocr_texts:', index, ocr_texts[index])

    return mask


def filter_boxes_v2(query_box,
                    query_text,
                    ocr_boxes,
                    ocr_texts,
                    threshold=0.5):
    """filter boxes by threshold; add text validation.

    Args:
        query_box: List or numpy. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ocr_boxes: List or numpy. [[x1, y1], [x2, y2], [x3, y3], [x4, y4],
        [x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
        threshold: Float. iou threthod

    Returns:
        mask: numpy. [0, 0, 1, 0]
    """
    if len(ocr_boxes) == 0:
        return []
    mask = np.zeros((len(ocr_boxes), ), dtype=np.int8)
    query_box = Polygon(np.array(query_box).reshape(-1, 2))
    iou_matrix = np.zeros(len(ocr_boxes))
    for i, box in enumerate(ocr_boxes):
        poly1 = Polygon(np.array(box).reshape(-1, 2))
        if poly1.intersects(query_box):
            inter = poly1.intersection(query_box).area
            iou = inter / min([poly1.area, query_box.area])
        else:
            iou = 0
        iou_matrix[i] = iou

    binds = np.where(np.array(iou_matrix) > threshold)[0]
    max_bind = np.argmax(iou_matrix)
    for ind in binds:
        # if '**' in query_text or '##' in query_text:
        #     if max_bind in binds:
        #         mask[max_bind] = 1
        #     break

        if len(ocr_texts[ind]) == 0 or not is_worth(ocr_texts[ind], query_text,
                                                    max_bind == ind):
            continue
        mask[ind] = 1
    return mask


def insert_in_ops(ops, ocr_text, query_text):
    # if '##' in query_text or '**' in query_text:
    #     return False
    replace_ops_num = 0
    for op in ops:
        if op[0] == 'insert':
            return True
        elif op[0] == 'replace':
            replace_ops_num += 1
    if len(ocr_text) >= len(query_text):
        delete_num = len(ocr_text) - len(query_text)
        if replace_ops_num >= len(ocr_text) - delete_num:
            return True

    return False


def is_worth(ocr_text, query_text, is_max):
    if len(set(list(ocr_text)).intersection(set(list(query_text)))) == 0:
        if is_max and len(ocr_text) == len(query_text):
            return True
        else:
            return False
    else:
        return True


def replace_hashhash(x):
    # x : str of a text
    return x.replace('##', '卍')


def Levenshtein_mask(raw_text, gt_text):
    # we do not do padding in this function
    raw_text = replace_hashhash(raw_text)
    gt_text = replace_hashhash(gt_text)
    if not len(gt_text):
        ops, min_part = [['delete', i] for i in range(len(raw_text))], 0
    else:
        # 出发点：想尽可能分段数小且index更紧凑一点
        ops, min_part = EditDistance(raw_text, gt_text).editops()

    size_raw = len(raw_text)
    mask = size_raw * [0]
    # 统计头有多少字符是多余的
    last = -1
    for op in ops:
        if op[0] == 'delete' and op[1] == (last + 1):
            mask[op[1]] = 1
            last += 1
        else:
            break

    # 统计尾有多少字符是多余的
    last = size_raw
    for op in ops[::-1]:
        if op[0] == 'delete' and op[1] == (last - 1):
            mask[op[1]] = 1
            last -= 1
        else:
            break
    return mask


def hack_xy(text):
    def isdigit(x):
        for c in x:
            if not c.isdigit():
                return False
        return True

    if text is None:
        return 0, 0, None
    pure_text = None
    if len(text) > 6 and text.startswith('~$'):
        xy = text.split('~')[1][1:].split('.')
        if len(xy) == 2 and isdigit(xy[0]) and isdigit(xy[1]):
            x = int(xy[0])
            y = int(xy[1])
            symbol_length = len(f'~${x}.{y}~')
            pure_text = text[symbol_length:]

    if pure_text is None:
        x = 0
        y = 0
        pure_text = text
    return x, y, pure_text


def match_label_v1(gt_bbox, gt_text, ocr_bboxes, ocr_texts):
    """
    match label v1: one gt_bbox match one ocr_bbox
                    one gt_bbox match many ocr_bbox
    """
    candidates = np.zeros(len(ocr_bboxes))
    gt_bbox = np.array(gt_bbox)
    # delete ~$1.1~
    _, _, gt_text = hack_xy(gt_text)
    mask = filter_box(gt_bbox, gt_text, ocr_bboxes, ocr_texts)
    candidates += mask
    candidates_ids = np.where(candidates >= 1)[0]

    positions = []
    global_offset = 0
    for index, ocr_text in enumerate(ocr_texts):
        text_len = len(ocr_text)
        if text_len == 0:
            continue

        if index in candidates_ids:
            matched_ocr_text = ocr_texts[index]
            mask = Levenshtein_mask(matched_ocr_text, gt_text)
            for ind, delete in enumerate(mask):
                if not delete:
                    positions.append(global_offset + ind)

        global_offset += text_len

    offsets = []
    if not positions:
        return offsets
    spos = positions[0]
    for i in range(1, len(positions)):
        if positions[i] != positions[i - 1] + 1:
            offsets.append((spos, positions[i - 1] + 1))
            spos = positions[i]
    offsets.append((spos, positions[-1] + 1))

    if len(offsets) > 1:
        print('gt_text', gt_text)
        print('matched_ocr_text:', [ocr_texts[i] for i in candidates_ids])
        print(offsets)

    return offsets
