import copy
import glob
import json
import math
import os

import numpy as np


def dist_euclid(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# def parentheses_selector(data):
#     """
#     这里假设每个bbox的坐标是以左上角为起点且为顺时针顺序排列的。
#     :param data:
#     :return:
#     """
#     # 所有框的坐标
#     points_list = [i[1] for i in data]

#     # 计算每个框的宽高比
#     parentheses_list = list()
#     rates_info = []
#     for points in points_list:
#         p1, p2, p3, p4 = points
#         w = int(dist_euclid(p2, p1)) + 1
#         h = int(dist_euclid(p3, p2)) + 1
#         rates_info.append((max(h, w) / (min(h, w) + 1e-9), p1, w, h))

#     # 计算均值与标准差，设定一倍标准差为阈值
#     mean_rate = np.mean([i[0] for i in rates_info])
#     std_rate = np.std([i[0] for i in rates_info])
#     thr = mean_rate + 1 * std_rate
#     # thr = 3.0
#     # print(f'阈值：{thr}')

#     for idx, info in enumerate(rates_info):
#         rate, p1, w, h = info
#         # print(f'{data[idx]["value"]}:', rate)
#         if rate > thr:
#             parentheses_list.append((idx, p1))
#             if w > h:  # 横向括号
#                 x_y = 1  # 按左上角y坐标升序排列
#             else:  # 纵向括号
#                 x_y = 0  # 按左上角x坐标升序排列

#     if parentheses_list:
#         parentheses_list.sort(key=lambda x: x[1][x_y])

#         left = parentheses_list[0][0]  # 左括号的索引
#         right = parentheses_list[-1][0]  # 右括号的索引

#         right_parenthesis = data.pop(right)  # 右括号
#         left_parenthesis = data.pop(left)  # 左括号

#         return data, left_parenthesis, right_parenthesis, parentheses_list

#     else:
#         return data


def parentheses_selector(data):
    """
    这里假设每个bbox的坐标是以左上角为起点且为顺时针顺序排列的。
    :param data:
    :return:
    """
    # 所有框的坐标
    points_list = [i[1] for i in data]

    # 计算每个框的宽高比
    parentheses_list = list()
    rates_info = []
    for points in points_list:
        p1, p2, p3, p4 = points
        w = int(dist_euclid(p2, p1)) + 1
        h = int(dist_euclid(p3, p2)) + 1
        rates_info.append((max(h, w) / (min(h, w) + 1e-9), p1, w, h))

    sorted_h_index = sorted(enumerate(rates_info), key=lambda x: x[1][3])
    sorted_ratio_index = sorted(enumerate(rates_info), key=lambda x: x[1][0])

    # 括号的h一般最小，ratio最大
    mean_h = np.mean([rates_info[i[0]][-1]
                      for i in sorted_h_index[2:]])  # 剔除括号计算平均h
    mean_ratio = np.mean([
        rates_info[i[0]][0] for i in sorted_ratio_index[:-2]
    ])  # 剔除括号计算平均ratio
    thr_h = mean_h / 2
    # thr_ratio = mean_ratio * 2
    thr_ratio = mean_ratio + 0.9

    for idx, info in enumerate(rates_info):
        rate, p1, w, h = info
        # print(f'{data[idx][0]}:', rate, w, h, thr_h, thr_ratio)
        if h < thr_h or rate > thr_ratio:
            parentheses_list.append((idx, p1))

    left = -1
    right = -1
    if parentheses_list:
        parentheses_list.sort(key=lambda x: x[1][1])  # 按左上角y坐标升序排列

        left = parentheses_list[0][0]  # 左括号的索引
        # print('left:', data[left])
        if len(parentheses_list) > 1:
            right = parentheses_list[-1][0]  # 右括号的索引
            # print('right:', data[right])
    return left, right


def two_word_name(predicts):
    """
    适用情况：两个字，从左到右
            两个字，从上到下
    :param predicts:
    :return:
    """
    # [(字，左上角坐标),...]
    predicts_list = [(predict['value'], predict['points'][0])
                     for predict in predicts]

    # 两个字的左上角坐标
    x1y1, x2y2 = predicts_list[0][1], predicts_list[1][1]

    # 判断阅读方向
    direction = np.argmax(abs(np.array(x1y1) - np.array(x2y2)))

    # 如果两字相同则直接输出
    if predicts_list[0][0] == predicts_list[1][0]:
        return predicts_list[0][0] + predicts_list[1][0]

    # 纵向阅读，y坐标升序排列
    elif direction:
        sorted_predicts = sorted(predicts_list, key=lambda x: x[1][1])
        result = ''.join([i[0] for i in sorted_predicts])
        return result

    # 横向阅读，x坐标升序排列
    else:
        sorted_predicts = sorted(predicts_list, key=lambda x: x[1][0])
        result = ''.join([i[0] for i in sorted_predicts])
        return result


def three_word_name(predicts):
    """
    适用情况：三个字，一行，从左往右，横向阅读
            三个字，两列，从上到下, 从右往左
    :param predicts:
    :return:
    """
    # [(字，左上角坐标),...]
    predicts_list = [(i['value'], i['points'][0]) for i in predicts]

    # 先按左上角x坐标升序
    sorted_predicts = sorted(predicts_list, key=lambda x: x[1][0])

    # 取出三个字的左上角坐标
    coo = [i for val, xy in sorted_predicts for i in xy]

    # 计算sin值
    res = path_angle_degree(*coo)

    thr = 0.4
    # 小于阈值时，为横向阅读顺序，字符串顺序按x坐标由小到大的顺序排列
    if res[1] < thr:
        result = ''.join([i[0] for i in sorted_predicts])
        return result
    else:
        x_min = sorted_predicts[0][1][0]
        x_max = sorted_predicts[-1][1][0]
        mid = (x_min + x_max) // 2  # 取中值

        # 将左上角x坐标小于中值的放入左列表，其余放入右列表
        left, right = [], []
        for predict in sorted_predicts:

            if predict[1][0] < mid:
                left.append(predict)
            else:
                right.append(predict)

        # 左列表和有列表，字符分别按右上角y坐标降序排列
        sorted_y_right = sorted(right, key=lambda x: x[1][1])
        sorted_y_left = sorted(left, key=lambda x: x[1][1])
        right_str = ''.join([i[0] for i in sorted_y_right])
        left_str = ''.join([i[0] for i in sorted_y_left])

        return right_str + left_str


def four_word_name(predicts):
    """
    适用情况：四个字，两列，阅读顺序为从上到下，从右往左
    :param predicts:
    :return:
    """
    # [(字，左上角坐标),...]
    predicts_list = [(i['value'], i['points'][0]) for i in predicts]
    # 先按左上角x坐标升序
    sorted_predicts = sorted(predicts_list, key=lambda x: x[1][0])

    # 计算中值
    x_min = sorted_predicts[0][1][0]
    x_max = sorted_predicts[-1][1][0]
    mid = (x_min + x_max) // 2  # 取中值

    # 将左上角x坐标小于中值的放入左列表，其余放入右列表
    left, right = [], []
    for predict in sorted_predicts:
        if predict[1][0] < mid:
            left.append(predict)
        else:
            right.append(predict)

    # 左列表和有列表，字符分别按右上角y坐标降序排列
    sorted_y_right = sorted(right, key=lambda x: x[1][1])
    sorted_y_left = sorted(left, key=lambda x: x[1][1])
    right_str = ''.join([i[0] for i in sorted_y_right])
    left_str = ''.join([i[0] for i in sorted_y_left])

    # 输出从右到左拼接的字符串
    return right_str + left_str


def financial_seal(data, filter_para):
    """
    适用情况：财务专用章-横向-无括号：三行、四行、五行，从左往右，从上到下
            财务专用章-横向-有括号：三行、四行、五行，从左往右，从上到下
            财务专用章-纵向-无括号：三列、四列、五列，从上到下，从右往左
            财务专用章-纵向-有括号：三列、四列、五列，从上到下，从右往左
                                *含有此类括号会拼接不准的：（1）
    :param data:
    :return:
    """
    # [（字，四个点坐标）,....】
    data = [[i['value'], i['points']] for i in data]

    # 判断排列顺序
    direction = get_direction(data)

    if direction:  # 纵向
        if filter_para:
            # 先进行括号过滤，纵向括号的起始点会判断错误
            left, right = parentheses_selector(copy.deepcopy(data))
            if left != -1:
                data[left][0] = '('
            if right != -1:
                data[right][0] = ')'

        # 拿到头元素
        heads, sorted_data = get_heads(data, direction)

        # 按左上角x坐标升序排列头元素
        sorted_heads = sorted(heads, key=lambda x: x[1][0][0], reverse=False)

        # 划定区间
        results = split_data(sorted_data, sorted_heads, direction)

        # 拼接字符串
        output = ''.join([result[0] for result in results])

        return output

    else:
        # 拿到头元素
        heads, sorted_data = get_heads(data, direction)
        # 按左上角y坐标升序排列头元素
        sorted_heads = sorted(heads, key=lambda x: x[1][0][1], reverse=False)
        # print(sorted_heads)
        # 划定区间
        result = split_data(sorted_data, sorted_heads, direction)

        return ''.join([i[0] for i in result])


def get_direction(data):
    """
    :param data: 识别模型输出的结果
    :return:
    """
    tmp_list = ['财', '务', '专', '用', '章']
    vals = [item[0] for item in data]  # preds中的字按识别出的顺序存入该列表中
    data_ = [data[vals.index(i)] for i in tmp_list
             if i in vals]  # 按tmp_list的顺序将preds中对应的元素及坐标存入该列表中

    idx_1, idx_2 = (len(data_) // 2) - 1, len(data_) // 2
    x1y1, x2y2 = data_[idx_1][1][0], data_[idx_2][1][0]

    # 方位，0为横向，1为纵向
    direct = np.argmax(abs(np.array(x1y1) - np.array(x2y2)))
    # print(direct)

    return direct


def get_heads(data, direction):
    heads = []

    if direction:
        # 所有元素按左上角y坐标升序排列
        sorted_data = sorted(data, key=lambda x: x[1][0][1])
        # print(sorted_data)
        y_min_str = sorted_data[0]  # 最小左上角y坐标对应的元素
        heads.append(y_min_str)

        # 拿到每一列第一个元素
        for idx in range(1, len(sorted_data)):
            x1y1x2y2 = y_min_str[1][0] + sorted_data[idx][1][0]
            res = cal_tan(*x1y1x2y2)  # 得到tan值的绝对值

            # 最小y坐标元素与其他元素左上角坐标连线的斜率小于tan(pi/18)的作为每一列第一个元素
            if res < math.tan(math.pi / 18):
                heads.append(sorted_data[idx])
    else:
        # 按x坐标升序排列
        sorted_data = sorted(data, key=lambda x: x[1][0][0])
        x_min_str = sorted_data[0]  # 最小x坐标对应的元素
        heads.append(x_min_str)

        #
        for idx in range(1, len(sorted_data)):
            x1y1x2y2 = x_min_str[1][0] + sorted_data[idx][1][0]
            res = cal_tan(*x1y1x2y2)  # 得到tan值的绝对值

            # 最小y坐标元素与其他元素左上角坐标连线的斜率小于tan(pi/18)的作为每一列第一个元素
            # 拿到每一行的头元素
            if res > math.tan(math.pi * 80 / 180):
                heads.append(sorted_data[idx])

    return heads, sorted_data


def split_data(data, heads, direction):
    # 将元素按头元素的个数切分到不同的集合
    tmp = []

    # 纵向阅读顺序
    if direction:
        x_y = 0  # 取右上角x坐标
    else:
        x_y = 1  # 取右上角y坐标

    # 取头元素的左上角x or y坐标
    for i in range(len(heads)):
        tmp.append(heads[i][1][0][x_y])

    # 计算区分点
    sp = []
    for j in range(len(tmp) - 1):
        sp.append((tmp[j] + tmp[j + 1]) / 2)

    line_points = []
    pre = []
    for index, p in enumerate(sp):
        cur = [item for item in data if item[1][0][x_y] < p]
        tmp = [j for j in cur if j not in pre]
        sorted_tmp = sorted(tmp, key=lambda x: x[1][0][abs(x_y - 1)])
        line_points.append(sorted_tmp)
        pre = cur

    tmp = [item for item in data if item[1][0][x_y] > sp[-1]]
    sorted_tmp = sorted(tmp, key=lambda x: x[1][0][abs(x_y - 1)])
    line_points.append(sorted_tmp)

    if direction:
        line_points = line_points[::-1]

    # print(line_points)
    extend_points = line_points[0]
    for index in range(1, len(line_points)):
        extend_points.extend(line_points[index])

    return extend_points


def cal_tan(x1, y1, x2, y2):
    y = abs(y1 - y2)
    x = abs(x1 - x2)

    return round(y / (x + 1e-9), 3)


def path_angle_degree(x1, y1, x2, y2, x3, y3):
    u = (x2 - x1, y2 - y1)
    v = (x3 - x2, y3 - y2)
    norm_u = math.sqrt(u[0] * u[0] + u[1] * u[1])
    norm_v = math.sqrt(v[0] * v[0] + v[1] * v[1])

    # this conditional is to check there has been movement between the points
    if norm_u < 0.001 or norm_v < 0.001:
        return (None, None, None)
    prod_n = norm_u * norm_v
    dot_uv = u[0] * v[0] + u[1] * v[1]
    cos_uv = dot_uv / prod_n

    # fixes floating point rounding
    if cos_uv > 1.0 or cos_uv < -1.0:
        cos_uv = round(cos_uv)
    radians = math.acos(cos_uv)
    sin_uv = math.sin(radians)
    degree = math.degrees(radians)
    return (cos_uv, sin_uv, degree)


def seal_postprocess(predicts, seal_label='', filter_para=True):
    """
    :param predicts:
    :param seal_label: 'name' or 'finance' or 'others' or ''
    :param filter_para: 是否需要过滤括号
    :return:
    """
    tmp_list = ['财', '务', '专', '用', '章']
    if not seal_label:
        if 2 <= len(predicts) <= 4:
            # 小于等于4个字的章，判断为人名章；不支持4个字以上的人名章
            seal_label = 'name'
        else:
            text_values = [pred['value'] for pred in predicts]
            match = 0
            for char in tmp_list:
                if char in text_values:
                    match += 1
            if match >= 2:
                seal_label = 'finance'
            else:
                seal_label = 'others'

    if seal_label == 'others':
        # 直接返回未拼接结果
        return [pred['value'] for pred in predicts]
    elif seal_label == 'name':
        if len(predicts) == 2:
            result = two_word_name(predicts)
            return [result]
        elif len(predicts) == 3:
            result = three_word_name(predicts)
            return [result]
        elif len(predicts) == 4:
            result = four_word_name(predicts)
            return [result]
        else:
            # 不支持4个字以上的人名章，直接返回未拼接结果
            return [pred['value'] for pred in predicts]
    else:
        if len(predicts) >= 5:
            result = financial_seal(predicts, filter_para)
            return [result]
        else:
            # 不支持5个字以下的财务专用章，直接返回未拼接结果
            return [pred['value'] for pred in predicts]


def ans(json_result):
    a = str()
    for i in json_result:
        a += i['value']
    return a


if __name__ == '__main__':
    people_four = '/Users/gulixin/Downloads/财务章-有括号/财务章-横向-有括号'
    files = glob.glob(os.path.join(people_four, '*.json'))

    for file in files:
        with open(file, 'r') as f:
            json_result = json.load(f)
            print(file)

        answer = ans(json_result)
        predict = seal_postprocess(json_result,
                                   seal_label='name',
                                   filter_para=True)
        print(answer, predict)
