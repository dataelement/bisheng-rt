import functools
import math

import cv2
import numpy as np
from scipy import integrate, optimize


def mean(nums):
    return float(sum(nums)) / float(len(nums))


# 寻找圆心和半径R、r
def find_center_of_circle(points):
    # Coordinates of the 2D points
    x = points[::2]
    y = points[1::2]
    # 质心坐标
    x_m = mean(x)
    y_m = mean(y)

    # 修饰器：用于输出反馈
    def countcalls(fn):
        'decorator function count function calls '

        @functools.wraps(fn)
        def wrapped(*args):
            wrapped.ncalls += 1
            return fn(*args)

        wrapped.ncalls = 0
        return wrapped

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    @countcalls
    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # 圆心估计
    center_estimate = x_m, y_m
    center_2, _ = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(xc_2, yc_2)
    R = Ri_2.max()
    r = Ri_2.min()
    # 拟合圆的半径
    R_2 = Ri_2.mean()
    return xc_2, yc_2, R_2, R, r


def find_angle(x, y, xc, yc):
    det_x = x - xc
    det_y = y - yc
    hypotenuse = math.sqrt(det_x**2 + det_y**2)
    sin_theta = det_y / hypotenuse
    theta = math.asin(sin_theta)
    if det_x >= 0 and det_y >= 0:
        return theta
    elif det_x <= 0 and det_y >= 0:
        return math.pi - theta
    elif det_x <= 0 and det_y <= 0:
        return math.pi - theta
    elif det_x >= 0 and det_y <= 0:
        return theta + 2 * math.pi


def diff_radian(theta1, theta2, max_radiant):
    thetas = []
    if 0 <= theta1 <= 90 and 270 <= theta2 <= 360:
        temp_theta1 = theta1 + 360
        temp_theta = theta2 + max_radiant
        while temp_theta < temp_theta1:
            thetas.append(temp_theta % 360)
            temp_theta += max_radiant
    elif 270 <= theta1 <= 360 and 0 <= theta2 <= 90:
        temp_theta = theta2 + 360 - max_radiant
        while temp_theta > theta1:
            thetas.append(temp_theta % 360)
            temp_theta -= max_radiant
    elif theta1 < theta2:
        temp_theta = theta2 - max_radiant
        while temp_theta > theta1:
            thetas.append(temp_theta % 360)
            temp_theta -= max_radiant
    elif theta1 > theta2:
        temp_theta = theta2 + max_radiant
        while temp_theta < theta1:
            thetas.append(temp_theta % 360)
            temp_theta += max_radiant
    return thetas


def find_alpha(points, xc, yc, text_angle=0, max_radiant=2):
    """
    寻找曲线轮廓的起点和终点的弧度
    """
    success_find = True
    n = len(points)
    thetas = []
    maxTheta = -1
    for i in range(0, n, 2):
        point_x = points[i]
        point_y = points[i + 1]
        theta = 180 * find_angle(point_x, point_y, xc, yc) / math.pi
        if i > 0:
            det_theta = min(abs(theta - thetas[-1]),
                            360 - abs(theta - thetas[-1]))
            if det_theta > max_radiant:
                thetas += diff_radian(theta, thetas[-1], max_radiant)
        thetas.append(theta)
    for idx in range(1, len(thetas)):
        maxTheta = max(
            maxTheta,
            min(abs(thetas[idx] - thetas[idx - 1]),
                360 - abs(thetas[idx] - thetas[idx - 1])))
    if maxTheta < 2:
        maxTheta = 2

    # 寻找曲形文本角度端点
    alpha1 = thetas[0]
    angles = set([int(alpha1)])
    while True:
        flag = False
        for theta in thetas:
            if min(abs(theta - alpha1),
                   360 - abs(theta - alpha1)) <= maxTheta / 2:
                flag = True
                break
        if not flag:
            # 找到了曲线文本端点
            break
        alpha1 = int((alpha1 + maxTheta / 2) % 360)
        if alpha1 in angles:
            # 没有找到曲线文本端点
            success_find = False
            break
        angles.add(alpha1)

    # 寻找曲形文本角度另一个端点
    alpha2 = thetas[0]
    angles = set([alpha2])
    while True:
        flag = False
        for theta in thetas:
            if min(abs(theta - alpha2),
                   360 - abs(theta - alpha2)) <= maxTheta / 2:
                flag = True
                break
        if not flag:
            # 找到了曲线文本端点
            break
        alpha2 = int((alpha2 - maxTheta / 2) % 360)
        if alpha2 in angles:
            # 没有找到曲线文本端点
            success_find = False
            break
        angles.add(alpha2)

    # print("**************maxTheta alpha1 alpha2****************")
    # print(maxTheta, alpha1, alpha2)

    # 确定起点和终点
    alpha1_radian = math.pi * alpha1 / 180
    alpha2_radian = math.pi * alpha2 / 180
    alpha_mid_radian = find_angle(
        math.cos(alpha1_radian) + math.cos(alpha2_radian),
        math.sin(alpha1_radian) + math.sin(alpha2_radian), 0, 0)
    alpha_mid = alpha_mid_radian / math.pi * 180
    flag = False
    for theta in thetas:
        if min(abs(theta - alpha_mid),
               360 - abs(theta - alpha_mid)) <= maxTheta / 2:
            flag = True
            break
    delta = min(abs(alpha1 - alpha2), 360 - abs(alpha1 - alpha2))
    flip = False
    if flag:
        alpha1 = (alpha_mid + delta / 2) % 360
        alpha2 = (alpha_mid - delta / 2) % 360
        delta1 = min(abs(text_angle - (alpha_mid - 90) % 360),
                     360 - abs(text_angle - (alpha_mid - 90) % 360))
        delta2 = min(abs(text_angle - (alpha_mid + 90) % 360),
                     360 - abs(text_angle - (alpha_mid + 90) % 360))
        if delta2 < delta1:
            flip = True
    else:
        alpha1 = (alpha_mid - delta / 2) % 360
        alpha2 = (alpha_mid + delta / 2) % 360
        delta = 360 - delta
        delta1 = min(abs(text_angle - (alpha_mid - 90) % 360),
                     360 - abs(text_angle - (alpha_mid - 90) % 360))
        delta2 = min(abs(text_angle - (alpha_mid + 90) % 360),
                     360 - abs(text_angle - (alpha_mid + 90) % 360))
        if delta1 < delta2:
            flip = True

    # print("************start and end angle ******************")
    # print(flag, alpha1, alpha2, delta)

    alpha1_radian = math.pi * alpha1 / 180
    alpha2_radian = math.pi * alpha2 / 180
    delta_radian = math.pi * delta / 180

    return alpha1_radian, alpha2_radian, delta_radian, success_find, flip


def curve2rect_round(img_ori, alpha1, alpha2, delta, R, r, xc, yc, flip):
    """
    根据圆极坐标转换，拉正曲线文本
    """
    img = np.zeros((int(R - r) + 2, int(R * delta) + 2, 3),
                   dtype=img_ori.dtype)
    h, w, _ = img_ori.shape
    x = np.asarray(range(int(R * delta) + 2))
    y = np.asarray(range(int(R - r) + 2))
    crop_w = len(x)
    crop_h = len(y)

    x = x.reshape((1, crop_w)).repeat(crop_h, axis=0)
    y = y.reshape((crop_h, 1)).repeat(crop_w, axis=1)

    x_ori = xc + (r + y) * np.cos(alpha1 - x / R)
    y_ori = yc + (r + y) * np.sin(alpha1 - x / R)

    x_ori = np.round(x_ori).astype(np.int)
    y_ori = np.round(y_ori).astype(np.int)
    x_ori = np.clip(x_ori, 0, w - 1)
    y_ori = np.clip(y_ori, 0, h - 1)
    img[y, x, :] = img_ori[y_ori, x_ori, :]

    if flip:
        img = cv2.flip(img, -1)
    return img


def find_normal_direction(a, b, x0, y0, eps=10e-12):
    """
    计算椭圆上点 (x0, y0) 对应的法线斜率
    """
    return ((a**2) * y0) / ((b**2) * x0 + eps)


def from_radian_find_point_on_ellipse(a, b, alpha, xc, yc):
    """
    计算弧度为alpha的直线与椭圆圆弧的交点
    """
    denominator = math.sqrt(a**2 * math.sin(alpha)**2 +
                            b**2 * math.cos(alpha)**2)
    y = a * b * math.sin(alpha) / denominator
    x = a * b * math.cos(alpha) / denominator
    s = [x, y]
    s1 = [-x, -y]
    if s1 != s:
        s1_alpha = find_angle(s1[0], s1[1], xc, yc)
        s_alpha = find_angle(s[0], s[1], xc, yc)
        if min(abs(s1_alpha - alpha), 2 * math.pi - abs(s1_alpha - alpha)) < \
                min(abs(s_alpha - alpha), 2 * math.pi - abs(s_alpha - alpha)):
            s = s1
    return s


def find_h_form_intersection_of_normal_conts(s, normal_direction, conts):
    """
    根据法线和轮廓的交点，计算轮廓的宽度，轮廓上的点到直线的距离
    """
    # 直线方程
    a = normal_direction
    b = -1
    c = s[1] - normal_direction * s[0]

    denominator = math.sqrt(a**2 + b**2)
    dist = abs(a * conts[0] + b * conts[1] + c) / denominator
    idxs = np.argwhere(dist <= 1).flatten()
    h = 0
    pts = np.array([conts[0][idxs], conts[1][idxs]])
    for i in range(len(pts[0]) - 1):
        for j in range(i + 1, len(pts[1])):
            h = max(
                h,
                math.sqrt((pts[0][i] - pts[0][j])**2 +
                          (pts[1][i] - pts[1][j])**2))
    return h


def from_h_find_pt(det_h, normal_direction, s, xc, yc):
    """
    计算与点 s 距离 det_h 的、靠近椭圆圆心 (xc, yc) 的点
    """
    s1 = [
        s[:, 0] + det_h / np.sqrt(normal_direction**2 + 1),
        s[:, 1] + normal_direction * det_h / np.sqrt(normal_direction**2 + 1)
    ]
    s2 = [
        s[:, 0] - det_h / np.sqrt(normal_direction**2 + 1),
        s[:, 1] - normal_direction * det_h / np.sqrt(normal_direction**2 + 1)
    ]
    dist1 = (yc - s1[1])**2 + (xc - s1[0])**2
    dist2 = (yc - s2[1])**2 + (xc - s2[0])**2

    return np.where(dist1 < dist2, s1, s2)


def find_outer_points(img,
                      bbox,
                      start,
                      end,
                      delta_of_circle,
                      circle,
                      debug=False):
    """
    计算轮廓点中的外侧点集合
    """
    x_c, y_c, r_2, R, r = circle
    outer_points = []
    for i in range(0, len(bbox), 2):
        x = int(bbox[i])
        y = int(bbox[i + 1])
        angle = find_angle(x, y, x_c, y_c)
        det_theta = min(
            min(abs(angle - start), math.pi * 2 - abs(angle - start)),
            min(abs(angle - end), math.pi * 2 - abs(angle - end)))
        if (x - x_c)**2 + (
                y - y_c)**2 > r_2**2 and det_theta > delta_of_circle / 25.0:
            # 保留圆外的点，去掉圆内的点和头尾各1/25点
            outer_points.append([x, y])
    if debug:
        img_copy = img.copy()
        # print(np.array(outer_points).astype("int").T)
        cv2.drawContours(img_copy, [np.array(outer_points).astype('int')], -1,
                         (0, 255, 0), 3)
        cv2.imwrite('find_outer_points.png', img_copy)
        print(np.array(outer_points).T)
    return np.array(outer_points)


def fit_ellipse_of_outer_points(img, bbox, debug=False):
    """
    根据外侧的轮廓点拟合椭圆
    """
    circle = find_center_of_circle(bbox)
    x_c, y_c, r_2, R, r = circle
    alpha1, alpha2, delta, success_find, _ = find_alpha(bbox, x_c, y_c)
    conts = find_outer_points(img, bbox, alpha1, alpha2, delta, circle, debug)
    ellipse = cv2.fitEllipseAMS(conts)
    if debug:
        print(ellipse)
        img_copy = img.copy()
        cv2.ellipse(img_copy, ellipse, (0, 255, 0), 3)
        cv2.drawContours(img_copy, [np.array(conts).astype('int')], -1,
                         (0, 255, 0), 3)
        cv2.imwrite('ellipse_fit.png', img_copy)
    return ellipse[0][0], ellipse[0][
        1], ellipse[1][0] // 2, ellipse[1][1] // 2, ellipse[2]


def affine_transformation_of_ellipse(theta, bbox, xc, yc):
    """
    坐标系仿射变换，将椭圆方程变为标准方程（以椭圆圆心为原点）
    """
    # 先旋转theta，再平移(xc, yc)
    at_matrix_ori = np.array([[math.cos(theta), -math.sin(theta), xc],
                              [math.sin(theta),
                               math.cos(theta), yc], [0, 0, 1]])
    # 先平移(-xc, -yc)，再旋转-theta
    at_matrix = np.linalg.inv(at_matrix_ori)
    conts = bbox.reshape(-1, 2).T
    r, c = conts.shape
    # 将轮廓点变换到对应坐标轴上
    conts = np.matmul(at_matrix, np.r_[conts, np.ones((1, c))])[0:2, :]
    bbox = np.zeros((2 * c))
    bbox[0:2 * c:2] = conts[0, :]
    bbox[1:2 * c:2] = conts[1, :]
    return at_matrix_ori, conts, bbox


def find_circumference_of_arc(alpha1, delta, ma, sma):
    """
    计算弧长
    """
    def f(x):
        return math.sqrt(ma**2 * math.cos(x)**2 + sma**2 * math.sin(x)**2)

    return integrate.quad(f, alpha1 - delta, alpha1)[0]


def find_h(ma, sma, alpha, conts):
    """
    计算高度
    """
    s = from_radian_find_point_on_ellipse(ma, sma, alpha, 0, 0)
    normal_direction = find_normal_direction(ma, sma, s[0], s[1])
    h = find_h_form_intersection_of_normal_conts(s, normal_direction, conts)
    return h


def find_points_on_ellipse(alphas, ma, sma, delta, circumference_of_arc):
    """
    求椭圆对应于 alphas 的点，并记录每个点到终点的弧长
    """
    points = []
    x_of_rects = []
    for i, alpha in enumerate(alphas):
        s = from_radian_find_point_on_ellipse(ma, sma, alpha, 0, 0)
        if i > 0:

            def f(x):
                return math.sqrt(ma**2 * math.cos(x)**2 +
                                 sma**2 * math.sin(x)**2)

            circumference_of_arc_temp = x_of_rects[-1] + \
                integrate.quad(
                    f, alpha - delta / int(circumference_of_arc * 2), alpha)[0]
            points.append(s)
            x_of_rects.append(circumference_of_arc_temp)
        else:
            points.append(s)
            x_of_rects.append(0)
    return np.asarray(points), np.asarray(x_of_rects)


def ellipse2rect(img, h, circumference_of_arc, points, x_of_rects, ma, sma,
                 at_matrix_ori, flip):
    """
    将弯曲轮廓拉直
    """
    img_h, img_w, _ = img.shape
    h = h + 4  # 里面轮廓向里扩4个像素
    rect_img_h = int(h) + 1 + 2  # 外面轮廓向外扩2个像素
    rect_img_w = int(circumference_of_arc) + 2
    rect_img = np.zeros((rect_img_h, rect_img_w, 3), dtype=img.dtype)
    rect_img = rect_img.transpose(1, 0, 2)
    # print(rect_img.shape)
    # print(img.shape)
    points_num = len(points)
    normal_direction = find_normal_direction(ma, sma, points[:, 0], points[:,
                                                                           1])
    # pt_h shape is (n, 2)
    pt_h = from_h_find_pt(int(h), normal_direction, points, 0,
                          0).transpose(1, 0)
    # h_vector shape is (rect_img_h, 2)
    h_vector = np.arange(rect_img_h).reshape(-1, 1)  # (0, 1, 2, ..., h)
    h_vector = np.tile(h_vector, (1, 2))
    # det_h shape is (n, rect_img_h, 2)
    det_h = np.tile(((points - pt_h) / h).reshape(points_num, 1, 2),
                    (rect_img_h, 1))
    # pts shape is (n, rect_img_h, 2)
    pts = np.tile(pt_h.reshape(points_num, 1, 2),
                  (rect_img_h, 1)) + det_h * h_vector
    # pts shape is (n, rect_img_h, 3, 1)
    pts = np.concatenate((pts, np.tile([1], (points_num, rect_img_h, 1))),
                         axis=2)[:, :, :, np.newaxis]
    # pts shape is (n, rect_img_h, 2)
    pts = np.round(np.matmul(at_matrix_ori, pts)[:, :, 0:2, 0]).astype('int')
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, img_w - 1)  # (n, rect_img_h)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, img_h - 1)  # (n, rect_img_h)

    x_of_rects, indices = np.unique(np.round(x_of_rects).astype('int'),
                                    return_index=True)
    rect_img[x_of_rects, :, :] = img[pts[indices, :, 1], pts[indices, :, 0], :]

    rect_img = rect_img.transpose(1, 0, 2)
    if flip:
        rect_img = cv2.flip(rect_img, -1)
    return rect_img


def curve2rect_ellipse(img, bbox, ellipse, bbox_angle, debug=False):
    """
    计算弯曲文本对映的矩形文本
    """
    # step1: 对轮廓外侧点拟合椭圆
    xc, yc, sma, ma, theta = ellipse
    img_with_ellipse = None
    if debug:
        img_with_ellipse = img.copy()
        cv2.ellipse(img_with_ellipse, (int(xc), int(yc)), (int(sma), int(ma)),
                    theta, 0, 360, (0, 250, 0), 3)

    # step2: 坐标系仿射变换，将椭圆方程变为标准方程（以椭圆圆心为原点, 先平移再旋转）
    rotate_theta = math.pi / 2 - theta / 180 * math.pi  # theta为短轴与x轴夹角
    at_matrix_ori, conts, bbox = affine_transformation_of_ellipse(
        -rotate_theta, bbox, xc, yc)
    if debug:
        img2 = img.copy()
        conts_copy = conts.copy()
        conts_copy[0, :] = conts_copy[0, :] + int(xc)
        conts_copy[1, :] = conts_copy[1, :] + int(yc)
        cv2.drawContours(img2, [np.array(conts_copy).astype('int').T], -1,
                         (0, 255, 0), 3)
        cv2.imwrite('img_rotate.png', img2)

    # step3: 找轮廓的终止角度，和轮廓弧度
    bbox_angle = bbox_angle + rotate_theta / math.pi * 180
    alpha1, alpha2, delta, success_find, flip = find_alpha(
        bbox, 0, 0, bbox_angle)
    # print(alpha1, delta, success_find)
    if not success_find:
        return None, None, False

    # step4: 求弧长
    circumference_of_arc = find_circumference_of_arc(alpha1, delta, ma, sma)
    alphas = [
        (alpha1 - delta * i / int(circumference_of_arc * 2)) % (2 * math.pi)
        for i in range(0,
                       int(circumference_of_arc * 2) + 1)
    ]

    # step5: 求轮廓宽度
    h = find_h(ma, sma, alphas[int(circumference_of_arc * 2) // 2 + 1], conts)
    if h == 0:
        return None, None, False

    # step6: 求椭圆对应于alphas的点
    points, x_of_rects = find_points_on_ellipse(alphas, ma, sma, delta,
                                                circumference_of_arc)

    # step7: 将弯曲轮廓拉直
    rect_img = ellipse2rect(img, h, circumference_of_arc, points, x_of_rects,
                            ma, sma, at_matrix_ori, flip)
    return rect_img, img_with_ellipse, success_find
