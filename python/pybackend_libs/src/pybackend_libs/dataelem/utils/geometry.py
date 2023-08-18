import copy
import math

import cv2
import numpy as np
from shapely.geometry import Polygon


def dist_euclid(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def crop(img, boxes):
    # img is the original image canvas
    # boxes is with dim [N, 8]
    boxes = np.array(boxes).reshape(-1, 4, 2)
    ans = []
    for box in boxes:
        # img0 = perspective_transform(img, box)
        # from v2.0.1, we use affine transform instead
        img0 = affine_transform(img, box)
        ans.append(img0)
    return ans


def affine_transform(img, pts):
    W = np.round(dist_euclid(pts[0], pts[1]))
    H = np.round(dist_euclid(pts[1], pts[2]))
    src_3points = np.float32([pts[0], pts[1], pts[2]])
    dest_3points = np.float32([[0, 0], [W, 0], [W, H]])
    M = cv2.getAffineTransform(src_3points, dest_3points)
    m = cv2.warpAffine(img, M, (int(W), int(H)))
    return m


def perspective_transform(img, pts):
    W = int(dist_euclid(pts[0], pts[1])) + 1
    H = int(dist_euclid(pts[1], pts[2])) + 1

    pts = np.array(pts, 'float32')
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], 'float32')
    M0 = cv2.getPerspectiveTransform(pts, dst)
    image = cv2.warpPerspective(img, M0, (W, H))
    return image


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
    return np.array([tl, tr, br, bl])


def rotate_image_only(im, angle):
    '''
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    '''
    def rotate(src, angle, scale=1.0):
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


def rotate_polys_only(old_center, new_center, polys, angle):
    '''
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    '''
    polys_copy = copy.deepcopy(polys)
    angle = angle * np.pi * 1.0 / 180
    new_polys = []
    for poly in polys_copy:
        # print('poly:', poly)
        poly[:, 0] = poly[:, 0] - new_center[0]
        poly[:, 1] = new_center[1] - poly[:, 1]
        x1 = poly[0, 0] * math.cos(angle) - poly[0, 1] * math.sin(
            angle) + old_center[0]
        y1 = old_center[1] - (poly[0, 0] * math.sin(angle) +
                              poly[0, 1] * math.cos(angle))
        x2 = poly[1, 0] * math.cos(angle) - poly[1, 1] * math.sin(
            angle) + old_center[0]
        y2 = old_center[1] - (poly[1, 0] * math.sin(angle) +
                              poly[1, 1] * math.cos(angle))
        x3 = poly[2, 0] * math.cos(angle) - poly[2, 1] * math.sin(
            angle) + old_center[0]
        y3 = old_center[1] - (poly[2, 0] * math.sin(angle) +
                              poly[2, 1] * math.cos(angle))
        x4 = poly[3, 0] * math.cos(angle) - poly[3, 1] * math.sin(
            angle) + old_center[0]
        y4 = old_center[1] - (poly[3, 0] * math.sin(angle) +
                              poly[3, 1] * math.cos(angle))
        new_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return np.array(new_polys, dtype=np.float32)


def rotate_image(im, polys, angle):
    '''
    rotate image in range[-20,20]
    :param polys:
    :param tags:
    :return:
    '''
    def rotate(src, angle, scale=1.0):
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

    angle = angle * np.pi * 1.0 / 180
    new_polys = []
    for poly in polys:
        # print('poly:', poly)
        poly = np.asarray(poly)
        poly[:, 0] = poly[:, 0] - old_center[0]
        poly[:, 1] = old_center[1] - poly[:, 1]

        poly_x = poly[:, 0] * math.cos(angle) - poly[:, 1] * math.sin(
            angle) + new_center[0]
        poly_y = new_center[1] - (poly[:, 0] * math.sin(angle) +
                                  poly[:, 1] * math.cos(angle))
        poly_x = np.expand_dims(poly_x, 1)
        poly_y = np.expand_dims(poly_y, 1)
        new_polys.append(np.concatenate((poly_x, poly_y), axis=1))

    return image, new_polys


def compute_angle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted
    # clockwise First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis,
        # then p0 must be the upper-left corner
        return 0
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        # p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1]) /
                          (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
            return angle
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            return -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            return angle


def intersection(g, p):
    """
    compute iou
    """
    g = Polygon(g)
    p = Polygon(p)
    if not g.is_valid or not p.is_valid:
        return 0, 0
    inter = Polygon(g).intersection(Polygon(p)).area
    p_in_g = inter / p.area
    union = g.area + p.area - inter
    if union == 0:
        return 0, 0
    else:
        return inter / union, p_in_g
