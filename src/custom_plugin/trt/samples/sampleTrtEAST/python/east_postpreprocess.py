import os
import east_lanms
import numpy as np
import math
import cv2

def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]

def load_feat(src_dir, name):
    bin_name = src_dir + '/bin/' + name
    shape_name = src_dir + '/shape/' + name
    shape = np.fromfile(shape_name,dtype=np.float32)
    s = shape[:4].astype(np.int32) if len(shape) == 6 else [1] + list(shape[:3].astype(np.int32))
    feat = np.fromfile(bin_name,dtype=np.float32).reshape(s)
    return feat, shape[-2], shape[-1]

def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    positive_samples = np.where(angle >= 0)[0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    negtive_samples = np.where(angle < 0)[0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1]), np.concatenate([positive_samples, negtive_samples])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:

        return p[[0, 3, 2, 1]]


def detect(score_map, geo_map, cos_map, sin_map, score_map_thresh=0.5, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
        cos_map = cos_map[0, :, : , 0]
        sin_map = sin_map[0, :, : , 0]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    idex = np.lexsort([xy_text[:,1], xy_text[:,0]])
    xy_text = xy_text[idex, :]

    cos_map = np.where(score_map > score_map_thresh, 2 * cos_map - 1, -2 * np.ones_like(cos_map))
    sin_map = np.where(score_map > score_map_thresh, 2 * sin_map - 1, -2 * np.ones_like(sin_map))

    # restore
    text_box_restored, permutation_index = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    xy_text = xy_text[permutation_index]

    centerness = compute_centerness_targets(xy_text, geo_map)
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]] * centerness

    all_boxes = boxes.copy()
    # nms part
    boxes = east_lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # 计算box的起始点
    start_points_indexs = []
    if boxes.shape[0] == 0:
        return None, start_points_indexs

    for i, box in enumerate(boxes):
        mask = np.zeros_like(cos_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        yx_text = np.argwhere((mask == 1) & (cos_map > -2))
        cos_value = np.mean(cos_map[yx_text[:, 0], yx_text[:, 1]])
        yx_text = np.argwhere((mask == 1) & (sin_map > -2))
        sin_value = np.mean(sin_map[yx_text[:, 0], yx_text[:, 1]])
        # print('cos_value:', cos_value)
        # print('sin_value:', sin_value)

        cos_value = cos_value / math.sqrt(math.pow(cos_value, 2) + math.pow(sin_value, 2))
        sin_value = sin_value / math.sqrt(math.pow(cos_value, 2) + math.pow(sin_value, 2))
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

        # print('cos_angle:', cos_angle)
        # print('sin_angle:', sin_angle)
        # print('angle:', angle)
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
        box_angle = np.array([box_angle, (box_angle + 90) % 360, (box_angle + 180) % 360, (box_angle + 270) % 360])

        # print('box_angle:', box_angle)
        delta_angle = np.append(np.abs(box_angle - angle), 360 - np.abs(box_angle - angle))
        start_points_indexs.append(np.argmin(delta_angle) % 4)

    # refine box
    for i, box in enumerate(boxes):
        boxes[i] = refine_box(box, xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1]], all_boxes)

    return boxes, start_points_indexs

def compute_sideness_targets(reg_targets,lr_or_tb):
    left_right = reg_targets[:, [3, 1]]
    top_bottom = reg_targets[:, [0, 2]]
    if lr_or_tb == 'lr':
        centerness = [(left_right[:,1] / left_right.sum(axis=-1)) * \
                      (top_bottom.min(axis=-1) / top_bottom.max(axis=-1)),
                      (left_right[:,0] / left_right.sum(axis=-1)) * \
                      (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))]
    else:
        centerness = [(left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                      (top_bottom[:,1] / top_bottom.sum(axis=-1)),
                      (left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                      (top_bottom[:,0] / top_bottom.sum(axis=-1))]
    return np.sqrt(centerness)

def refine_box(box, xy_text, geometry, boxes):
    tbox = box[:8].reshape((4,2))

    if max(np.linalg.norm(tbox[1]-tbox[0]),np.linalg.norm(tbox[2]-tbox[1]))<500:return box

    inside_idex = point_inside_of_box(xy_text,tbox)
    # print('inside_idex:',len(inside_idex))
    left_right = 'lr' if np.linalg.norm(tbox[1]-tbox[0])>np.linalg.norm(tbox[2]-tbox[1]) else 'tb'
    sideness = compute_sideness_targets(geometry[inside_idex,:4],left_right)

    #refine box
    idex = np.argmax(sideness,axis=1)
    if left_right =='lr':
        left_box = boxes[inside_idex[idex[0]]]
        right_box = boxes[inside_idex[idex[1]]]
        if check_angle_length(left_box[6:8]-left_box[:2],box[6:8]-box[:2]):
            box[:2] = left_box[:2]
            box[6:8] = left_box[6:8]
        if check_angle_length(right_box[4:6]-right_box[2:4],box[4:6]-box[2:4]):
            box[2:6] = right_box[2:6]

    else:
        top_box = boxes[inside_idex[idex[0]]]
        bottom_box = boxes[inside_idex[idex[1]]]
        if check_angle_length(top_box[2:4]-top_box[:2],box[2:4]-box[:2]):
            box[:4] = top_box[:4]
        if check_angle_length(bottom_box[6:8]-bottom_box[4:6],box[6:8]-box[4:6]):
            box[4:8] = bottom_box[4:8]
    return box


def merge_side(side_box,check):
    return np.mean(side_box[check],axis=-2)


def check_angle_length(v1,v2):
    # v1 v2 两个向量是否是一个方向，如果大于0 一个方向，小于0，反方向
    return True if np.dot(v1,v2)>0 and 0.3*np.linalg.norm(v2)<np.linalg.norm(v1)<3*np.linalg.norm(v2) else False


def point_inside_of_box(xy_text,quad_xy_list):
    xy_list = np.zeros((4, 2))
    xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
    xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
    yx_list = np.zeros((4, 2))
    yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
    a = np.array(list(map(lambda xy:xy_list * (xy - yx_list),xy_text[:,::-1])))
    b = a[:,:,0] - a[:,:,1]
    idex = (np.amin(b,axis=-1)>=0)+(np.amax(b,axis=-1)<=0)
    return np.argwhere(idex==True).reshape(-1).tolist()


def compute_centerness_targets(xy_text,geo_map):
    im_geo_map = geo_map[xy_text[:, 0], xy_text[:, 1], :]
    return _compute_centerness_targets(im_geo_map[:,:4])


def _compute_centerness_targets(reg_targets):
    if reg_targets.shape[0]!=0:
        left_right = reg_targets[:, [3, 1]]
        top_bottom = reg_targets[:, [0, 2]]
        centerness = (left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                     (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))
        # return np.square(centerness)
        return np.sqrt(centerness)
    else:
        return []


def order_points(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    if xSorted[1][0] == xSorted[2][0] and xSorted[1][1] >= xSorted[2][1]:
        xSorted = xSorted[[0,2,1,3], :]

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
    return np.array([tl, tr, br, bl], dtype="int32")
#######################################


def postProcess(score, geometry, cos_map, sin_map, ratio_h, ratio_w, order_points_flag=True):
    boxes, start_points_indexs = detect(score_map=score, geo_map=geometry, cos_map=cos_map,
                                        sin_map=sin_map, score_map_thresh=0.8)
    bbs = []
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        for index, box in enumerate(boxes):
            start_point_index = start_points_indexs[index]
            box = box[[start_point_index, (start_point_index + 1) % 4, (start_point_index + 2) % 4,
                       (start_point_index + 3) % 4]]

        if (len(boxes)):
            if order_points_flag:
                bbs = [order_points(boxes[i,:,:]).tolist() for i in range(boxes.shape[0])]
            else:
                bbs = [boxes[i,:,:].tolist() for i in range(boxes.shape[0])]
    return bbs

def run(src_dir, dst_dir, is_transpose=False, order_points_flag=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    names = load_files(src_dir+'/score/bin')
    for name in names:
        score, ratio_h, ratio_w = load_feat(src_dir+'/score', name)

        geometry, _, _ = load_feat(src_dir+'/geometry', name)
        cos_map, _, _ = load_feat(src_dir+'/cos_map', name)
        sin_map, _, _ = load_feat(src_dir+'/sin_map', name)
        print (name)
        if is_transpose:
            score = np.transpose(score, [0,2,3,1])
            geometry = np.transpose(geometry, [0,2,3,1])
            cos_map = np.transpose(cos_map, [0,2,3,1])
            sin_map = np.transpose(sin_map, [0,2,3,1])
        # print (name, score.shape, geometry.shape, cos_map.shape, sin_map.shape)
        boxes, start_points_indexs = detect(score_map=score, geo_map=geometry, cos_map=cos_map,
                                            sin_map=sin_map, score_map_thresh=0.8)
        bbs = []
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for index, box in enumerate(boxes):
                start_point_index = start_points_indexs[index]
                box = box[[start_point_index, (start_point_index + 1) % 4, (start_point_index + 2) % 4,
                           (start_point_index + 3) % 4]]
            if (len(boxes)):
                if order_points_flag:
                    bbs = [order_points(boxes[i,:,:]).tolist() for i in range(boxes.shape[0])]
                else:
                    bbs = [boxes[i,:,:].tolist() for i in range(boxes.shape[0])]
        with open(dst_dir+'/'+name+'.txt', 'w') as f:
            for bb in bbs:
                f.write(str(bb[0][0])+','+str(bb[0][1])+','+str(bb[1][0])+','+str(bb[1][1])+','+str(bb[2][0])+','+str(bb[2][1])+','+str(bb[3][0])+','+str(bb[3][1])+'\n')

if __name__ == '__main__':

    src_path = "../../../build/test_data/ocr_east_data/sampleTrtEAST_fix_trt_fp32"
    dst_path = "../../../build/test_data/ocr_east_data/sampleTrtEAST_fix_trt_fp32_pred"
    is_transpose = False # nhwc:False, nchw:True

    run(src_path, dst_path, is_transpose=is_transpose)

