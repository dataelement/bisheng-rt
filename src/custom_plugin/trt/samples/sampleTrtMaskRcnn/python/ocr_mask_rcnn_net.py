#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generalized_rcnn.py

import tensorflow as tf
from config import finalize_configs, config as cfg
from op import BatchNorm, Conv2D, FixedUnPooling, MaxPooling, FullyConnected, Conv2DTranspose
import numpy as np
import itertools
from six.moves import range
import math


# preprocess
def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.PREPROC.PIXEL_MEAN
        std = np.asarray(cfg.PREPROC.PIXEL_STD)
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        # N,H,W,C输入时需要做特殊处理，否则uff parse时报错： Invalid scale mode, nbWeight=3
        image_mean_b = tf.constant(mean[0], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_mean_g = tf.constant(mean[1], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_mean_r = tf.constant(mean[2], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_b = tf.constant(1.0 / std[0], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_g = tf.constant(1.0 / std[1], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_r = tf.constant(1.0 / std[2], dtype=tf.float32, shape=[cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_mean_b = tf.expand_dims(image_mean_b, 0)
        image_mean_g = tf.expand_dims(image_mean_g, 0)
        image_mean_r = tf.expand_dims(image_mean_r, 0)
        image_invstd_b = tf.expand_dims(image_invstd_b, 0)
        image_invstd_g = tf.expand_dims(image_invstd_g, 0)
        image_invstd_r = tf.expand_dims(image_invstd_r, 0)
        image_mean_b = tf.reshape(image_mean_b, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_mean_g = tf.reshape(image_mean_g, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_mean_r = tf.reshape(image_mean_r, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_b = tf.reshape(image_invstd_b, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_g = tf.reshape(image_invstd_g, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])
        image_invstd_r = tf.reshape(image_invstd_r, [1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1])

        image_mean = tf.concat([image_mean_b, image_mean_g, image_mean_r], axis=3)
        image_invstd = tf.concat([image_invstd_b, image_invstd_g, image_invstd_r], axis=3)

        # image_mean = tf.constant(mean, dtype=tf.float32)
        # image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        # image = tf.transpose(image, [0, 2, 3, 1])
        image = (image - image_mean) * image_invstd
        image = tf.transpose(image, [0, 3, 1, 2])
        return image


# backbone
def maybe_reverse_pad(topleft, bottomright):
    if cfg.BACKBONE.TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]


def nonlin(x):
    x = BatchNorm('bn', x, gamma_initializer=None)
    return tf.nn.relu(x)


def get_norm(zero_init=False):
    return lambda x: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer() if zero_init else None)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.shape[1]
    if n_in != n_out:   # change dimension when channel is not the same
        # TF's SAME mode output ceil(x/stride), which is NOT what we want when x is odd and stride is 2
        l = Conv2D('convshortcut', l, n_out, 1, strides=stride,
                   use_bias=False,
                   activation=activation,
                   kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
        return l
    else:
        return l


def resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=1,
               use_bias=False,
               activation=nonlin,
               kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
    if stride == 2:
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
        l = Conv2D('conv2', l, ch_out, 3, strides=2, padding='VALID',
                   use_bias=False,
                   activation=nonlin,
                   kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
    else:
        l = Conv2D('conv2', l, ch_out, 3, strides=stride,
                   use_bias=False,
                   activation=nonlin,
                   kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))

    l = Conv2D('conv3', l, ch_out * 4, 1,
               use_bias=False,
               activation=get_norm(zero_init=True),
               kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))

    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_norm(zero_init=False))
    return tf.nn.relu(ret, name='output')


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_fpn_backbone(image, num_blocks):
    shape2d = tf.shape(image)[2:]
    mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)
    new_shape2d = tf.cast(tf.ceil(tf.cast(shape2d, tf.float32) / mult) * mult, tf.int32)
    pad_shape2d = new_shape2d - shape2d
    assert len(num_blocks) == 4, num_blocks

    chan = image.shape[1]
    pad_base = maybe_reverse_pad(2, 3)
    # l = tf.pad(image, tf.stack(
    #     [[0, 0], [0, 0],
    #      [pad_base[0], pad_base[1] + pad_shape2d[0]],
    #      [pad_base[0], pad_base[1] + pad_shape2d[1]]]))
    l = tf.pad(image, [[0, 0], [0, 0], [pad_base[0], pad_base[1]], [pad_base[0], pad_base[1]]])
    l.set_shape([None, chan, None, None])
    l = Conv2D('conv0', l, 64, 7, strides=2, padding='VALID',
               use_bias=False,
               activation=nonlin,
               kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
    l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
    l = MaxPooling('pool0', l, 3, strides=2, padding='VALID')

    c2 = resnet_group('group0', l, resnet_bottleneck, 64, num_blocks[0], 1)
    c3 = resnet_group('group1', c2, resnet_bottleneck, 128, num_blocks[1], 2)
    c4 = resnet_group('group2', c3, resnet_bottleneck, 256, num_blocks[2], 2)
    c5 = resnet_group('group3', c4, resnet_bottleneck, 512, num_blocks[3], 2)
    # 32x downsampling up to now
    # size of c5: ceil(input/32)
    return c2, c3, c4, c5


# fpn
def fpn_model(features):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    with tf.variable_scope('fpn'):
        assert len(features) == 4, features
        num_channel = cfg.FPN.NUM_CHANNEL

        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1, activation=tf.identity, use_bias=True,
                    kernel_initializer=tf.variance_scaling_initializer(scale=1.))
                    for i, c in enumerate(features)]

        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + FixedUnPooling('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1], 2, unpool_mat=np.ones((2, 2), dtype='float32'))
                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3, activation=tf.identity, use_bias=True,
                 kernel_initializer=tf.variance_scaling_initializer(scale=1.))
                 for i, c in enumerate(lat_sum_5432[::-1])]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first', padding='VALID')
        return p2345 + [p6]


# generate anchors
def generate_anchors_keras(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # # Enumerate heights and widths from scales and ratios
    # heights = scales / np.sqrt(ratios)
    # widths = scales * np.sqrt(ratios)

    # base anchor为4*4大小
    size_ratios = feature_stride * feature_stride / ratios
    widths = np.round(np.sqrt(size_ratios))
    heights = np.round(widths * ratios)
    widths = widths * (scales / feature_stride)
    heights = heights * (scales / feature_stride)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride + (feature_stride - 1) / 2
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride + (feature_stride - 1) / 2
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * (box_sizes - 1),
                            box_centers + 0.5 * (box_sizes - 1)], axis=1)
    # boxes = boxes.reshape([shape[0], shape[1], len(ratios), 4])
    # boxes[:, :, :, [2, 3]] += 1
    # boxes = boxes[:, :, :, [1, 0, 3, 2]].astype(np.float32)
    boxes[:, [2, 3]] += 1
    boxes = boxes[:, [1, 0, 3, 2]].astype(np.float32)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors_keras(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return anchors


# rpn层
def rpn_head(featuremap, channel, num_anchors):
    """
    Returns:
        label_logits: 1, fHxfWxNA
        box_logits: 1, fHxfWxNA, 4
    """
    with tf.variable_scope('rpn', reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
        hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        label_logits = Conv2D('class', hidden, num_anchors, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        box_logits = Conv2D('box', hidden, 4 * num_anchors, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        # 1, NA(*4), im/16, im/16 (NCHW)
        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.reshape(label_logits, [1, shp[2] * shp[3] * num_anchors, 1])

        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, [1, shp[2] * shp[3] * num_anchors, 4])  # fHxfWxNAx4
        return label_logits, box_logits


def rpn(image, features):
    assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

    # Multi-Level RPN Proposals
    rpn_outputs = [rpn_head(pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS)) for pi in features]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]

    image_shape2d = tf.shape(image)[2:]     # h,w
    backbone_shapes = np.array([[int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride)),
                                 int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride))]
                                 for stride in cfg.FPN.ANCHOR_STRIDES])
    all_anchors_fpn = generate_pyramid_anchors(scales=tuple(cfg.RPN.ANCHOR_SIZES),
                                               ratios=tuple(cfg.RPN.ANCHOR_RATIOS),
                                               feature_shapes=backbone_shapes,
                                               feature_strides=tuple(cfg.FPN.ANCHOR_STRIDES),
                                               anchor_stride=1)

    all_anchors_concate = np.concatenate(all_anchors_fpn, axis=0)
    all_anchors_concate = np.broadcast_to(all_anchors_concate, (1,) + all_anchors_concate.shape)
    all_label_logits = tf.concat(multilevel_label_logits, axis=1)
    all_box_logits = tf.concat(multilevel_box_logits, axis=1)

    proposal_boxes = generate_fpn_proposals(
        all_label_logits, all_box_logits, all_anchors_concate, image_shape2d)

    return proposal_boxes


# proposal层
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    with tf.name_scope('clip_boxes'):
        boxes = tf.maximum(boxes, 0.0)
        m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
        boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
        return boxes


def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    with tf.name_scope('decode_bbox_target'):
        orig_shape = tf.shape(anchors)
        box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
        box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
        # each is (...)x1x2
        anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
        anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

        waha = anchors_x2y2 - anchors_x1y1
        xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

        clip = np.log(cfg.PREPROC.MAX_SIZE / 16.)
        wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
        xbyb = box_pred_txty * waha + xaya
        x1y1 = xbyb - wbhb * 0.5
        x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
        out = tf.concat([x1y1, x2y2], axis=-2)
        return tf.reshape(out, orig_shape)


def generate_rpn_proposals(scores, boxes_logits, anchors, img_shape,
                           pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        scores: [batch, num_anchors, 1]
        boxes_logits: [batch, num_anchors, (dx, dy, log(dw), log(dh))]
        anchors: [batch, num_anchors, (x1, y1, x2, y2)] anchors in normalized coordinates
        img_shape: [h, w]
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: batch, k, 4 float
    """
    with tf.name_scope('generate_rpn_proposals'):
        if post_nms_topk is None:
            post_nms_topk = pre_nms_topk

        # boxes = batch_slice([boxes_logits, anchors], lambda x, y: decode_bbox_target(x, y), 1,
        #                           names=["refined_anchors"])

        # boxes = batch_slice(boxes, lambda x: clip_boxes(x, img_shape), 1,
        #                           names=["refined_anchors_clipped"])
        # proposal_boxes = boxes

        scores = scores[:, :, 0]
        topk = tf.minimum(pre_nms_topk, tf.shape(anchors)[1]) # Pick top k1 by scores
        ix = tf.nn.top_k(scores, topk, sorted=True, name="top_anchors").indices
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y), 1)
        deltas = batch_slice([boxes_logits, ix], lambda x, y: tf.gather(x, y), 1)
        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), 1, names=["pre_nms_anchors"])

        boxes = batch_slice([deltas, pre_nms_anchors], lambda x, y: decode_bbox_target(x, y), 1,
                                  names=["refined_anchors"])

        boxes = batch_slice(boxes, lambda x: clip_boxes(x, img_shape), 1,
                                  names=["refined_anchors_clipped"])

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, post_nms_topk,
                cfg.RPN.PROPOSAL_NMS_THRESH, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(post_nms_topk - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposal_boxes = batch_slice([boxes, scores], nms, 1)

        return tf.identity(proposal_boxes, name='boxes')


def generate_fpn_proposals(
        all_label_logits, all_box_logits, all_anchors, image_shape2d):
    """
    Args:
        all_box_logits: batch, num_anchors, 4
        all_label_logits: batch, num_anchors
        all_anchors: batch, num_anchors, 4

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    with tf.name_scope('generate_fpn_proposals'):
        proposal_boxes = generate_rpn_proposals(
            all_label_logits, all_box_logits, all_anchors, image_shape2d,
            cfg.RPN.TEST_PRE_NMS_TOPK, cfg.RPN.TEST_POST_NMS_TOPK)

    return proposal_boxes


# RoiAlign
def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution

    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    with tf.name_scope('roi_align'):
        ret = crop_and_resize(
            featuremap, boxes,
            tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
            resolution * 2)
        ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')

        return ret


def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes (n, 4):

    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    with tf.name_scope('fpn_map_rois_to_levels'):
        x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
        h = y2 - y1
        w = x2 - x1
        # shape: (n, 1)
        sqrtarea = tf.sqrt(h * w)
        level = tf.cast(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)
        level = tf.squeeze(level, 1)

        # RoI levels range from 2~5 (not 6)
        level_ids = [
            tf.where(level <= 2),
            tf.where(tf.equal(level, 3)),   # == is not supported
            tf.where(tf.equal(level, 4)),
            tf.where(level >= 5)]
        level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                     for i, x in enumerate(level_ids)]
        num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                         for i, x in enumerate(level_ids)]

        level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
        return level_ids, level_boxes


def multilevel_roi_align(rcnn_boxes, features, resolution):
    """
    Args:
        rcnn_boxes (tf.Tensor): nx4 boxes
        features ([tf.Tensor]): 4 FPN feature level 2-5
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    with tf.name_scope('multilevel_roi_align'):
        assert len(features) == 4, features
        # Reassign rcnn_boxes to levels
        level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
        all_rois = []

        # Crop patches from corresponding levels
        for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
            with tf.name_scope('roi_level{}'.format(i + 2)):
                boxes_on_featuremap = boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
                all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

        # this can fail if using TF<=1.8 with MKL build
        all_rois = tf.concat(all_rois, axis=0)  # NCHW
        # Unshuffle to the original order, to match the original samples
        level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
        level_id_invert_perm = tf.invert_permutation(level_id_perm)
        all_rois = tf.gather(all_rois, level_id_invert_perm, name="output")
        return all_rois


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    with tf.name_scope('crop_and_resize'):
        assert isinstance(crop_size, int), crop_size
        boxes = tf.identity(boxes)

        # TF's crop_and_resize produces zeros on border
        if pad_border:
            # this can be quite slow
            image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
            boxes = boxes + 1

        def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
            """
            The way tf.image.crop_and_resize works (with normalized box):
            Initial point (the value of output[0]): x0_box * (W_img - 1)
            Spacing: w_box * (W_img - 1) / (W_crop - 1)
            Use the above grid to bilinear sample.

            However, what we want is (with fpcoor box):
            Spacing: w_box / W_crop
            Initial point: x0_box + spacing/2 - 0.5
            (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
             (0.0, 0.0) is the same as pixel value (0, 0))

            This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

            Returns:
                y1x1y2x2
            """
            with tf.name_scope('transform_fpcoor_for_tf'):
                x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

                spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
                spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

                imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
                nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
                ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

                nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
                nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

                return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        image_shape = tf.shape(image)[2:]
        boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
        image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
        ret = tf.image.crop_and_resize(image, boxes, tf.cast(box_ind, tf.int32), crop_size=[crop_size, crop_size])
        ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
        return ret


def roi_align_v2(featuremap, boxes, box_indices, resolution):
    """
    Args:
        featuremap: BxCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution

    Returns:
        N x C x res x res
    """
    # sample 4 locations per roi bin
    with tf.name_scope('roi_align'):
        ret = crop_and_resize(featuremap, boxes, box_indices, resolution * 2)

        return ret


def multilevel_roi_align_v2(rcnn_boxes, features, resolution, name):
    """
    Args:
        rcnn_boxes (tf.Tensor): batch, n, 4 boxes
        features ([tf.Tensor]): 4 FPN feature level 2-5
        resolution (int): output spatial resolution
    Returns:
        batch, N, C, res, res
    """
    with tf.name_scope(name):
        assert len(features) == 4, features
        rcnn_boxes = tf.reshape(rcnn_boxes, [1, -1, 4])
        # Reassign rcnn_boxes to levels
        x1, y1, x2, y2 = tf.split(rcnn_boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # shape: (batch, n, 1)
        sqrtarea = tf.sqrt(h * w)
        roi_level = tf.cast(tf.floor(4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)
        roi_level = tf.minimum(5, tf.maximum(2, roi_level))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(rcnn_boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            level_boxes = level_boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            pooled.append(roi_align_v2(features[i], level_boxes, box_indices, resolution))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(rcnn_boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)

    return pooled


# Rcnn层
def fastrcnn_2fc_head(feature):
    """
    Args:
        feature (any shape):

    Returns:
        2D head feature
    """
    with tf.variable_scope('head'):
        dim = cfg.FPN.FRCNN_FC_HEAD_DIM
        init = tf.variance_scaling_initializer()
        hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
        hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)
        return hidden


def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4

    Returns:
        cls_logits: N x num_class classification logits
        reg_logits: N x num_classx4 or Nx1x4 if class agnostic
    """
    with tf.variable_scope('outputs'):
        classification = FullyConnected(
            'class', feature, num_classes,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        num_classes_for_box = 1 if class_agnostic_regression else num_classes
        box_regression = FullyConnected(
            'box', feature, num_classes_for_box * 4,
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        box_cos = FullyConnected(
            'cos', feature, num_classes_for_box,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        box_sin = FullyConnected(
            'sin', feature, num_classes_for_box,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return classification, box_regression, box_cos, box_sin


def decoded_output_boxes_class_agnostic(proposals, box_logits, reg_weights):
    """ Returns: Nx4 """
    # bbox_class_agnostic = int(box_logits.shape[1]) == 1
    # assert bbox_class_agnostic
    box_logits = tf.reshape(box_logits, [-1, 4])
    decoded = decode_bbox_target(box_logits / reg_weights, proposals)
    return decoded


def decoded_output_scores(label_logits, name=None):
    """ Returns: N x #class scores, summed to one for each box."""
    return tf.nn.softmax(label_logits, name=name)


# box detector layer
def fastrcnn_predictions(boxes, scores, boxes_cos, boxes_sin):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#classx4 floatbox in float32
        scores: nx#class

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """
    with tf.name_scope('output'):
        assert boxes.shape[1] == scores.shape[1]
        boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
        scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn
        boxes_cos = tf.transpose(boxes_cos[:, 1:], [1, 0])  # #catxn
        boxes_sin = tf.transpose(boxes_sin[:, 1:], [1, 0])  # #catxn

        max_coord = tf.reduce_max(boxes)
        filtered_ids = tf.where(scores > cfg.TEST.RESULT_SCORE_THRESH)  # Fx2
        filtered_boxes = tf.gather_nd(boxes, filtered_ids)  # Fx4
        filtered_scores = tf.gather_nd(scores, filtered_ids)  # F,
        filtered_boxes_cos = tf.gather_nd(boxes_cos, filtered_ids)  # F,
        filtered_boxes_sin = tf.gather_nd(boxes_sin, filtered_ids)  # F,
        cls_per_box = tf.slice(filtered_ids, [0, 0], [-1, 1])
        offsets = tf.cast(cls_per_box, tf.float32) * (max_coord + 1)  # F,1
        # 不同类别的box加了不同偏移量，这样所有类别的box可以一起做nms，效果等价于每一类单独做nms，最后选出分数排名前RESULTS_PER_IM个box
        nms_boxes = filtered_boxes + offsets
        selection = tf.image.non_max_suppression(
            nms_boxes,
            filtered_scores,
            cfg.TEST.RESULTS_PER_IM,
            cfg.TEST.FRCNN_NMS_THRESH)
        final_scores = tf.gather(filtered_scores, selection, name='scores')
        final_labels = tf.add(tf.gather(cls_per_box[:, 0], selection), 1, name='labels')
        final_boxes = tf.gather(filtered_boxes, selection, name='boxes')
        final_boxes_cos = tf.gather(filtered_boxes_cos, selection)
        final_boxes_sin = tf.gather(filtered_boxes_sin, selection)
        final_boxes_cos = tf.sigmoid(final_boxes_cos, name='boxes_cos')
        final_boxes_sin = tf.sigmoid(final_boxes_sin, name='boxes_sin')
        return final_boxes, final_scores, final_labels, final_boxes_cos, final_boxes_sin


# box detector layer
def fastrcnn_predictions_v2(proposals, box_logits, label_logits,
                            box_cos_logits, box_sin_logits, image_shape2d, cascade_stage_index):
    """
    Generate final results from predictions of all proposals.

    Args:
        proposals: nx4
        box_logits: nx4 floatbox in float32
        label_logits: nxclass
        box_cos_logits: nx1
        box_sin_logits: nx1

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """

    proposals = tf.reshape(proposals, [cfg.RPN.TEST_POST_NMS_TOPK, 4])
    # 每个box只会归属到一个类别（根据分数最大来归属）
    # class_ids = tf.argmax(label_logits, axis=1, output_type=tf.int32)
    class_ids = tf.ones(label_logits.shape[0], dtype=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(label_logits.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(label_logits, indices)
    # Class-specific bounding box deltas
    deltas_specific = box_logits
    # Apply bounding box deltas
    # Shape: [boxes, (x1, y1, x2, y2)] in normalized coordinates
    reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[cascade_stage_index], dtype=tf.float32)
    refined_rois = decode_bbox_target(deltas_specific / reg_weights, proposals)
    # Clip boxes to image window
    refined_rois = clip_boxes(refined_rois, image_shape2d)

    box_cos_logits = tf.reshape(box_cos_logits, [cfg.RPN.TEST_POST_NMS_TOPK])
    box_sin_logits = tf.reshape(box_sin_logits, [cfg.RPN.TEST_POST_NMS_TOPK])

    # 算面积
    x1, y1, x2, y2 = tf.split(refined_rois, 4, axis=1)
    h = y2 - y1
    w = x2 - x1
    sqrtarea = tf.sqrt(h * w)
    sqrtarea = tf.squeeze(sqrtarea, 1)

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes and padding boxes(area=0)
    if cfg.TEST.RESULT_SCORE_THRESH:
        conf_keep = tf.where(tf.math.logical_and(class_scores > cfg.TEST.RESULT_SCORE_THRESH, sqrtarea > 0))[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=cfg.TEST.RESULTS_PER_IM,
                iou_threshold=cfg.TEST.FRCNN_NMS_THRESH)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # # Pad with -1 so returned tensors have the same shape
        gap = cfg.TEST.RESULTS_PER_IM - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([cfg.TEST.RESULTS_PER_IM])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = cfg.TEST.RESULTS_PER_IM
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (x1, y1, x2, y2, class_id, score, cos, sin)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis],
        tf.gather(box_cos_logits, keep)[..., tf.newaxis],
        tf.gather(box_sin_logits, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < cfg.TEST.RESULTS_PER_IM
    gap = cfg.TEST.RESULTS_PER_IM - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

    return detections


# maskrcnn layer
def maskrcnn_upXconv_head(feature, num_category, num_convs=4):
    """
    Args:
        feature (NxCx s x s): size is 7 in C4 models and 14 in FPN models.
        num_category(int):
        num_convs (int): number of convolution layers
        norm (str or None): either None or 'GN'

    Returns:
        mask_logits (N x num_category x 2s x 2s):
    """
    with tf.variable_scope('maskrcnn'):
        l = feature
        # c2's MSRAFill is fan_out
        for k in range(num_convs):
            l = Conv2D('fcn{}'.format(k), l, cfg.MRCNN.HEAD_DIM, 3, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
        l = Conv2DTranspose('deconv', l, cfg.MRCNN.HEAD_DIM, 2, strides=2, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
        l = Conv2D('conv', l, num_category, 1, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
        return l


# 回归box和mask
def roi_heads(image, features, proposals):
    image_shape2d = tf.shape(image)[2:]     # h,w
    assert len(features) == 5, "Features have to be P23456!"

    with tf.variable_scope('cascade_rcnn_stage1'):
        proposals = tf.reshape(proposals, [-1, 4])
        # num_rois, 256, 7, 7
        roi_feature_fastrcnn = multilevel_roi_align(proposals, features[:4], 7)
        head_feature = fastrcnn_2fc_head(roi_feature_fastrcnn)
        label_logits, box_logits, box_cos_logits, box_sin_logits = fastrcnn_outputs(head_feature,
                                                                   cfg.DATA.NUM_CLASS, class_agnostic_regression=True)

        reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[0], dtype=tf.float32)
        refined_boxes = decoded_output_boxes_class_agnostic(proposals, box_logits, reg_weights)

    # box
    refined_boxes = tf.expand_dims(refined_boxes, 1)     # class-agnostic
    decoded_boxes =  tf.tile(refined_boxes, [1, cfg.DATA.NUM_CLASS, 1])
    decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
    boxes_cos = tf.reshape(box_cos_logits, [-1])
    boxes_cos = tf.expand_dims(boxes_cos, 1)     # class-agnostic
    boxes_cos = tf.tile(boxes_cos, [1, cfg.DATA.NUM_CLASS])
    boxes_sin = tf.reshape(box_sin_logits, [-1])
    boxes_sin = tf.expand_dims(boxes_sin, 1)     # class-agnostic
    boxes_sin = tf.tile(boxes_sin, [1, cfg.DATA.NUM_CLASS])
    # score
    label_scores = decoded_output_scores(label_logits, 'cascade_scores_stage1')
    # finnal box socre output
    final_boxes, final_scores, final_labels, final_boxes_cos, final_boxes_sin = fastrcnn_predictions(decoded_boxes,
                                                                                                     label_scores,
                                                                                                     boxes_cos,
                                                                                                     boxes_sin)

    # Cascade inference needs roi transform with refined boxes.
    roi_feature_maskrcnn = multilevel_roi_align(final_boxes, features[:4], 14)
    mask_logits = maskrcnn_upXconv_head(roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
    indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
    final_masks = tf.sigmoid(final_mask_logits, name='output/masks')

    return final_boxes, final_scores, final_masks, final_boxes_cos, final_boxes_sin


# 回归box和mask
def roi_heads_v2(image, features, proposals):
    image_shape2d = tf.shape(image)[2:]     # h,w
    assert len(features) == 5, "Features have to be P23456!"

    # RoIAlign + rcnn
    with tf.variable_scope('cascade_rcnn_stage1'):
        # 1, num_rois, 256, 14, 14
        roi_feature_fastrcnn = multilevel_roi_align_v2(proposals, features[:4], 7, 'multilevel_roi_align_rcnn')
        roi_feature_fastrcnn = tf.reshape(roi_feature_fastrcnn, [cfg.RPN.TEST_POST_NMS_TOPK, cfg.FPN.NUM_CHANNEL, 14, 14],
                                          name='rcnn_reshape')
        # num_rois, 256, 7, 7
        roi_feature_fastrcnn = tf.nn.avg_pool(roi_feature_fastrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                              padding='SAME', data_format='NCHW')
        head_feature = fastrcnn_2fc_head(roi_feature_fastrcnn)
        label_logits, box_logits, box_cos_logits, box_sin_logits= fastrcnn_outputs(head_feature,
                                                                                   cfg.DATA.NUM_CLASS,
                                                                                   class_agnostic_regression=True)
        label_scores = decoded_output_scores(label_logits, 'cascade_scores_stage1')

    # rcnn output
    with tf.name_scope('frcnn_output'):
        detections = fastrcnn_predictions_v2(proposals, box_logits, label_scores,
                                             box_cos_logits, box_sin_logits, image_shape2d, 0)

    # slice output
    with tf.name_scope('detection_slice'):
        final_boxes = detections[:, 0:4]

    # RoIAlign + mask
    # 1, num_rois, 256, 28, 28
    roi_feature_maskrcnn = multilevel_roi_align_v2(final_boxes, features[:4], 14, 'multilevel_roi_align_mask')
    roi_feature_maskrcnn = tf.reshape(roi_feature_maskrcnn, [cfg.TEST.RESULTS_PER_IM, cfg.FPN.NUM_CHANNEL, 28, 28], name='mask_reshape')
    # num_rois, 256, 14, 14
    roi_feature_maskrcnn = tf.nn.avg_pool(roi_feature_maskrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                          padding='SAME', data_format='NCHW')
    mask_logits = maskrcnn_upXconv_head(roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
    # mask_logits = tf.reshape(mask_logits, [cfg.TEST.RESULTS_PER_IM, 28, 28])
    final_masks = tf.sigmoid(mask_logits, name='output/masks')

    return detections, final_masks


# 回归box和mask
def roi_heads_cascade(image, features, proposals):
    image_shape2d = tf.shape(image)[2:]     # h,w
    assert len(features) == 5, "Features have to be P23456!"

    # RoIAlign + rcnn
    with tf.variable_scope('cascade_rcnn_stage1'):
        # 1, num_rois, 256, 14, 14
        roi_feature_fastrcnn = multilevel_roi_align_v2(proposals, features[:4], 7, 'multilevel_roi_align_rcnn')
        roi_feature_fastrcnn = tf.reshape(roi_feature_fastrcnn, [cfg.RPN.TEST_POST_NMS_TOPK, cfg.FPN.NUM_CHANNEL, 14, 14],
                                          name='rcnn_reshape')
        # num_rois, 256, 7, 7
        roi_feature_fastrcnn = tf.nn.avg_pool(roi_feature_fastrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                              padding='SAME', data_format='NCHW')
        head_feature = fastrcnn_2fc_head(roi_feature_fastrcnn)
        label_logits_1, box_logits_1, box_cos_logits_1, box_sin_logits_1 = fastrcnn_outputs(head_feature,
                                                                                            cfg.DATA.NUM_CLASS,
                                                                                            class_agnostic_regression=True)
        label_scores_1 = decoded_output_scores(label_logits_1, 'cascade_scores_stage1')
        with tf.name_scope('decode_output_boxes'):
            proposals = tf.reshape(proposals, [cfg.RPN.TEST_POST_NMS_TOPK, 4])
            reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[0], dtype=tf.float32)
            refined_boxes = decoded_output_boxes_class_agnostic(proposals, box_logits_1, reg_weights)
            proposals = clip_boxes(refined_boxes, image_shape2d)

    with tf.variable_scope('cascade_rcnn_stage2'):
        # 1, num_rois, 256, 14, 14
        roi_feature_fastrcnn = multilevel_roi_align_v2(proposals, features[:4], 7, 'multilevel_roi_align_rcnn')
        roi_feature_fastrcnn = tf.reshape(roi_feature_fastrcnn, [cfg.RPN.TEST_POST_NMS_TOPK, cfg.FPN.NUM_CHANNEL, 14, 14],
                                          name='rcnn_reshape')
        # num_rois, 256, 7, 7
        roi_feature_fastrcnn = tf.nn.avg_pool(roi_feature_fastrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                              padding='SAME', data_format='NCHW')
        head_feature = fastrcnn_2fc_head(roi_feature_fastrcnn)
        label_logits_2, box_logits_2, box_cos_logits_2, box_sin_logits_2 = fastrcnn_outputs(head_feature,
                                                                                            cfg.DATA.NUM_CLASS,
                                                                                            class_agnostic_regression=True)
        label_scores_2 = decoded_output_scores(label_logits_2, 'cascade_scores_stage2')
        with tf.name_scope('decode_output_boxes'):
            proposals = tf.reshape(proposals, [cfg.RPN.TEST_POST_NMS_TOPK, 4])
            reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[1], dtype=tf.float32)
            refined_boxes = decoded_output_boxes_class_agnostic(proposals, box_logits_2, reg_weights)
            proposals = clip_boxes(refined_boxes, image_shape2d)

    with tf.variable_scope('cascade_rcnn_stage3'):
        # 1, num_rois, 256, 14, 14
        roi_feature_fastrcnn = multilevel_roi_align_v2(proposals, features[:4], 7, 'multilevel_roi_align_rcnn')
        roi_feature_fastrcnn = tf.reshape(roi_feature_fastrcnn, [cfg.RPN.TEST_POST_NMS_TOPK, cfg.FPN.NUM_CHANNEL, 14, 14],
                                          name='rcnn_reshape')
        # num_rois, 256, 7, 7
        roi_feature_fastrcnn = tf.nn.avg_pool(roi_feature_fastrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                              padding='SAME', data_format='NCHW')
        head_feature = fastrcnn_2fc_head(roi_feature_fastrcnn)
        label_logits_3, box_logits_3, box_cos_logits_3, box_sin_logits_3 = fastrcnn_outputs(head_feature,
                                                                                            cfg.DATA.NUM_CLASS,
                                                                                            class_agnostic_regression=True)
        label_scores_3 = decoded_output_scores(label_logits_3, 'cascade_scores_stage3')

    label_scores = (label_scores_1 + label_scores_2 + label_scores_3) / 3
    # rcnn output
    with tf.name_scope('frcnn_output'):
        detections = fastrcnn_predictions_v2(proposals, box_logits_3, label_scores,
                                             box_cos_logits_3, box_sin_logits_3, image_shape2d, 2)

    # slice output
    with tf.name_scope('detection_slice'):
        final_boxes = detections[:, 0:4]

    # RoIAlign + mask
    # 1, num_rois, 256, 28, 28
    roi_feature_maskrcnn = multilevel_roi_align_v2(final_boxes, features[:4], 14, 'multilevel_roi_align_mask')
    roi_feature_maskrcnn = tf.reshape(roi_feature_maskrcnn, [cfg.TEST.RESULTS_PER_IM, cfg.FPN.NUM_CHANNEL, 28, 28], name='mask_reshape')
    # num_rois, 256, 14, 14
    roi_feature_maskrcnn = tf.nn.avg_pool(roi_feature_maskrcnn, [1, 1, 2, 2], [1, 1, 2, 2],
                                          padding='SAME', data_format='NCHW')
    mask_logits = maskrcnn_upXconv_head(roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
    # mask_logits = tf.reshape(mask_logits, [cfg.TEST.RESULTS_PER_IM, 28, 28])
    final_masks = tf.sigmoid(mask_logits, name='output/masks')

    return detections, final_masks


def build_ocr_maskrcnn(image):
    # preprocess
    image = image_preprocess(image, bgr=True)

    # backbone
    c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)

    # fpn
    p23456 = fpn_model(c2345)

    # rpn
    proposals = rpn(image, p23456)

    # roi
    detections, final_masks = roi_heads_v2(image, p23456, proposals)

    return detections, final_masks


def build_ocr_maskrcnn_cascade(image):
    # preprocess
    image = image_preprocess(image, bgr=True)

    # backbone
    c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)

    # fpn
    p23456 = fpn_model(c2345)

    # rpn
    proposals = rpn(image, p23456)

    # roi
    detections, final_masks = roi_heads_cascade(image, p23456, proposals)

    return detections, final_masks

