import base64
import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from shapely.geometry import Polygon
from torchvision import ops

from .config import config as cfg
from .config import finalize_configs

############################################################
#  Pytorch Utility Functions
############################################################


def intersection(x, y):
    combined = torch.concat([x, y], dim=0)
    uniques, counts = torch.unique(combined, return_counts=True)
    res = uniques[counts > 1]
    return res


class Pad2d(nn.Module):
    # BCWH (left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad)
    def __init__(self, pad_info=(2, 3, 2, 3)):
        super(Pad2d, self).__init__()
        self.pad_info = pad_info

    def forward(self, input):
        (pad_left, pad_right, pad_top, pad_bottom) = self.pad_info
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom),
                     'constant', 0)


class SamePad2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(in_width / self.stride[0])
        out_height = math.ceil(in_height / self.stride[1])
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom),
                     'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


def clip_boxes(boxes, window):
    """
        boxes: [N, 4] each col is x1, y1, x2, y2
        window: [h, w]
        """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(0, window[0]),
        boxes[:, 1].clamp(0, window[1]),
        boxes[:, 2].clamp(0, window[0]),
        boxes[:, 3].clamp(0, window[1])], 1)
    return boxes


def bbox_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] +
                                                 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] +
                                                 1)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = anchors.size()
    box_pred_txtytwth = torch.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = torch.chunk(box_pred_txtytwth, 2, dim=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = torch.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = torch.chunk(anchors_x1y1x2y2, 2, dim=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = torch.tensor(np.log(cfg.PREPROC.MAX_SIZE / 16.))
    wbhb = torch.exp(torch.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5  # (...)x1x2
    out = torch.concat([x1y1, x2y2], dim=-2)
    return torch.reshape(out, orig_shape)


############################################################
#  preprocess
############################################################
def image_preprocess(image, bgr=True):
    image = torch.tensor(image).to(torch.float32)
    mean = cfg.PREPROC.PIXEL_MEAN
    std = np.asarray(cfg.PREPROC.PIXEL_STD)
    if bgr:
        mean = mean[::-1]
        std = std[::-1]
    image_mean_b = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + mean[0]
    image_mean_g = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + mean[1]
    image_mean_r = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + mean[2]
    image_invstd_b = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + 1.0 / std[0]
    image_invstd_g = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + 1.0 / std[1]
    image_invstd_r = torch.zeros(
        [cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1],
        dtype=torch.float32) + 1.0 / std[2]

    image_mean_b = torch.unsqueeze(image_mean_b, 0)
    image_mean_g = torch.unsqueeze(image_mean_g, 0)
    image_mean_r = torch.unsqueeze(image_mean_r, 0)
    image_invstd_b = torch.unsqueeze(image_invstd_b, 0)
    image_invstd_g = torch.unsqueeze(image_invstd_g, 0)
    image_invstd_r = torch.unsqueeze(image_invstd_r, 0)

    image_mean_b = torch.reshape(image_mean_b, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])
    image_mean_g = torch.reshape(image_mean_g, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])
    image_mean_r = torch.reshape(image_mean_r, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])
    image_invstd_b = torch.reshape(image_invstd_b, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])
    image_invstd_g = torch.reshape(image_invstd_g, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])
    image_invstd_r = torch.reshape(image_invstd_r, [
        1, cfg.PREPROC.TEST_LONG_EDGE_SIZE, cfg.PREPROC.TEST_LONG_EDGE_SIZE, 1
    ])

    image_mean = torch.concat([image_mean_b, image_mean_g, image_mean_r],
                              axis=3)
    image_invstd = torch.concat(
        [image_invstd_b, image_invstd_g, image_invstd_r], axis=3)

    image = image.permute(0, 2, 3, 1)
    image = (image - image_mean) * image_invstd

    image = image.permute(0, 3, 1, 2)
    return image


############################################################
#  Resnet Backbone
############################################################


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.padding1 = SamePad2d(kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.9)

        if stride == 2:
            self.conv2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   bias=False)
            self.padding2 = Pad2d((1, 0, 1, 0))
        else:
            self.conv2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=stride,
                                   bias=False)
            self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.9)

        self.padding3 = SamePad2d(kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(planes,
                               planes * 4,
                               kernel_size=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.padding1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.padding3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFPNBackbone(nn.Module):
    def __init__(self, architecture='resnet101'):
        super(ResNetFPNBackbone, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.inplanes = 64
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
        self.block = Bottleneck
        self.conv0 = nn.Sequential(
            Pad2d(pad_info=(3, 2, 3, 2)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
            Pad2d(pad_info=(1, 0, 1, 0)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.group0 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        self.group1 = self.make_layer(self.block,
                                      128,
                                      self.layers[1],
                                      stride=2)
        self.group2 = self.make_layer(self.block,
                                      256,
                                      self.layers[2],
                                      stride=2)
        self.group3 = self.make_layer(self.block,
                                      512,
                                      self.layers[3],
                                      stride=2)

    def forward(self, x):
        x = self.conv0(x)
        c2 = self.group0(x)
        c3 = self.group1(c2)
        c4 = self.group2(c3)
        c5 = self.group3(c4)
        return [c2, c3, c4, c5]

    def stages(self):
        return [self.group0, self.group1, self.group2, self.group3]

    def make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion,
                               eps=1e-5,
                               momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)


############################################################
#  fpn
############################################################


class FPN(nn.Module):
    def __init__(self, out_channels=cfg.FPN.NUM_CHANNEL):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.lateral_1x1_c2 = nn.Sequential(
            SamePad2d(kernel_size=1, stride=1),
            nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1),
        )
        self.lateral_1x1_c3 = nn.Sequential(
            SamePad2d(kernel_size=1, stride=1),
            nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1),
        )
        self.lateral_1x1_c4 = nn.Sequential(
            SamePad2d(kernel_size=1, stride=1),
            nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1),
        )
        self.lateral_1x1_c5 = nn.Sequential(
            SamePad2d(kernel_size=1, stride=1),
            nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1),
        )
        self.posthoc_3x3_p2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1),
        )
        self.posthoc_3x3_p3 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1),
        )
        self.posthoc_3x3_p4 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1),
        )
        self.posthoc_3x3_p5 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1),
        )

        self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, c2345):
        [c2, c3, c4, c5] = c2345
        lat_2 = self.lateral_1x1_c2(c2)
        lat_3 = self.lateral_1x1_c3(c3)
        lat_4 = self.lateral_1x1_c4(c4)
        lat_5 = self.lateral_1x1_c5(c5)

        lat_sum_5 = lat_5
        lat_sum_4 = lat_4 + F.interpolate(
            lat_sum_5, scale_factor=2, mode='nearest')
        lat_sum_3 = lat_3 + F.interpolate(
            lat_sum_4, scale_factor=2, mode='nearest')
        lat_sum_2 = lat_2 + F.interpolate(
            lat_sum_3, scale_factor=2, mode='nearest')

        p2 = self.posthoc_3x3_p2(lat_sum_2)
        p3 = self.posthoc_3x3_p3(lat_sum_3)
        p4 = self.posthoc_3x3_p4(lat_sum_4)
        p5 = self.posthoc_3x3_p5(lat_sum_5)
        p6 = self.maxpool_p6(p5)

        return [p2, p3, p4, p5, p6]


############################################################
#  rpn
############################################################
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        self.backbone_shapes = np.array([[
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride)),
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride))
        ] for stride in cfg.FPN.ANCHOR_STRIDES])
        self.conv0 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=cfg.FPN.NUM_CHANNEL,
                      out_channels=cfg.FPN.NUM_CHANNEL,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      bias=True), nn.ReLU(inplace=True))
        self.class_conv = nn.Sequential(
            nn.Conv2d(in_channels=cfg.FPN.NUM_CHANNEL,
                      out_channels=self.num_anchors,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), )
        self.box_conv = nn.Sequential(
            nn.Conv2d(in_channels=cfg.FPN.NUM_CHANNEL,
                      out_channels=self.num_anchors * 4,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), )

    def forward(self, image, features):
        multilevel_label_logits = []
        multilevel_box_logits = []
        for feature in features:
            hidden = self.conv0(feature)
            label_logits = self.class_conv(hidden)
            box_logits = self.box_conv(hidden)
            shp = box_logits.size()  # 1x(NAx4)xfHxfW
            label_logits = label_logits.permute(0, 2, 3, 1)
            label_logits = torch.reshape(
                label_logits, [1, shp[2] * shp[3] * self.num_anchors, 1])
            box_logits = box_logits.permute(0, 2, 3, 1)  # 1xfHxfWx(NAx4)
            box_logits = torch.reshape(
                box_logits,
                [1, shp[2] * shp[3] * self.num_anchors, 4])  # fHxfWxNAx4
            multilevel_label_logits.append(label_logits)
            multilevel_box_logits.append(box_logits)

        all_label_logits = torch.concat(multilevel_label_logits, dim=1)
        all_box_logits = torch.concat(multilevel_box_logits, dim=1)

        image_shape2d = image.shape[2:]
        backbone_shapes = np.array([[
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride)),
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride))
        ] for stride in cfg.FPN.ANCHOR_STRIDES])
        all_anchors_fpn = self.generate_pyramid_anchors(
            scales=tuple(cfg.RPN.ANCHOR_SIZES),
            ratios=tuple(cfg.RPN.ANCHOR_RATIOS),
            feature_shapes=backbone_shapes,
            feature_strides=tuple(cfg.FPN.ANCHOR_STRIDES),
            anchor_stride=1)

        all_anchors_concate = np.concatenate(all_anchors_fpn, axis=0)
        all_anchors_concate = torch.tensor(all_anchors_concate).to(
            image.device)

        proposal_boxes = self.generate_rpn_proposals(
            all_label_logits, all_box_logits, all_anchors_concate,
            image_shape2d, cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TEST_POST_NMS_TOPK)
        return proposal_boxes

############################################################
#  proposal
############################################################

    def clip_boxes(self, boxes, window):
        """
        boxes: [N, 4] each col is x1, y1, x2, y2
        window: [h, w]
        """
        boxes = torch.stack( \
            [boxes[:, 0].clamp(float(0), float(window[0])),
            boxes[:, 1].clamp(float(0), float(window[1])),
            boxes[:, 2].clamp(float(0), float(window[0])),
            boxes[:, 3].clamp(float(0), float(window[1]))], 1)
        return boxes

    def generate_anchors_keras(self, scales, ratios, shape, feature_stride,
                               anchor_stride):
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # base anchor为4*4大小
        size_ratios = feature_stride * feature_stride / ratios
        widths = np.round(np.sqrt(size_ratios))
        heights = np.round(widths * ratios)
        widths = widths * (scales / feature_stride)
        heights = heights * (scales / feature_stride)

        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride + (
            feature_stride - 1) / 2
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride + (
            feature_stride - 1) / 2
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        box_centers = np.stack([box_centers_y, box_centers_x],
                               axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths],
                             axis=2).reshape([-1, 2])

        boxes = np.concatenate([
            box_centers - 0.5 * (box_sizes - 1), box_centers + 0.5 *
            (box_sizes - 1)
        ],
                               axis=1)
        boxes[:, [2, 3]] += 1
        boxes = boxes[:, [1, 0, 3, 2]].astype(np.float32)
        return boxes

    def generate_pyramid_anchors(self, scales, ratios, feature_shapes,
                                 feature_strides, anchor_stride):
        anchors = []
        for i in range(len(scales)):
            anchors.append(
                self.generate_anchors_keras(scales[i], ratios,
                                            feature_shapes[i],
                                            feature_strides[i], anchor_stride))
        return anchors

    def anchors_concate(self, ):
        backbone_shapes = np.array([[
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride)),
            int(math.ceil(cfg.PREPROC.TEST_LONG_EDGE_SIZE / stride))
        ] for stride in cfg.FPN.ANCHOR_STRIDES])
        all_anchors_fpn = self.generate_pyramid_anchors(
            scales=tuple(cfg.RPN.ANCHOR_SIZES),
            ratios=tuple(cfg.RPN.ANCHOR_RATIOS),
            feature_shapes=backbone_shapes,
            feature_strides=tuple(cfg.FPN.ANCHOR_STRIDES),
            anchor_stride=1)

        all_anchors_concate = np.concatenate(all_anchors_fpn, axis=0)
        all_anchors_concate = np.broadcast_to(all_anchors_concate, (1, ) +
                                              all_anchors_concate.shape)
        return all_anchors_concate

    def generate_rpn_proposals(self,
                               scores,
                               boxes_logits,
                               anchors,
                               img_shape,
                               pre_nms_topk,
                               post_nms_topk=None):
        if post_nms_topk is None:
            post_nms_topk = pre_nms_topk
        scores = scores.squeeze()
        boxes_logits = boxes_logits.squeeze()
        topk = min(pre_nms_topk, anchors.shape[0])
        scores, order = torch.topk(scores, k=topk, dim=0)

        deltas = boxes_logits[order, :].squeeze(1)
        pre_nms_anchors = anchors[order, :].squeeze(1)
        boxes = decode_bbox_target(deltas, pre_nms_anchors)
        boxes = clip_boxes(boxes, img_shape)

        def nms(boxes, scores):
            scores = scores.squeeze()
            idx = ops.nms(boxes, scores, cfg.RPN.PROPOSAL_NMS_THRESH)
            proposals = torch.index_select(boxes, 0, idx)
            if post_nms_topk > idx.size()[0]:
                padding = max(post_nms_topk - idx.size()[0], 0)
                proposals = F.pad(proposals, (0, 0, 0, padding))
            else:
                proposals = proposals[:post_nms_topk]
            return proposals

        proposal_boxes = nms(boxes, scores)

        return proposal_boxes


############################################################
#  ROI Aligns
############################################################


def crop_and_resize(image, boxes, box_ind, crop_size, name, pad_border=True):
    assert isinstance(crop_size, int), crop_size
    if pad_border:
        image = image.squeeze(0)
        image = torch.concat([image[:, :1, :], image, image[:, -1:, :]], dim=1)
        image = torch.concat([image[:, :, :1], image, image[:, :, -1:]], dim=2)
        image = image.unsqueeze(0)
        boxes = boxes + 1

    box_ind = box_ind.unsqueeze(1)

    boxes = torch.concat([box_ind, boxes], dim=1)
    ret = ops.roi_align(image,
                        boxes, (crop_size, crop_size),
                        sampling_ratio=1,
                        aligned=True)  #TODO  aligned=True
    return ret


def multilevel_roi_align_v2(rcnn_boxes, features, resolution,
                            name):  # resolution = 7
    assert len(features) == 4, features
    x1, y1, x2, y2 = torch.chunk(rcnn_boxes, 4, dim=1)
    h = y2 - y1
    w = x2 - x1
    sqrtarea = torch.sqrt(h * w)
    roi_level = 4 + torch.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))
    roi_level = roi_level.type(torch.IntTensor)
    roi_level = torch.minimum(
        torch.tensor(5, dtype=torch.int32),
        torch.maximum(torch.tensor(2, dtype=torch.int32), roi_level))
    roi_level = torch.squeeze(roi_level)
    device = rcnn_boxes.device
    pooleds = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = torch.where(roi_level == level)[0].to(device)
        level_boxes = rcnn_boxes[ix]
        box_indices = torch.zeros(level_boxes.size()[0],
                                  dtype=torch.float32).to(device)
        level_boxes = level_boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
        pooled = crop_and_resize(features[i], level_boxes, box_indices,
                                 resolution * 2, name)
        pooleds.append(pooled)
        box_to_level.append(ix)

    pooleds = torch.cat(pooleds, dim=0)

    box_to_level = torch.cat(box_to_level, dim=0)

    _, box_to_level = torch.sort(box_to_level)
    pooleds = pooleds[box_to_level, :, :]
    pooleds = pooleds.unsqueeze(0)
    return pooleds


############################################################
#  Rcnn
############################################################


def decoded_output_boxes_class_agnostic(proposals, box_logits, reg_weights):
    box_logits = torch.reshape(box_logits, [-1, 4])
    decoded = decode_bbox_target(box_logits / reg_weights, proposals)
    return decoded


class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.batch_flatten(x)
        hidden = self.fc(x)
        return hidden

    def batch_flatten(self, x):
        return torch.flatten(x, start_dim=1)


class FastrcnnOutputs(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 class_agnostic_regression=False):
        super(FastrcnnOutputs, self).__init__()
        num_classes_for_box = 1 if class_agnostic_regression else num_classes
        self.classification = FullyConnected(in_channels, num_classes)
        self.box_regression = FullyConnected(in_channels,
                                             num_classes_for_box * 4)
        self.box_cos = FullyConnected(in_channels, num_classes_for_box)
        self.box_sin = FullyConnected(in_channels, num_classes_for_box)

    def forward(self, feature):
        classification = self.classification(feature)
        box_regression = self.box_regression(feature)
        box_cos = self.box_cos(feature)
        box_sin = self.box_sin(feature)
        return classification, box_regression, box_cos, box_sin


# box detector layer
def fastrcnn_predictions_v2(proposals, box_logits, label_logits,
                            box_cos_logits, box_sin_logits, image_shape2d,
                            cascade_stage_index):
    proposals = torch.reshape(proposals, [cfg.RPN.TEST_POST_NMS_TOPK, 4])
    # Each box belongs to only one category based on the highest score
    class_ids = torch.ones(label_logits.size()[0],
                           dtype=torch.int64).to(label_logits.device)
    class_scores = label_logits[:, 1]  # choose class_id=1
    deltas_specific = box_logits
    reg_weights = torch.tensor(
        cfg.CASCADE.BBOX_REG_WEIGHTS[cascade_stage_index],
        dtype=torch.float32).to(box_logits.device)
    refined_rois = decode_bbox_target(deltas_specific / reg_weights, proposals)

    refined_rois = clip_boxes(refined_rois, image_shape2d)
    box_cos_logits = torch.reshape(box_cos_logits,
                                   [cfg.RPN.TEST_POST_NMS_TOPK])
    box_sin_logits = torch.reshape(box_sin_logits,
                                   [cfg.RPN.TEST_POST_NMS_TOPK])

    # calculate area
    x1, y1, x2, y2 = torch.chunk(refined_rois, 4, dim=1)
    h = y2 - y1
    w = x2 - x1
    sqrtarea = torch.sqrt(h * w)
    sqrtarea = torch.squeeze(sqrtarea, 1)
    # Filter out background boxes
    keep = torch.nonzero(class_ids).squeeze(1)

    # Filter out low confidence boxes and padding boxes(area=0)
    if cfg.TEST.RESULT_SCORE_THRESH:  # 0.3
        score_keep = torch.nonzero(
            torch.gt(class_scores,
                     cfg.TEST.RESULT_SCORE_THRESH)).to(sqrtarea.device)
        area_keep = torch.nonzero(sqrtarea)
        score_keep = torch.squeeze(score_keep).to(torch.int64)
        area_keep = torch.squeeze(area_keep).to(torch.int64)
        conf_keep = intersection(score_keep, area_keep)
        conf_keep = torch.squeeze(conf_keep)
        keep = intersection(keep, conf_keep)
    pre_nms_class_ids = class_ids[keep]
    unique_pre_nms_class_ids = torch.unique(pre_nms_class_ids)  # 1

    def nms_keep_map(class_id):
        ixs = torch.nonzero(torch.eq(pre_nms_class_ids, class_id)).squeeze(1)
        class_keep = ops.nms(refined_rois[ixs],
                             class_scores[ixs],
                             iou_threshold=cfg.TEST.FRCNN_NMS_THRESH)
        class_keep = class_keep[:cfg.TEST.RESULTS_PER_IM]
        return class_keep

    # 2. Map over class IDs
    nms_keep = nms_keep_map(unique_pre_nms_class_ids)
    keep = intersection(keep, nms_keep)
    roi_count = cfg.TEST.RESULTS_PER_IM  # 300
    class_scores_keep = class_scores[keep]
    num_keep = min(class_scores_keep.size()[0], roi_count)
    top_ids = torch.topk(class_scores_keep, k=num_keep, sorted=True)[1]

    keep = keep[top_ids]
    detections = torch.concat([
        refined_rois[keep], class_ids[keep].unsqueeze(1),
        class_scores[keep].unsqueeze(1), box_cos_logits[keep].unsqueeze(1),
        box_sin_logits[keep].unsqueeze(1)
    ],
                              dim=1)

    gap = cfg.TEST.RESULTS_PER_IM - detections.size()[0]
    detections = F.pad(detections, (0, 0, 0, gap))

    return detections


# maskrcnn layer
class MaskrcnnUpXconvHead(nn.Module):
    def __init__(self, num_category):
        super(MaskrcnnUpXconvHead, self).__init__()
        self.fcn_conv = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(cfg.MRCNN.HEAD_DIM,
                      cfg.MRCNN.HEAD_DIM,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(cfg.MRCNN.HEAD_DIM,
                      cfg.MRCNN.HEAD_DIM,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(cfg.MRCNN.HEAD_DIM,
                      cfg.MRCNN.HEAD_DIM,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(cfg.MRCNN.HEAD_DIM,
                      cfg.MRCNN.HEAD_DIM,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            SamePad2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(cfg.MRCNN.HEAD_DIM,
                               cfg.MRCNN.HEAD_DIM,
                               kernel_size=2,
                               stride=2),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            SamePad2d(kernel_size=1, stride=1),
            nn.Conv2d(cfg.MRCNN.HEAD_DIM,
                      num_category,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, feature, num_convs=4):
        l = feature
        # c2's MSRAFill is fan_out'
        l = self.fcn_conv(l)
        l = self.deconv(l)
        l = self.conv(l)
        return l


class Rcnn(nn.Module):
    def __init__(self, decode_outputboxes=False, reg_weights=None):
        super(Rcnn, self).__init__()
        self.avg_pool = nn.AvgPool2d([2, 2], [2, 2], padding=0, ceil_mode=True)
        channels = [cfg.MRCNN.HEAD_DIM * 7 * 7, cfg.FPN.FRCNN_FC_HEAD_DIM]
        self.relu = nn.ReLU(inplace=True)
        self.decode_outputboxes = decode_outputboxes
        self.reg_weights = reg_weights
        self.fastrcnn_head_fc6 = FullyConnected(channels[0], channels[1])
        self.fastrcnn_head_fc7 = FullyConnected(channels[1], channels[1])
        self.fastrcnn_outputs = FastrcnnOutputs(channels[1],
                                                cfg.DATA.NUM_CLASS,
                                                class_agnostic_regression=True)

    def forward(self, image_shape2d, features, proposals):
        roi_feature_fastrcnn = multilevel_roi_align_v2(
            proposals, features[:4], 7, 'multilevel_roi_align_rcnn')
        roi_feature_fastrcnn = torch.reshape(
            roi_feature_fastrcnn,
            [cfg.RPN.TEST_POST_NMS_TOPK, cfg.FPN.NUM_CHANNEL, 14, 14])
        roi_feature_fastrcnn = self.avg_pool(roi_feature_fastrcnn)

        head_feature = self.fastrcnn_head_fc6(roi_feature_fastrcnn)
        head_feature = self.relu(head_feature)
        head_feature = self.fastrcnn_head_fc7(head_feature)
        head_feature = self.relu(head_feature)

        label_logits, box_logits, box_cos_logits, box_sin_logits = self.fastrcnn_outputs(
            head_feature)
        label_scores = self.decoded_output_scores(label_logits)
        if self.decode_outputboxes:
            proposals = self.decode_output_boxes(image_shape2d, proposals,
                                                 box_logits)

        return label_scores, label_logits, box_logits, box_cos_logits, box_sin_logits, proposals

    def decoded_output_scores(self, label_logits):
        """ Returns: N x #class scores, summed to one for each box."""
        return torch.softmax(label_logits, dim=1)

    def decode_output_boxes(self, image_shape2d, proposals, box_logits):
        proposals = torch.reshape(proposals, [cfg.RPN.TEST_POST_NMS_TOPK, 4])
        reg_weights = self.reg_weights
        refined_boxes = decoded_output_boxes_class_agnostic(
            proposals, box_logits, reg_weights)
        proposals = clip_boxes(refined_boxes, image_shape2d)
        return proposals


class CascadeRcnnStage(nn.Module):
    def __init__(self, device):
        super(CascadeRcnnStage, self).__init__()
        self.device = device
        REG_W1 = torch.tensor(cfg.CASCADE.BBOX_REG_WEIGHTS[0],
                              dtype=torch.float32,
                              device=device)
        REG_W2 = torch.tensor(cfg.CASCADE.BBOX_REG_WEIGHTS[1],
                              dtype=torch.float32,
                              device=device)
        self.stage1 = Rcnn(decode_outputboxes=True, reg_weights=REG_W1)
        self.stage2 = Rcnn(decode_outputboxes=True, reg_weights=REG_W2)
        self.stage3 = Rcnn()
        self.avg_pool = nn.AvgPool2d([2, 2], [2, 2], padding=0, ceil_mode=True)
        self.maskrcnn_upXconv_head = MaskrcnnUpXconvHead(cfg.DATA.NUM_CATEGORY)

    def forward(self, image, features, proposals):
        image_shape2d = image.size()[2:]
        label_scores_1, _, box_logits_1, _, _, proposals1 = self.stage1(
            image_shape2d, features, proposals)
        label_scores_2, _, _, _, _, proposals2 = self.stage2(
            image_shape2d, features, proposals1)
        label_scores_3, _, box_logits_3, box_cos_logits_3, box_sin_logits_3, proposals3 = self.stage3(
            image_shape2d, features, proposals2)
        label_scores = (label_scores_1 + label_scores_2 + label_scores_3) / 3
        detections = fastrcnn_predictions_v2(proposals3, box_logits_3,
                                             label_scores, box_cos_logits_3,
                                             box_sin_logits_3, image_shape2d,
                                             2)
        final_boxes = detections[:, 0:4]
        roi_feature_maskrcnn = multilevel_roi_align_v2(
            final_boxes, features[:4], 14, 'multilevel_roi_align_mask')
        roi_feature_maskrcnn = torch.reshape(
            roi_feature_maskrcnn,
            [cfg.TEST.RESULTS_PER_IM, cfg.FPN.NUM_CHANNEL, 28, 28])

        roi_feature_maskrcnn = self.avg_pool(roi_feature_maskrcnn)
        mask_logits = self.maskrcnn_upXconv_head(roi_feature_maskrcnn, 4)
        final_masks = torch.sigmoid(mask_logits)

        return detections, final_masks


class Maskrcnn(nn.Module):
    def __init__(self, device='cpu'):
        super(Maskrcnn, self).__init__()
        self.device = device
        finalize_configs()
        self.resnet_fpn_backbone = ResNetFPNBackbone()
        self.fpn_model = FPN()
        self.rpn = RPN()
        self.cascade_rcnn_stage = CascadeRcnnStage(device)

    def forward(self, x):
        with torch.no_grad():
            image = x
            c2345 = self.resnet_fpn_backbone(image)
            p23456 = self.fpn_model(c2345)
            proposals = self.rpn(image, p23456)
            detections, final_masks = self.cascade_rcnn_stage(
                image, p23456, proposals)
            return detections, final_masks


CLASSES = ['印章', '图片', '标题', '段落', '表格', '页眉', '页码', '页脚', '其他']
CLASS_MAP = {(k + 1): v for k, v in enumerate(CLASSES)}


def clip_boxes_v1(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


class LayoutMrcnnPt(object):
    """
    Layout
    """
    def __init__(self, model_path, **kwargs):
        devices = kwargs.get('devices').split(',')
        self.default_device = f'cuda:{devices[0]}'

        self.precision = kwargs.get('precision', 'fp32')

        self.model = Maskrcnn(device=self.default_device)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        if self.precision == 'fp16':
            self.model.half()

        self.model.eval()
        self.model.to(self.default_device)

        self.scale_list = np.array([600, 800, 1000, 1200, 1400, 1600])

    def infer(self, img, longer_edge_size=0):
        orig_shape = img.shape[:2]

        # Step 1. prep
        h = orig_shape[0]
        w = orig_shape[1]
        if longer_edge_size == 0:
            side0 = max(h, w)
            distance = np.abs(self.scale_list - side0)
            longer_edge_size = self.scale_list[np.argmin(distance)]
        scale = longer_edge_size * 1.0 / max(h, w)
        if h > w:
            newh, neww = longer_edge_size, scale * w
        else:
            newh, neww = scale * h, longer_edge_size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        resized_img = cv2.resize(img, dsize=(neww, newh))
        scale = np.sqrt(
            resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] /
            img.shape[1],
            dtype=np.float32,
        )

        resized_img0 = np.zeros([1600, 1600, 3], dtype=np.float32)
        resized_img0[:newh, :neww, :] = resized_img
        resized_img = resized_img0

        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        mean = mean[::-1]
        std = std[::-1]

        image_mean = np.array(mean)
        image_std = np.array(std)
        resized_img = (resized_img - image_mean) / image_std

        resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)
        resized_img = np.transpose(resized_img, [0, 3, 1, 2])

        print('resized_img', resized_img.shape)
        # Step 2. infer
        start = time.time()
        inp_tensor = torch.from_numpy(resized_img.copy()).to(
            self.default_device)
        with torch.no_grad():
            detections, masks = self.model(inp_tensor)

        detections = detections.cpu().numpy()
        masks = masks.cpu().numpy()

        (pre_boxes, pre_scores, pre_boxes_cos, pre_boxes_sin, pre_masks,
         pre_labels) = (detections[:, :4], detections[:, 5], detections[:, 6],
                        detections[:, 7], masks, detections[:, 4])

        print('pre_boxes', pre_boxes.shape, pre_boxes)
        print('pre_scores', pre_scores)
        print('pre_labels', pre_labels)

        end = time.time()
        print('[Layout Analysis] %d ms per frame' % ((end - start) * 1000))

        # Step 3. post
        pre_boxes = pre_boxes / scale
        pre_boxes = clip_boxes_v1(pre_boxes, orig_shape)

        boxes = pre_boxes.astype(np.int32)
        new_boxes = np.zeros((boxes.shape[0], 8), dtype=np.int32)
        new_boxes[:, 0] = boxes[:, 0]  # x1
        new_boxes[:, 1] = boxes[:, 1]  # y1
        new_boxes[:, 2] = boxes[:, 2]  # x2
        new_boxes[:, 3] = boxes[:, 1]  # y1
        new_boxes[:, 4] = boxes[:, 2]  # x2
        new_boxes[:, 5] = boxes[:, 3]  # y2
        new_boxes[:, 6] = boxes[:, 0]  # x1
        new_boxes[:, 7] = boxes[:, 3]  # y2
        scores = pre_scores.astype(np.float32)
        labels = pre_labels.astype(np.int32)
        return boxes, scores, labels

    def predict(self, img, **kwargs):
        img = base64.b64decode(img)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        boxes, scores, labels = self.infer(img)
        res = []
        for i, box in enumerate(boxes):
            tmp_dict = {}
            score = float(scores[i])
            label = int(labels[i])
            category_name = CLASS_MAP[label]
            tmp_dict['category_id'] = label
            tmp_dict['category_name'] = category_name
            tmp_dict['bbox'] = []
            tmp_dict['bbox'].extend(box.tolist())
            tmp_dict['score'] = score
            res.append(tmp_dict)
        return res
