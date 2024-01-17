# flake8: noqa
import base64
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torchvision


# follow WongKinYiu/yolov7 projects
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction: (1, 30492, 7)
    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # nc = prediction.shape[2] - 5  # number of classes
    nc = 2
    x = prediction
    if not x.shape[0]:
        return x

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # Compute conf
    # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
    # so there is no need to multiplicate.
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

    return x[i]


# Codes from https://github.com/WongKinYiu/yolov7/tree/main
#  https://github.com/breezedeus/CnSTD/tree/master
def letterbox(
      img, new_shape=(640, 640), color=(114, 114, 114),
      auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add border
    img = cv2.copyMakeBorder(
      img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # boxes[:, 0].clamp_(0, img_shape[1])  # x1
    # boxes[:, 1].clamp_(0, img_shape[0])  # y1
    # boxes[:, 2].clamp_(0, img_shape[1])  # x2
    # boxes[:, 3].clamp_(0, img_shape[0])  # y2
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
          img1_shape[0] / img0_shape[0],
          img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xyxy24p(x, ret_type=np.array):
    xmin, ymin, xmax, ymax = [float(_x) for _x in x]
    out = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    if ret_type is not None:
        return ret_type(out).reshape((4, 2))
    return out


class LatexDetection(object):
    """A yolov7 based detector.

    model trained by https://github.com/breezedeus/CnSTD/tree/master.
    algorithm provided by https://github.com/WongKinYiu/yolov7/tree/main
    """
    def __init__(self, **kwargs):
        sig = {
            'inputs': ['image'],
            'outputs': ['output']
        }

        self.conf_thres = kwargs.get('conf_thres', 0.25)
        self.iou_thres = kwargs.get('iou_thres', 0.45)
        self.categories = {0: 'embedding', 1: 'isolated'}
        self.has_graph_executor = kwargs.get('has_graph_executor', False)
        if not self.has_graph_executor:
            from pybackend_libs.dataelem.framework.pt_graph import PTGraph
            import torchvision  # noqa: F401
            devices = kwargs.get('devices')
            used_device = devices.split(',')[0]
            self.graph = PTGraph(sig, used_device, **kwargs)
        else:
            self.xs = ['INPUT__0']
            self.ys = ['OUTPUT__0']

    def predict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        b64_image = context.get('b64_image')
        img = base64.b64decode(b64_image)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        prep_outs = self.preprocess(context, [img])
        # with open('/public/bisheng/latex_data/eng2.npy', 'wb') as f:
        #     np.save(f, prep_outs[0])
        if not self.has_graph_executor:
            infer_outs = self.graph.run(prep_outs, ret_type='tensor')
        else:
            graph_executor = context.get('graph_executor')
            infer_outs = graph_executor.run(
                self.ys, self.xs, prep_outs, ret_type='tensor')

        return self.postprocess(context, infer_outs)

    def preprocess(self,
                   context: Dict[str, Any],
                   inputs: List[Any]) -> List[Any]:
        resized_shape = context.get('resized_shape', 704)
        stride = context.get('stride', 32)
        img0 = inputs[0]
        context.update(orig_shape=img0.shape)

        img_size = (resized_shape, resized_shape)
        img = letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        context.update(new_shape=img.shape)
        return [img]

    def _expand(self, xyxy, box_margin, shape):
        xmin, ymin, xmax, ymax = [float(_x) for _x in xyxy]
        xmin = max(0, xmin - box_margin)
        ymin = max(0, ymin - box_margin)
        xmax = min(shape[1], xmax + box_margin)
        ymax = min(shape[0], ymax + box_margin)
        return [xmin, ymin, xmax, ymax]

    def postprocess(
          self, context: Dict[str, Any], inputs: List[Any]) -> Dict[str, Any]:
        det = non_max_suppression(
            inputs[0].cpu(), self.conf_thres, self.iou_thres).numpy()

        box_margin = context.get('box_margin', 2)
        orig_shape = context['orig_shape']
        new_shape = context['new_shape']

        if det[:, :4].shape[0] == 0:
            return {'result': []}

        one_out = []
        det[:, :4] = scale_coords(
          new_shape[2:], det[:, :4], orig_shape).round()

        # sort by conf by decreasing mode
        det = det[det[:, 4].argsort()[::-1]]
        for *xyxy, conf, label in det:
            xyxy = self._expand(xyxy, box_margin, orig_shape)
            one_out.append(
                {
                    'type': self.categories[int(label)],
                    'box': xyxy24p(xyxy, ret_type=None),
                    'score': float(conf),
                }
            )
        return {'result': one_out}
