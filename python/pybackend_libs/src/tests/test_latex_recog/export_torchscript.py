# flake8: noqa
import argparse
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from cnstd.yolov7.consts import CATEGORY_DICT
from cnstd.yolov7.general import check_img_size
from cnstd.yolov7.layout_analyzer import attempt_load
from cnstd.yolov7.yolo import Model as YoloModel


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
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction, conf_thres=0.25, iou_thres=0.45, nc=2,
        classes=None, agnostic=False, multi_label=False,
        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    print('prediction.shape', prediction.size(), prediction.sum())
    # nc = prediction.shape[2] - 5  # number of classes
    print('conf_thres', conf_thres)
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * 1
    for xi, x in enumerate(prediction):  # image index, image inference
        print('xi', xi, x)
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        print('x', x)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
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

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output[0]


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, nc=2):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.iou_thres = iou_thres
        self.score_thres = score_thres
        self.nc = 2
        self.model = model.to(device)
        # self.model.model[-1].concat = True

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        print('e2e.x', x.size())
        x = non_max_suppression(x,
            conf_thres=self.score_thres, iou_thres=self.iou_thres, nc=self.nc)
        return x


def export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    opt = parser.parse_args()

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)

    t = time.time()

    # Load PyTorch model
    device = torch.device('cpu')
    model_name = 'yolov7'
    categories = CATEGORY_DICT['mfd']
    cfg_fp = '/usr/local/lib/python3.8/dist-packages/cnstd/yolov7/yolov7-mfd.yaml'
    model = attempt_load(categories, opt.weights, cfg_fp, map_location=device)  # load FP32 model
    labels = model.names
    print(labels)

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    with open('data/x1.npy', 'rb') as f:
        x = np.load(f)

    img = torch.from_numpy(x).to(device)
    # img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection
    print('---img', img.size(), img.dtype)

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        # if isinstance(m, Conv):  # assign export-friendly activations
        #     if isinstance(m.act, nn.Hardswish):
        #         m.act = Hardswish()
        #     elif isinstance(m.act, nn.SiLU):
        #         m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)

    print(model.model[-1])

    model.model[-1].concat = True
    model.model[-1].export = not opt.grid

    e2e_model = End2End(model, score_thres=opt.conf_thres, iou_thres=opt.iou_thres, nc=2)
    e2e_model.eval()

    y = e2e_model(img)  # dry run
    # y = model(img)  # dry run
    print('y', y.size(), y.numpy())
    # return
    # if opt.include_nms:
    #     model.model[-1].include_nms = True
    #     y = None

    # TorchScript export
    print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = './data/graph.torchscript.pt'
        ts = torch.jit.trace(e2e_model, img, strict=False)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    return

    # return
    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = './data/graph.onnx'
        model.eval()
        output_names = ['output']
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {
                'images': {2: 'height', 3: 'width'},
                'output': {1: 'x1'}
            }

        # if opt.dynamic_batch:
        #     opt.batch_size = 'batch'
        #     dynamic_axes = {
        #         'images': {
        #             0: 'batch',
        #         }, }
        #     if opt.end2end and opt.max_wh is None:
        #         output_axes = {
        #             'num_dets': {0: 'batch'},
        #             'det_boxes': {0: 'batch'},
        #             'det_scores': {0: 'batch'},
        #             'det_classes': {0: 'batch'},
        #         }
        #     else:
        #         output_axes = {
        #             'output': {0: 'batch'},
        #         }
        #     dynamic_axes.update(output_axes)

        if opt.grid:
            if opt.end2end:
                print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            # else:
            #     model.model[-1].concat = True

        torch.onnx.export(
          model,
          img,
          f,
          verbose=True,
          opset_version=12,
          input_names=['images'],
          output_names=output_names,
          dynamic_axes=dynamic_axes)

    except Exception as e:
        print('e', e)


export()
