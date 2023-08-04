import argparse
import os

import numpy as np
import Polygon as plg


class eval_IOU(object):
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh

    def __call__(self, gt_boxes_list, boxes_list):
        detMatched_list = []
        numDetCare_list = []
        numGtCare_list = []
        for i in range(len(gt_boxes_list)):
            gt_boxes = gt_boxes_list[i]
            boxes = boxes_list[i]
            detMatched, numDetCare, numGtCare = self.eval(gt_boxes, boxes)
            detMatched_list.append(detMatched)
            numDetCare_list.append(numDetCare)
            numGtCare_list.append(numGtCare)
        matchedSum = np.sum(np.array(detMatched_list))
        numGlobalCareDet = np.sum(np.array(numDetCare_list))
        numGlobalCareGt = np.sum(np.array(numGtCare_list))
        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else \
            2*methodRecall*methodPrecision / (methodRecall+methodPrecision)
        return methodPrecision, methodRecall, methodHmean

    def eval(self, gt_boxes, boxes):
        detMatched = 0
        numDetCare = 0
        numGtCare = 0
        if gt_boxes is None:
            return 0, 0, 0

        gtPols = []
        detPols = []
        detDontCarePolsNum = []
        iouMat = np.empty([1, 1])
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            gtPols.append(self.polygon_from_box(gt_box))

        if boxes is None:
            return 0, 0, len(gtPols)

        for box in boxes:
            detPol = self.polygon_from_box(box)
            detPols.append(detPol)

        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            pairs = []
            detMatchedNums = []
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum,
                           detNum] = self.get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_thresh:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        numGtCare = len(gtPols)
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        return detMatched, numDetCare, numGtCare

    def get_intersection(self, pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def get_union(self, pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - self.get_intersection(pD, pG)

    def get_intersection_over_union(self, pD, pG):
        try:
            return self.get_intersection(pD, pG) / self.get_union(pD, pG)
        except Exception:
            return 0

    def polygon_from_box(self, box):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(box[0][0])
        resBoxes[0, 4] = int(box[0][1])
        resBoxes[0, 1] = int(box[1][0])
        resBoxes[0, 5] = int(box[1][1])
        resBoxes[0, 2] = int(box[2][0])
        resBoxes[0, 6] = int(box[2][1])
        resBoxes[0, 3] = int(box[3][0])
        resBoxes[0, 7] = int(box[3][1])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


def main(gt_dir, pred_dir, iou=0.7):
    det_eval = eval_IOU(iou_thresh=iou)
    names = load_files(gt_dir)
    boxes_list = []
    gt_boxes_list = []
    for name in names:
        boxes = []
        if os.path.exists(os.path.join(pred_dir, name)):
            for line in open(os.path.join(pred_dir, name)):
                line = line.strip()
                lines = line.split(',')
                lines = list(map(float, lines))
                box = np.array(lines).reshape([4, 2])
                boxes.append(np.int0(np.round(box)))
                # boxes.append(np.int0(box))

        boxes = np.array(boxes, dtype=np.int32)
        boxes_list.append(boxes)

        gt_boxes = []
        for line in open(os.path.join(gt_dir, name)):
            line = line.strip()
            lines = line.split(',')[:8]
            lines = list(map(float, lines))
            box = np.array(lines).reshape([4, 2])
            gt_boxes.append(np.int0(np.round(box)))
            # gt_boxes.append(np.int0(box))

        gt_boxes = np.array(gt_boxes, dtype=np.int32)
        gt_boxes_list.append(gt_boxes)
    precision, recall, hmean = det_eval(gt_boxes_list, boxes_list)
    return precision, recall, hmean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir',
                        default=None,
                        type=str,
                        required=True,
                        help='gt txts.')
    parser.add_argument('--pred_dir',
                        default=None,
                        type=str,
                        required=True,
                        help='predict txts.')
    parser.add_argument('--iou', default=0.7, type=float)
    args = parser.parse_args()
    precision, recall, hmean = main(args.gt_dir, args.pred_dir, args.iou)
    print('precision:{}, recall:{}, hmean:{}'.format(precision, recall, hmean))
