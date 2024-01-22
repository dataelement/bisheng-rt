# flake8: noqa

import base64
import os
import time

import Levenshtein
import numpy as np
import requests


def recog_infer(image_file):
    RT_EP = os.environ.get('RT_EP', '192.168.106.20:19001')
    ep_prefix = f'http://{RT_EP}'
    headers = {'Content-type': 'application/json'}
    name = 'formula_recog_v1'
    infer_ep = f'{ep_prefix}/v2.1/models/{name}/infer'

    # image_file = '/public/bisheng/latex_data/formula.jpg'
    payload = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }

    resp = requests.post(url = infer_ep, json=payload, headers=headers).json()
    return resp['result']


def test():
    dataset_dir = '/public/bisheng/latex_data/formula_test_v1'
    label = dataset_dir + '/label.txt.txt'
    gt = dict([v.strip().split(' ', 1) for v in open(label).readlines()])

    files = os.listdir(dataset_dir)
    out = []
    for f in files:
        if f.endswith('.png'):
            image_file = os.path.join(dataset_dir, f)
            tic = time.time()
            result = recog_infer(image_file)
            print('result', f, result, time.time() - tic)
            out.append((f, result))

    ed_sum = 0
    N = 0
    for f, pred in out:
        ed = Levenshtein.distance(pred, gt[f])
        ed_sum += ed
        N += len(gt[f])

    cer = ed_sum / N
    print('---test cer', cer)


def test2():
    dataset_dir = '/public/bisheng/latex_data/formula_test_v1'
    label = dataset_dir + '/label.txt.txt'
    gt = dict([v.strip().split(' ', 1) for v in open(label).readlines()])

    pred_file = '/public/bisheng/latex_data/formula_test_v1_pred_ori_v1.json'

    ratios = []
    import json
    preds = json.load(open(pred_file))
    ed_sum = 0
    N = 0
    for f, pred in preds:
        ed = Levenshtein.distance(pred, gt[f])
        ed_sum += ed
        N += len(gt[f])

    cer = ed_sum / N
    print('---test cer', cer)


test()
# test2()
