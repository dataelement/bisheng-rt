import base64
import time

import numpy as np
from pybackend_libs.dataelem.model.layout.layout_mrcnn import LayoutMrcnn
from pybackend_libs.dataelem.model.layout.mrcnn_pt import LayoutMrcnnPt


def test_layout_mrcnn():
    params = {
        'model_path': '/home/hanfeng/models/layout_mrcnn/freeze.pb',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    # params = {
    #     'model_path': '/home/hanfeng/models/layout_mrcnn/freeze_fp16.pb',
    #     'devices': '6',
    #     'gpu_memory': 3,
    #     'precision': 'fp16'
    # }

    model = LayoutMrcnn(**params)

    test_image = '../data/maoxuan_mulu.jpg'
    b64data = base64.b64encode(open(test_image, 'rb').read())
    inp = {'img': b64data}
    outp = model.predict(inp)
    # outp = model.predict(inp)
    # outp = model.predict(inp)

    print(outp)


def test_layout_mrcnn_pt():
    # params = {
    #     'model_path': '/home/hanfeng/models/layout_mrcnn/freeze.pb',
    #     'devices': '6',
    #     'gpu_memory': 3,
    #     'precision': 'fp32'
    # }

    params = {
        'model_path': '/home/hanfeng/models/layout_mrcnn/model.pth',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = LayoutMrcnnPt(**params)

    test_image = '../data/maoxuan_mulu.jpg'
    b64data = base64.b64encode(open(test_image, 'rb').read())
    inp = {'img': b64data}
    outp = model.predict(inp)
    # outp = model.predict(inp)
    # outp = model.predict(inp)

    print(outp)


# test_layout_mrcnn_pt()
test_layout_mrcnn()
