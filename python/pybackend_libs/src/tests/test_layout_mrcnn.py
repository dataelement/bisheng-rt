import base64

# import numpy as np
from pybackend_libs.dataelem.model.layout.layout_mrcnn import LayoutMrcnn
from pybackend_libs.dataelem.model.layout.mrcnn_pt import LayoutMrcnnPt

# import time


def test_elem_layout_v1_fp16():
    params = {
        'model_path': '/home/public/models/elem_layout_v1/freeze_fp16.pb',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp16'
    }

    model = LayoutMrcnn(**params)

    test_image = '../data/maoxuan_mulu.jpg'
    b64data = base64.b64encode(open(test_image, 'rb').read())
    inp = {'b64_image': b64data}
    outp = model.predict(inp)
    print(outp)


def test_elem_layout_v1():
    params = {
        'model_path': '/home/public/models/elem_layout_v1/freeze.pb',
        'devices': '6',
        'gpu_memory': 4,
        'precision': 'fp32'
    }

    model = LayoutMrcnn(**params)

    test_image = '../data/maoxuan_mulu.jpg'
    b64data = base64.b64encode(open(test_image, 'rb').read())
    inp = {'b64_image': b64data}
    outp = model.predict(inp)
    print(outp)


def test_elem_layout_v1_pt():
    params = {
        'model_path': '/home/public/models/elem_layout_v1/model.pth',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = LayoutMrcnnPt(**params)

    test_image = '../data/maoxuan_mulu.jpg'
    b64data = base64.b64encode(open(test_image, 'rb').read())
    inp = {'img': b64data}
    outp = model.predict(inp)
    print(outp)


# test_elem_layout_v1_pt()
test_elem_layout_v1()
