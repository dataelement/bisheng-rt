import base64
import time

import cv2
import numpy as np
from pybackend_libs.dataelem.model.table.table_mrcnn import (
    MrcnnTableCellDetect, MrcnnTableDetect, MrcnnTableRowColDetect)
from pybackend_libs.dataelem.utils import convert_file_to_base64

REPO = '/home/hanfeng/models/'


def test_table_detect():
    params = {
        'model_path': REPO + 'general_table_detect_graph/1/model.graphdef',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = MrcnnTableDetect(**params)

    test_image = '../data/table1.jpg'
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    outp = model.predict(inp)
    print(outp)


def test_table_cell_det():
    bbox = np.asarray([77., 119., 568., 119., 568., 800., 77., 800.])
    test_image = '../data/table1.jpg'
    img = cv2.imread(test_image)
    sub_img = crop(img, bbox)[0]

    params = {
        'model_path':
        REPO + 'general_table_cell_detect_graph/1/model.graphdef',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = MrcnnTableCellDetect(**params)
    outp = model.predict({}, [sub_img])
    print(outp)


def test_table_rowcol_det():
    bbox = np.asarray([77., 119., 568., 119., 568., 800., 77., 800.])
    test_image = '../data/table1.jpg'
    img = cv2.imread(test_image)
    sub_img = crop(img, bbox)[0]

    params = {
        'model_path':
        REPO + 'general_table_rowcol_detect_graph/1/model.graphdef',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = MrcnnTableRowColDetect(**params)
    outp = model.predict({}, [sub_img])
    print(outp)


def test_table_detect2():
    params = {
        'model_path': '/home/public/llm/elem_table_detect_v1/',
        'devices': '8',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    model = MrcnnTableDetect(**params)

    test_image = '../data/001.png'
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    outp = model.predict(inp)
    print(outp)


# test_table_rowcol_det()
# test_table_cell_det()
# test_table_detect()
test_table_detect2()
