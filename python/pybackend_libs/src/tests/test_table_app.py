import base64
import json
import time

import cv2
import numpy as np
from pybackend_libs.dataelem.model.ocr.ocr_client import OCRClient
from pybackend_libs.dataelem.model.table import TableCellApp, TableRowColApp
from pybackend_libs.dataelem.utils import convert_file_to_base64

REPO = '/home/hanfeng/models/'


def test_ocr_client():
    params = {'url': 'http://192.168.106.12:36001'}

    model = OCRClient(**params)

    test_image = '../data/table1.jpg'
    outp = model.predict(test_image)

    print(outp)


def test_table_row_col_app():
    params = {'url': 'http://192.168.106.12:36001'}
    model = OCRClient(**params)
    test_image = '../data/table1.jpg'
    resp = model.predict(test_image)
    ocr_result = resp['result']['ocr_result']

    table_bbox = [[77., 119., 568., 119., 568., 800., 77., 800.]]
    app_params = {
        'model_path':
        REPO + 'general_table_rowcol_detect_graph/1/model.graphdef',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }
    model = TableRowColApp(**app_params)
    test_image = '../data/table1.jpg'
    b64 = convert_file_to_base64(test_image)

    inp = {
        'b64_image': b64,
        'ocr_result': json.dumps(ocr_result),
        'table_bboxes': table_bbox,
    }

    outp = model.predict(inp)
    print(outp)


def test_table_cell_app():
    params = {'url': 'http://192.168.106.12:36001'}
    model = OCRClient(**params)
    test_image = '../data/table1.jpg'
    resp = model.predict(test_image)
    ocr_result = resp['result']['ocr_result']

    table_bbox = [[77., 119., 568., 119., 568., 800., 77., 800.]]
    app_params = {
        'model_path':
        REPO + 'general_table_cell_detect_graph/1/model.graphdef',
        'devices': '6',
        'gpu_memory': 3,
        'precision': 'fp32'
    }
    model = TableCellApp(**app_params)
    test_image = '../data/table1.jpg'
    b64 = convert_file_to_base64(test_image)

    inp = {
        'b64_image': b64,
        'ocr_result': json.dumps(ocr_result),
        'table_bboxes': table_bbox,
    }

    outp = model.predict(inp)
    print(outp)


# test_ocr_client()
# test_table_row_col_app()
test_table_cell_app()
