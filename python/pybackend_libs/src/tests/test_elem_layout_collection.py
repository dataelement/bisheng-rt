import json

import requests
from pybackend_libs.dataelem.model.ocr.ocr_client import OCRClient
from pybackend_libs.dataelem.utils import convert_file_to_base64  # import time


def test_ocr_client():
    params = {
        'url': 'http://192.168.106.12:19001',
        'model_name': 'elem_ocr_collection_v3'}

    model = OCRClient(**params)

    test_image = '../data/table1.jpg'
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    outp = model.predict(inp)
    assert outp is not None
    print(outp)


def test_table_row_col_app():
    params = {
        'url': 'http://192.168.106.12:19001',
        'model_name': 'elem_ocr_collection_v3'}

    model = OCRClient(**params)
    test_image = '../data/table1.jpg'
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    resp = model.predict(inp)
    ocr_result = resp['result']['ocr_result']

    table_bboxes = [[77., 119., 568., 119., 568., 800., 77., 800.]]

    url = 'http://192.168.106.12:19001'
    model_name = 'elem_table_rowcol_detect_v1'
    ep = f'{url}/v2.1/models/{model_name}/infer'
    inp = {
       'b64_image': b64,
       'ocr_result': json.dumps(ocr_result),
       'table_bboxes': table_bboxes,
       'sep_char': ' ',
       'longer_edge_size': None,
       'padding': False,
    }

    outp = requests.post(url=ep, json=inp).json()
    assert outp is not None
    # print(outp)

    url = 'http://192.168.106.12:19001'
    model_name = 'elem_table_cell_detect_v1'
    ep = f'{url}/v2.1/models/{model_name}/infer'
    inp = {
       'b64_image': b64,
       'ocr_result': json.dumps(ocr_result),
       'table_bboxes': table_bboxes,
       'sep_char': ' ',
       'longer_edge_size': None,
       'padding': False,
    }

    outp = requests.post(url=ep, json=inp).json()
    assert outp is not None
    print(outp)


test_table_row_col_app()
