import base64
import json
import os

import requests
from pybackend_libs.dataelem.model.ocr.ocr_client import OCRClient


def test_ocr():
    rt_ep = os.environ.get('RT_EP', '192.168.106.12:19001')
    kwargs = {
        'url': f'http://{rt_ep}',
        'model_name': 'elem_ocr_collection_v3'
    }
    ocr_client = OCRClient(**kwargs)
    proj_dir = '/home/hanfeng/projects/bisheng-rt'
    image_file = proj_dir + '/python/pybackend_libs/data/table1.jpg'
    print(image_file)
    inp = {
        'b64_image':  base64.b64encode(open(image_file, 'rb').read()).decode(),
    }

    outp = ocr_client.predict(inp)
    assert outp is not None
    print(outp)


def test_table_row_col_app():
    rt_ep = os.environ.get('RT_EP', '192.168.106.12:19001')
    kwargs = {
        'url': f'http://{rt_ep}',
        'model_name': 'elem_ocr_collection_v3'
    }
    ocr_client = OCRClient(**kwargs)
    proj_dir = '/home/hanfeng/projects/bisheng-rt'
    image_file = proj_dir + '/python/pybackend_libs/data/table1.jpg'
    b64 = base64.b64encode(open(image_file, 'rb').read()).decode()
    inp = {'b64_image': b64}
    resp = ocr_client.predict(inp)
    ocr_result = resp['result']['ocr_result']
    table_bboxes = [[77., 119., 568., 119., 568., 800., 77., 800.]]

    model_name = 'elem_table_rowcol_detect_v1'
    ep = f'http://{rt_ep}/v2.1/models/{model_name}/infer'
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


# test_ocr()
test_table_row_col_app()
