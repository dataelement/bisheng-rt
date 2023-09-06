# import base64
import json

# import cv2
# import numpy as np
from pybackend_libs.dataelem.model.ocr.ocr_client import OCRClient
from pybackend_libs.dataelem.model.table import TableCellApp, TableRowColApp
from pybackend_libs.dataelem.utils import convert_file_to_base64

# import time

REPO = '/home/public/models/'


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
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    resp = model.predict(inp)
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
    b64 = convert_file_to_base64(test_image)
    inp = {'b64_image': b64}
    resp = model.predict(inp)
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
    inp = {
        'b64_image': b64,
        'ocr_result': json.dumps(ocr_result),
        'table_bboxes': table_bbox,
    }

    outp = model.predict(inp)
    print(outp)


def test2():
    ocr_result = {
        'texts': ['序', '号', '资质名称', '/', '编号', '主体', '发证机构', '发证日期', '证书有效期'],
        'bboxes': [[(90.50399780273438, 726.7449951171875),
                    (101.06399536132812, 726.7449951171875),
                    (101.06399536132812, 737.3049926757812),
                    (90.50399780273438, 737.3049926757812)],
                   [(90.50399780273438, 740.3009643554688),
                    (101.06399536132812, 740.3009643554688),
                    (101.06399536132812, 750.8609619140625),
                    (90.50399780273438, 750.8609619140625)],
                   [(130.94000244140625, 733.5809936523438),
                    (173.17999267578125, 733.5809936523438),
                    (173.17999267578125, 744.1409912109375),
                    (130.94000244140625, 744.1409912109375)],
                   [(173.3000030517578, 731.63134765625),
                    (176.23568725585938, 731.63134765625),
                    (176.23568725585938, 746.2569580078125),
                    (173.3000030517578, 746.2569580078125)],
                   [(176.05999755859375, 733.5809936523438),
                    (197.17999267578125, 733.5809936523438),
                    (197.17999267578125, 744.1409912109375),
                    (176.05999755859375, 744.1409912109375)],
                   [(238.1300048828125, 733.5809936523438),
                    (259.25, 733.5809936523438), (259.25, 744.1409912109375),
                    (238.1300048828125, 744.1409912109375)],
                   [(311.2099914550781, 733.5809936523438),
                    (353.4499816894531, 733.5809936523438),
                    (353.4499816894531, 744.1409912109375),
                    (311.2099914550781, 744.1409912109375)],
                   [(396.07000732421875, 733.5809936523438),
                    (438.30999755859375, 733.5809936523438),
                    (438.30999755859375, 744.1409912109375),
                    (396.07000732421875, 744.1409912109375)],
                   [(453.54998779296875, 733.5809936523438),
                    (506.3499755859375, 733.5809936523438),
                    (506.3499755859375, 744.1409912109375),
                    (453.54998779296875, 744.1409912109375)]]
    }

    # table_bbox = [[91, 724, 513, 724, 513, 753, 91, 753]]
    table_bbox = [[88.0, 725.0, 511.0, 725.0, 511.0, 753.0, 88.0, 753.0]]
    app_params = {
        'model_path': '/home/public/llm/elem_table_rowcol_detect_v1',
        'devices': '8',
        'gpu_memory': 3,
        'precision': 'fp32'
    }

    test_image = '../data/001.png'
    b64 = convert_file_to_base64(test_image)

    model = TableRowColApp(**app_params)
    inp = {
        'b64_image': b64,
        'ocr_result': json.dumps(ocr_result),
        'table_bboxes': table_bbox,
    }

    outp = model.predict(inp)
    print(outp)


# test_ocr_client()
# test_table_row_col_app()
# test_table_cell_app()
test2()
