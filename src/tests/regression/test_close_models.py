import base64
import copy
import json
import os
import time

import requests


class OCRClient(object):
    def __init__(self, **kwargs):
        url = kwargs.get('url')
        elem_ocr_collection_v3 = kwargs.get('model_name')
        self.ep = f'{url}/v2.1/models/{elem_ocr_collection_v3}/infer'
        self.client = requests.Session()
        self.timeout = kwargs.get('timeout', 10000)
        self.params = {
            'sort_filter_boxes': True,
            'enable_huarong_box_adjust': True,
            'rotateupright': False,
            'support_long_image_segment': True,
        }

        self.scene_mapping = {
            'print': {
                'det': 'general_text_det_mrcnn_v2.0',
                'recog': 'transformer-blank-v0.2-faster'
            },
            'hand': {
                'det': 'general_text_det_mrcnn_v2.0',
                'recog': 'transformer-hand-v1.16-faster'
            }
        }

    def predict(self, inp):
        scene = inp.pop('scene', 'print')
        b64_image = inp.pop('b64_image')
        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping[scene])
        params.update(inp)

        req_data = {'param': params, 'data': [b64_image]}
        try:
            r = self.client.post(url=self.ep,
                                 json=req_data,
                                 timeout=self.timeout)
            return r.json()
        except Exception as e:
            return {'code': 400, 'message': str(e)}


def test_infer(name):
    image_file = './data/table1.jpg'
    b64_data = base64.b64encode(open(image_file, 'rb').read()).decode()
    if 'layout' in name:
        rt_ep = os.environ.get('RT_EP', '192.168.106.12:9001')
        inp = {'b64_image': b64_data}
        url = f'http://{rt_ep}/v2.1/models/elem_layout_v1/infer'
        outp = requests.post(url, json=inp).json()
        print(outp)
        assert outp is not None

        url = f'http://{rt_ep}/v2.1/models/elem_table_detect_v1/infer'
        outp = requests.post(url, json=inp).json()
        print(outp)
        assert outp is not None

    elif 'ocr' in name:
        rt_ep = os.environ.get('RT_EP', '192.168.106.12:9001')
        url = f'http://{rt_ep}'
        kwargs = {'model_name': name, 'url': url}
        print('kwargs', kwargs)
        ocr_client = OCRClient(**kwargs)
        inp = {'b64_image':  b64_data}
        outp = ocr_client.predict(inp)
        # print(outp)
        assert outp['code'] == 200, outp


def test_closed_model():
    RT_EP = os.environ.get('RT_EP', '192.168.106.12:9001')
    ep_prefix = f'http://{RT_EP}/v2'
    repo_ep = f'http://{RT_EP}/v2/repository'

    test_config = 'config/closed_models.json'
    config = json.load(open(test_config))
    model_names = config['models']
    load_params_map = {}
    for name, params in zip(model_names, config['load_params']):
        load_params_map[name] = params

    test_models = config.get('test_models', [])
    if test_models:
        model_names = test_models

    headers = {'Content-type': 'application/json'}
    for name in model_names:
        load_ep = f'{repo_ep}/models/{name}/load'
        unload_ep = f'{repo_ep}/models/{name}/unload'
        ready_ep = f'{ep_prefix}/models/{name}/ready'
        load_params = load_params_map.get(name)

        print(f'load model {name}...')
        resp = requests.post(load_ep, json=load_params, headers=headers)
        assert resp.status_code == 200
        time.sleep(2)

        ready_resp = requests.get(ready_ep, json={}, headers=headers)
        assert ready_resp.status_code == 200

        try:
            print(f'infer with model {name}...')
            test_infer(name)
        except Exception as e:
            print('err in infer:', e)
        finally:
            print(f'unload model {name} ...')
            resp = requests.post(unload_ep, json={}, headers=headers)
            assert resp.status_code == 200
            time.sleep(3)
            ready_resp = requests.get(ready_ep, json={}, headers=headers)
            assert ready_resp.status_code == 400


test_closed_model()
