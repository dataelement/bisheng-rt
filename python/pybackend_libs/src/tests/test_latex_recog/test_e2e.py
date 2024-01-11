# flake8: noqa
import base64
import os
import time

import requests

det_model_config = {
  'parameters': {
    'type': 'dataelem.private_model.App',
    'model_type': 'dataelem.pymodel.elem_alg_v1',
    'model_params': "{\"pymodel_type\": \"ocr.LatexDetection\", \"dep_model_name\": \"latex_det_graph_v1\"}",
    'gpu_memory': '3',
    'instance_groups': 'device=gpu;gpus=0',
    'reload': '1'
  }
}

recog_model_config = {
  'parameters': {
    'type': 'dataelem.private_model.App',
    'model_type': 'dataelem.pymodel.elem_alg_v1',
    'model_params': "{\"pymodel_type\": \"ocr.LatexRecog\", \"dep_model_name\": \"latex_recog_graph_v1\", \"enable_safe_encoder\": \"1\"}",
    'gpu_memory': '4',
    'instance_groups': 'device=gpu;gpus=0',
    'reload': '1'
  }
}


def test_model_load_unload(name, model_config, model_type='det'):
    RT_EP = os.environ.get('RT_EP')
    ep_prefix = f'http://{RT_EP}'
    repo_ep = f'http://{RT_EP}/v2/repository'
    headers = {'Content-type': 'application/json'}

    params = model_config
    load_ep = f'{repo_ep}/models/{name}/load'
    unload_ep = f'{repo_ep}/models/{name}/unload'
    ready_ep = f'{ep_prefix}/v2/models/{name}/ready'
    infer_ep = f'{ep_prefix}/v2.1/models/{name}/infer'
    repo_index_ep = f'{repo_ep}/index'

    resp = requests.post(repo_index_ep, json={}, headers=headers).json()
    print('repo', resp)

    print(f'load model {name} with {load_ep}...')
    resp = requests.post(load_ep, json=params, headers=headers)
    assert resp.status_code == 200
    time.sleep(2)

    ready_resp = requests.get(ready_ep, json={}, headers=headers)
    assert ready_resp.status_code == 200

    succ = True
    try:
        print(f'infer with model {name}...')
        if model_type == 'det':
            test_det_infer()
        else:
            test_recog_infer()
    except Exception as e:
        print('infer has execption', e)
        succ = False
    finally:
        print(f'unload model {name} ...')
        resp = requests.post(unload_ep, json={}, headers=headers)
        assert resp.status_code == 200

        time.sleep(3)
        ready_resp = requests.get(ready_ep, json={}, headers=headers)
        assert ready_resp.status_code == 400
        time.sleep(5)
        pass

    assert succ, 'infer has execption'


def test_det_infer():
    RT_EP = os.environ.get('RT_EP')
    ep_prefix = f'http://{RT_EP}'
    headers = {'Content-type': 'application/json'}
    name = 'latex_det'
    infer_ep = f'{ep_prefix}/v2.1/models/{name}/infer'

    image_file = '/public/bisheng/latex_data/zh6.jpg'
    payload = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }

    resp = requests.post(url = infer_ep, json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    print('resp', resp.json())


def test_recog_infer():
    RT_EP = os.environ.get('RT_EP')
    ep_prefix = f'http://{RT_EP}'
    headers = {'Content-type': 'application/json'}
    name = 'latex_recog'
    infer_ep = f'{ep_prefix}/v2.1/models/{name}/infer'

    image_file = '/public/bisheng/latex_data/formula.jpg'
    payload = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }

    resp = requests.post(url = infer_ep, json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    print('resp', resp.json())


# test_model_load_unload('latex_det', det_model_config, model_type='det')
test_model_load_unload('latex_recog', recog_model_config, model_type='recog')
# test_det_infer()
