import copy
import json
import os
import time

import requests


def test_repo_index():
    RT_EP = os.environ.get('RT_EP', '192.168.106.12:9001')
    # mode_ep_prefix = f'http://{RT_EP}/v2.1/models'
    repo_ep = f'http://{RT_EP}/v2/repository/index'
    result = requests.post(repo_ep, json={}).json()
    assert result, 'model not exsited'

    models_dict = {}
    for m in result:
        models_dict[m['name']] = '1'
    assert 'chatglm2-6b' in models_dict
    assert 'Qwen-14B-Chat' in models_dict


def call_llm(model, ep):
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素'}],
      'max_tokens': 256
    }
    payload = copy.copy(input_template)
    payload['model'] = model
    response = requests.post(url=ep, json=payload).json()
    choices = response.get('choices', [])
    assert choices, response


def test_load_unload():
    RT_EP = os.environ.get('RT_EP', '192.168.106.12:9001')
    ep_prefix = f'http://{RT_EP}'
    repo_ep = f'http://{RT_EP}/v2/repository'

    test_config = 'config/load_unload.json'
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
        ready_ep = f'{ep_prefix}/v2/models/{name}/ready'
        load_params = load_params_map.get(name)
        infer_ep = f'{ep_prefix}/v2.1/models/{name}/infer'

        print(f'load model {name}...')
        resp = requests.post(load_ep, json=load_params, headers=headers)
        assert resp.status_code == 200
        time.sleep(2)

        ready_resp = requests.get(ready_ep, json={}, headers=headers)
        assert ready_resp.status_code == 200

        try:
            print(f'infer with model {name}...')
            call_llm(name, infer_ep)
        except Exception:
            pass
        finally:
            print(f'unload model {name} ...')
            resp = requests.post(unload_ep, json={}, headers=headers)
            assert resp.status_code == 200
            time.sleep(3)
            ready_resp = requests.get(ready_ep, json={}, headers=headers)
            assert ready_resp.status_code == 400


test_load_unload()
# test_repo_index()
