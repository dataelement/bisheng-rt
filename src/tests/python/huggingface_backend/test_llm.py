# flake8: noqa
import copy
import json
import os
import time

import requests
import sseclient


def call_sse_llm(model, ep):
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {
          'role': 'user',
          'content': 'what can you do?'}],
          # 'content': '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素'}],
      'max_tokens': 256,
      'stream': True,
    }
    payload = copy.copy(input_template)
    payload['model'] = model
    headers = {'Accept': 'text/event-stream'}
    print('1.1')
    res = requests.post(url=ep,
        data=json.dumps(payload), headers=headers, stream=False)
    res.raise_for_status()

    print('1.2')
    # print('res.text', res.text, res.code)
    client = sseclient.SSEClient(res)
    print('1.3')
    res_count = 0
    for event in client.events():
        print('event data', event.data)
        # delta_data = json.loads(event.data)
        # print('sse data', delta_data)
        res_count += 1


def call_llm(model, ep):
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {
          'role': 'user',
          'content': '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素'}],
      'max_tokens': 256,
    }
    payload = copy.copy(input_template)
    payload['model'] = model
    response = requests.post(url=ep, json=payload).json()
    choices = response.get('choices', [])
    assert choices, response


def run_model_lifecycle(name, params, stream=False):
    RT_EP = os.environ.get('RT_EP')
    ep_prefix = f'http://{RT_EP}'
    repo_ep = f'http://{RT_EP}/v2/repository'
    headers = {'Content-type': 'application/json'}

    load_ep = f'{repo_ep}/models/{name}/load'
    unload_ep = f'{repo_ep}/models/{name}/unload'
    ready_ep = f'{ep_prefix}/v2/models/{name}/ready'
    if not stream:
        infer_ep = f'{ep_prefix}/v2.1/models/{name}/generate'
        infer_func = call_llm
    else:
        infer_ep = f'{ep_prefix}/v2.1/models/{name}/generate_stream'
        infer_func = call_sse_llm


    print(f'load model {name}...')
    resp = requests.post(load_ep, json=params, headers=headers)
    assert resp.status_code == 200
    time.sleep(2)

    ready_resp = requests.get(ready_ep, json={}, headers=headers)
    assert ready_resp.status_code == 200

    succ = True
    try:
        print(f'infer with model {name}...')
        infer_func(name, infer_ep)
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

        # vllm model will release memory with a delay
        time.sleep(10)

    assert succ, 'infer has execption'


def test_llm_model():
    hf_models = json.load(open('llm_models.config'))
    models = hf_models['models']
    params = hf_models['load_params']

    # test non decoupled models
    # run_model_lifecycle(models[0], params[0])
    # run_model_lifecycle(models[1], params[1])
    # run_model_lifecycle(models[2], params[2])

    # test decoupled models
    params0 = copy.copy(params[0])
    params0['parameters']['decoupled'] = '1'
    run_model_lifecycle(models[0], params0, True)


test_llm_model()
