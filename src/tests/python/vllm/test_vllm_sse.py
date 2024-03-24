# flake8: noqa
import concurrent.futures
import copy
import json
import math
import os
import threading
import time

import requests
import sseclient
import tqdm

RT_EP = os.environ['RT_EP']


def call_llm(model, ep, user_prompt=''):
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {
          'role': 'user',
          'content': 'write a suspense story starts with a beautify night'}],
      'temperature': 0,
      'max_tokens': 512,
      'stream': False
    }
    payload = copy.copy(input_template)
    payload['model'] = model
    if user_prompt:
      payload['messages'][-1]['content'] = user_prompt
    t0 = time.perf_counter()
    # response = requests.post(url=ep, json=payload).json()

    try:
      raw_resp = response = requests.post(url=ep, json=payload)
      response = raw_resp.json()
    except Exception as e:
        print('------------', [raw_resp.text,])
        raise e

    elapse = time.perf_counter() - t0
    response['usage']['first_token_latency'] = elapse
    response['usage']['req_latency'] = elapse
    return response

def call_sse_llm(model, ep, user_prompt=''):
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {
          'role': 'user',
          # 'content': 'what can you do?'}],
          'content': 'write a suspense story starts with a beautify night'}],
      'max_tokens': 512,
      'temperature': 0,
      'stream': True,
    }
    payload = copy.copy(input_template)
    payload['model'] = model
    if user_prompt:
      payload['messages'][-1]['content'] = user_prompt

    headers = {'Accept': 'text/event-stream'}

    res = requests.post(url=ep,
        data=json.dumps(payload), headers=headers, stream=True)
    # res.raise_for_status()

    t0 = time.perf_counter()
    client = sseclient.SSEClient(res)
    res_count = 0
    tokens = 0
    content = ''
    for event in client.events():
        if event.data == '[DONE]':
          break

        if res_count == 0:
            first_token_latency = time.perf_counter() - t0

        delta_data = json.loads(event.data)
        # print('delta_data', delta_data)
        if 'content' in delta_data['choices'][0]['delta']:
          content += delta_data['choices'][0]['delta']['content']
          res_count += 1

    req_latency = time.perf_counter() - t0
    res = {
      'choices': [{'index': 0, 'messages':{'role': 'assistant', 'content': content}}],
      'usage': {
        'completion_tokens': res_count,
        'req_latency': req_latency,
        'first_token_latency': first_token_latency
      }
    }
    return res


def test_baichuan2_13b_chat():
  model = 'Baichuan2-13B-Chat'
  ep = f'http://{RT_EP}/v2.1/models/{model}/generate'
  call_llm(model, ep)


def test_baichuan_13b_chat():
  model = 'Baichuan-13B-Chat'
  ep = f'http://{RT_EP}/v2.1/models/{model}/generate'
  call_llm(model, ep)


def test_qwen_chat(prompt, stream=False):
  model = 'Qwen-1_8B-Chat'
  ep = f'http://{RT_EP}/v1/chat/completions'
  # ep = f'http://{RT_EP}/v2.1/models/{model}/infer'
  if not stream:
    return call_llm(model, ep, prompt)
  else:
    return call_sse_llm(model, ep, prompt)

# barrier = threading.Barrier(100, action=None, timeout=None)

def test_qwen_batch(prompts, stream=False):
  model = 'Qwen-1_8B-Chat'
  ep = f'http://{RT_EP}/v1/chat/completions'
  resps = []
  llm_func = call_sse_llm if stream else call_llm
  for idx, prompt in enumerate(prompts):
    r = llm_func(model, ep, prompt)
    # print('resp:', r)
    resps.append(r)
    # barrier.wait()

  return resps


def perf2():
  alpaca_file = '../../benchmarks/data/alpaca_data_zh_51k.json'
  alpaca = json.load(open(alpaca_file))
  prompts = []
  for sample in alpaca:
    input_text = sample.get('input', '')
    instruction = sample.get('instruction', '')
    prompts.append('{} {}'.format(instruction, input_text))

  print('--- number of prompts', len(prompts))
  stream = True
  print(test_qwen_chat(prompts[0], stream))
  # test_qwen_batch(prompts[:10], stream)
  print('--- start perf test...')
  # return

  warm_n = 100
  start_n = 20000
  perf_n = 5000
  n_parallel = 200

  perf_prompts = prompts[start_n + warm_n: start_n + perf_n + warm_n]
  n_prompts = len(perf_prompts)
  batch = int(math.ceil(n_prompts / n_parallel))

  t0 = time.perf_counter()
  futures_data = []
  rets = []

  with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
      for i in range(n_parallel):
        start = i * batch
        end = (i + 1) * batch
        batch_prompts = perf_prompts[start: end]
        futures_data.append(executor.submit(test_qwen_batch, batch_prompts, stream))

      for future in concurrent.futures.as_completed(futures_data):
          ret = future.result()
          rets.extend(ret)

  elapse = time.perf_counter() - t0
  token_n = sum([ret['usage']['completion_tokens'] for ret in rets])
  token_speed = token_n / elapse
  mean_1token_latency = sum([ret['usage']['first_token_latency'] for ret in rets]) / n_prompts
  mean_req_latency = sum([ret['usage']['req_latency'] for ret in rets]) / n_prompts
  req_speed = n_prompts / elapse

  print('Speed: {} tokens/s, req_latency: {}, first_token_latency: {}, req_speed: {}, tokens: {}'.format(
    token_speed, mean_req_latency, mean_1token_latency, req_speed, token_n))

def perf():
  alpaca_file = '../../benchmarks/data/alpaca_data_zh_51k.json'
  alpaca = json.load(open(alpaca_file))
  prompts = []
  for sample in alpaca:
    input_text = sample.get('input', '')
    instruction = sample.get('instruction', '')
    prompts.append('{} {}'.format(instruction, input_text))

  print('--- number of total prompts', len(prompts))
  stream = False
  print(test_qwen_chat(prompts[0], stream))
  # test_qwen_batch(prompts[:10])
  # return

  warm_n = 100
  start_n = 20000
  perf_n = 5000
  n_parallel = 200

  perf_prompts = prompts[start_n + warm_n: start_n + perf_n + warm_n]
  n_prompts = len(perf_prompts)
  batch = int(math.ceil(n_prompts / n_parallel))

  t0 = time.perf_counter()
  futures_data = []
  rets = []
  with tqdm.tqdm(total=n_prompts) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures_data = [executor.submit(test_qwen_chat, prompt, stream) for prompt in perf_prompts]
        for future in concurrent.futures.as_completed(futures_data):
            rets.append(future.result())
            pbar.update(1)

  elapse = time.perf_counter() - t0
  token_n = sum([ret['usage']['completion_tokens'] for ret in rets])
  token_speed = token_n / elapse
  mean_1token_latency = sum([ret['usage']['first_token_latency'] for ret in rets]) / n_prompts
  mean_req_latency = sum([ret['usage']['req_latency'] for ret in rets]) / n_prompts
  req_speed = n_prompts / elapse

  print('Speed: {} tokens/s, req_latency: {}, first_token_latency: {}, req_speed: {}, tokens: {}'.format(
    token_speed, mean_req_latency, mean_1token_latency, req_speed, token_n))


# test_baichuan_13b_chat()
# test_baichuan2_13b_chat()
# test_qwen_chat()
# perf()
perf2()
