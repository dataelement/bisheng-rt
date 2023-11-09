import copy
import os

import requests


def test1():
    RT_EP = os.environ.get('RT_EP', '192.168.106.12:9001')
    ep_prefix = f'http://{RT_EP}/v2.1/models'

    models = ['Qwen-14B-Chat', 'chatglm3-6b', 'Baichuan2-13B-Chat']
    input_template = {
      'model': 'unknown',
      'messages': [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素'}],
      'max_tokens': 256
    }

    for model in models:
        payload = copy.copy(input_template)
        payload['model'] = model
        ep = '{0}/{1}/infer'.format(ep_prefix, model)
        response = requests.post(url=ep, json=payload).json()
        choices = response.get('choices', [])
        assert choices, response
        print(f'model {model} was envoked', choices)


test1()
