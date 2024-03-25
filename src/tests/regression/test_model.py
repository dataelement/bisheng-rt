import argparse
import json
import os
import time

import requests

RT_EP = os.environ.get('RT_EP', '127.0.0.1:9001')


def load(url, model_name, load_params):
    load_ep = f'http://{url}/v2/repository/models/{model_name}/load'
    ready_ep = f'http://{url}/v2/models/{model_name}/ready'
    headers = {'Content-type': 'application/json'}
    resp = requests.post(load_ep, json=load_params, headers=headers)
    assert resp.status_code == 200
    time.sleep(2)
    ready_resp = requests.get(ready_ep, json={}, headers=headers)
    assert ready_resp.status_code == 200


def unload(url, model_name):
    headers = {'Content-type': 'application/json'}
    print(f'unload model {model_name} ...')
    unload_ep = f'http://{url}/v2/repository/models/{model_name}/unload'
    resp = requests.post(unload_ep, json={}, headers=headers)
    assert resp.status_code == 200


def infer_emb(url, model_name):
    url = f'http://{url}/v1/embeddings'
    payload = {'model': model_name, 'texts': ['今天天气真好']}
    resp = requests.post(url, json=payload)
    print(resp.text)


def infer(url, model_name):
    from openai import OpenAI

    client = OpenAI(
        api_key='DUMMY_API_KEY',
        base_url=f'http://{url}/v1')

    prompt = '以`今晚夜色真美`写一个短文，包含悬疑元素'
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': '你是来自数据项素的智能助手'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.85,
        top_p=0.8,
        stream=True)

    for chunk in completion:
        print('---chunk---', chunk)
    # print(completion.choices[0].message)


def main(args):
    url = args.url
    model_tag = args.model_tag
    model_type = args.model_type
    model_config = json.load(open('./config/models.json'))
    model_name = model_config.get(model_tag)['model_name']
    load_params = model_config.get(model_tag)['config']

    load(url, model_name, load_params)
    if model_type == 'embedding':
        infer_emb(url, model_name)
    else:
        infer(url, model_name)

    unload(url, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='127.0.0.1:9001',
        help='model url.',
    )

    parser.add_argument(
        '-t',
        '--model-tag',
        type=str,
        required=False,
        default=None,
        help='model tag',
    )

    parser.add_argument(
        '-m',
        '--model-type',
        type=str,
        required=False,
        default='llm',
        help='model type llm or embedding',
    )

    args = parser.parse_args()
    main(args)
