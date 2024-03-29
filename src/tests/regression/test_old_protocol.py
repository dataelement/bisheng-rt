import argparse

import requests


def infer(url, model_name):
    infer_urls = [
        f'http://{url}/v2.1/models/{model_name}/generate_stream',
        f'http://{url}/v2.1/models/{model_name}/generate',
        f'http://{url}/v2.1/models/{model_name}/infer',
        f'http://{url}/v1/chat/completions'
    ]

    prompt = '以`今晚夜色真美`写一个短文，包含悬疑元素'
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': '你是来自数据项素的智能助手'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.85,
        'top_p': 0.8,
        'stream': False
    }
    headers = {'Content-type': 'application/json'}
    for infer_url in infer_urls:
        resp = requests.post(url=infer_url, json=payload, headers=headers)
        print('url: {}, resp: {}'.format(infer_url, resp.text))


def main(args):
    url = args.url
    model_name = args.model_name
    assert model_name, 'empty model_name'
    infer(url, model_name)


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
        '-m',
        '--model-name',
        type=str,
        required=False,
        default=None,
        help='model name',
    )

    args = parser.parse_args()
    main(args)
