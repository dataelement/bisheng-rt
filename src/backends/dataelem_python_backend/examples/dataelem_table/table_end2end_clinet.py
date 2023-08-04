import base64

import requests


def SendReqWithRest(client, ep, req_body):
    def post(ep, json_data=None, timeout=10000):
        url = 'http://{}/v2/idp/ocr_app/infer'.format(ep)
        # print('url', url)
        r = client.post(url=url, json=json_data, timeout=timeout)
        # print('bodylen', len(r.text), (r.text))
        return r

    try:
        r = post(ep, req_body)
    except Exception as e:
        print('Exception: ', e)
        pass

    return r.json()


def main():
    client = requests.Session()
    ep = '127.0.0.1:8502'
    image_file = './完税证明_convert/val_images/00505.jpg'
    bytes_data = open(image_file, 'rb').read()
    b64enc = base64.b64encode(bytes_data).decode()
    params = {
        'longer_edge_size': 1600,
        'padding': False,
        'enable_huarong_box_adjust': True,
        'sort_filter_boxes': True,
        'rotateupright': False,
        'support_long_image_segment': False,
        'refine_boxes': True,
        'det': 'mrcnn-v5.1',
        'recog': 'transformer-v2.8-gamma-faster',
        'table': 'general_table_rowcol_app'
    }
    req_data = {'param': params, 'data': [b64enc]}
    r = SendReqWithRest(client, ep, req_data)
    print(r)


main()
