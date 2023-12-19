import base64
import os

import requests


def test():
    rt_ep = os.environ.get('RT_EP', '192.168.106.127:9001')
    proj_dir = '/home/hanfeng/projects/bisheng-rt'
    image_file = proj_dir + '/python/pybackend_libs/data/table1.jpg'
    print(image_file)
    inp = {
        'b64_image':  base64.b64encode(open(image_file, 'rb').read()).decode(),
    }
    url = f'http://{rt_ep}/v2.1/models/elem_layout_v1/infer'
    outp = requests.post(url, json=inp).json()
    print(outp)
    assert outp is not None


def test2():
    rt_ep = os.environ.get('RT_EP', '192.168.106.127:9001')
    proj_dir = '/home/hanfeng/projects/bisheng-rt'
    image_file = proj_dir + '/python/pybackend_libs/data/table1.jpg'
    print(image_file)
    inp = {
        'b64_image':  base64.b64encode(open(image_file, 'rb').read()).decode(),
    }
    url = f'http://{rt_ep}/v2.1/models/elem_table_detect_v1/infer'
    outp = requests.post(url, json=inp).json()
    print(outp)
    assert outp is not None


test()
test2()
