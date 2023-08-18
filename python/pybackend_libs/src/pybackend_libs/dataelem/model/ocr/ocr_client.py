import base64
import copy

import requests


class OCRClient(object):
    def __init__(self, url, timeout=10000):
        self.ep = f'{url}/v2/idp/idp_app/infer'
        self.client = requests.Session()
        self.timeout = timeout
        self.params = {
            'sort_filter_boxes': True,
            'enable_huarong_box_adjust': True,
            'rotateupright': False,
            'support_long_image_segment': True,
        }

        self.scene_mapping = {
            'doc': {
                'det': 'general_text_det_mrcnn_v1.0',
                'recog': 'transformer-v2.8-gamma-faster'
            },
            'form': {
                'det': 'mrcnn-v5.1',
                'recog': 'transformer-v2.8-gamma-faster'
            },
            'hand': {
                'det': 'mrcnn-v5.1',
                'recog': 'transformer-hand-v1.16-faster'
            }
        }

    def predict(self, image_file, **kwargs):
        scene = kwargs.get('scene', 'doc')
        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping[scene])

        bytes_data = open(image_file, 'rb').read()
        b64enc = base64.b64encode(bytes_data).decode()
        req_data = {'param': params, 'data': [b64enc]}

        try:
            r = self.client.post(url=self.ep,
                                 json=req_data,
                                 timeout=self.timeout)
            return r.json()
        except Exception as e:
            return {'status_code': 400, 'status_message': str(e)}
