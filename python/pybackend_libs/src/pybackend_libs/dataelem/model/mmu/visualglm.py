import base64
import copy
import tempfile

import torch
from pybackend_libs.dataelem.framework.hf_model import HFModel, torch_gc


class VisualGLM(HFModel):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        temperature = kwargs.get('temperature', 0.8)
        top_p = kwargs.get('top_p', 0.4)
        max_length = kwargs.get('max_length', 2048)
        min_length = kwargs.get('min_length', 50)
        do_sample = kwargs.get('do_sample', False)
        top_k = kwargs.get('top_k', 100)
        repetition_penalty = kwargs.get('repetition_penalty', 1.2)

        self.default_params = {
            'max_length': max_length,
            'min_length': min_length,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            'repetition_penalty': repetition_penalty
        }

        self.load(pretrain_path,
                  precision,
                  devices,
                  gpu_memory,
                  use_auto_model=True,
                  use_dispatch=True)

    def predict(self, inp):
        b64_image = inp.pop('b64_image')
        prompt = inp.pop('prompt', '描述这张图片。')

        params = copy.deepcopy(self.default_params)
        params.update(inp)

        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(base64.b64decode(b64_image))
            image_path = tmp_file.name
            with torch.no_grad():
                response, history = self.model.chat(self.tokenizer, image_path,
                                                    prompt, **params)

        resp = {'response': response}
        torch_gc(self.devices)
        return resp
