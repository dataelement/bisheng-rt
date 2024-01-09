import os
from typing import Any, List

import torch


class PTGraph(object):
    def load(self, sig, device, model_path):
        self.enable_gpu = device != 'cpu'
        device = torch.device(f'cuda:{device}' if device else 'cpu')
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device
        self.ys = sig['outputs']
        self.xs = sig['inputs']

    def __init__(self, sig, device, **kwargs):
        model_path = kwargs.get('model_path')
        model_file = os.path.join(model_path, 'model.pt')
        if not os.path.exists(model_file):
            raise Exception(f'{model_file} not exists')
        self.load(sig, device, model_file)

    def run(self, inputs: List[Any]) -> List[Any]:
        assert len(inputs) == len(self.xs)
        pt_tensors = [torch.from_numpy(nd).to(self.device) for nd in inputs]
        with torch.no_grad():
            outputs = self.model(*pt_tensors)

        if not self.enable_gpu:
            return [outputs.numpy()]
        else:
            return [outputs.cpu().numpy()]
