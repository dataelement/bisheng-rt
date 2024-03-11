# import copy
# import time

import torch
import torch.nn.functional as F

from .embedding import BaseEmbedding, EmbResponse, average_pool, torch_gc


class JINAEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.batch_size = int(kwargs.get('batch_size', '32'))
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'
        self._load(pretrain_path, precision, devices, gpu_memory, jina_mode=True)

    def predict(self, kwargs):
        model = kwargs.get('model')
        input_texts = kwargs.get('texts')

        def infer_handler(input_texts):
            embeddings = self.model.encode(input_texts)
            return embeddings.tolist()

        embs = self._batch_predict(self.batch_size, input_texts, infer_handler)
        torch_gc(self.devices)
        return EmbResponse(model=model, embeddings=embs).dict()
