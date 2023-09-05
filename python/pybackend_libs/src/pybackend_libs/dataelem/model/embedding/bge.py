import copy
import time

import torch
import torch.nn.functional as F

from .embedding import BaseEmbedding, EmbResponse, cls_pool


class BGEZhEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'

        instruction = '为这个句子生成表示以用于检索相关文章：'
        self.query_instruction = kwargs.get('query_instruction', instruction)

        self._load(
            pretrain_path,
            precision,
            devices,
            gpu_memory,
        )

    def predict(self, kwargs):
        model = kwargs.get('model')
        input_texts = kwargs.get('texts')
        emb_type = kwargs.get('type')

        if emb_type == 'query':
            input_texts = [self.query_instruction + q for q in input_texts]

        encoded_input = self.tokenizer(input_texts,
                                       max_length=512,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt')

        input_ids = encoded_input['input_ids'].to(self.default_device)
        attention_mask = encoded_input['attention_mask'].to(
            self.default_device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

        embeddings = cls_pool(outputs.last_hidden_state)
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

        return EmbResponse(model=model, embeddings=embeddings.tolist()).dict()
