import time
from typing import List, Optional

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from pydantic import BaseModel, Field
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer

# from transformers.generation.utils import GenerationConfig

from .jina_util import JinaBertConfig, JinaBertModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def cls_pool(last_hidden_states: Tensor) -> Tensor:
    # Perform pooling. In this case, cls pooling.
    return last_hidden_states[:, 0]


def torch_gc(devices):
    if torch.cuda.is_available():
        for device_id in devices:
            with torch.cuda.device(f'cuda:{device_id}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def torch_seed(seed=1947):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


class BaseEmbedding(object):
    def __init__(self, **kwargs):
        pass

    def predict(self, kwargs):
        raise Exception('not implemented')

    def _batch_predict(self, batch_size, texts, infer_handler):
        n = len(texts)
        batchs = int(np.ceil(n / batch_size))
        embs = []
        # todo: use parallel infer
        for i in range(batchs):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_texts = texts[start:end]
            embs.extend(infer_handler(batch_texts))

        return embs

    def _load(self,
              pretrain_path,
              precision,
              devices,
              gpu_memory,
              use_safetensors=False,
              use_sentence_transformers=False,
              jina_mode=False):

        torch_seed()

        if use_sentence_transformers:
            raise Exception('not support for sentence-transformers')
            # torch.cuda.set_device(int(devices[0]))
            # self.model = SentenceTransformer(pretrain_path)
            # self.model.eval()
            return

        memory_per_device = int(int(gpu_memory) / len(devices))
        memory = f'{memory_per_device}GiB'
        max_memory = {int(device_id): memory for device_id in devices}

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path,
                                                       use_fast=False,
                                                       trust_remote_code=True)
        with init_empty_weights():
            if jina_mode is True:
                config = JinaBertConfig()
                config._name_or_path = pretrain_path
            else:
                config = AutoConfig.from_pretrained(pretrain_path,
                                                trust_remote_code=True)
            if jina_mode is True:
                model = JinaBertModel(config)
            else:
                model = AutoModel.from_config(config,
                                          torch_dtype=torch.float16,
                                          trust_remote_code=True)

        no_split_modules = model._no_split_modules
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_modules)

        if jina_mode is True:
            self.model = JinaBertModel.from_pretrained(pretrain_path,
                                               device_map=device_map,
                                               trust_remote_code=True,
                                               use_safetensors=use_safetensors
                                               )
        else:
            self.model = AutoModel.from_pretrained(pretrain_path,
                                               device_map=device_map,
                                               torch_dtype=torch.float16,
                                               trust_remote_code=True,
                                               use_safetensors=use_safetensors
                                               )

        self.model.eval()


class EmbResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))