import copy
import os

import torch
from transformers import AutoModel, AutoTokenizer

from .chatglm2_utils import auto_configure_device_map
from .llm import (BaseLLM, Choise, CompletionRequest, CompletionResponse,
                  torch_gc)


def create_completion(model, tokenizer, request: CompletionRequest,
                      engine_type):
    kwargs = {
        'temperature': 0.2,
        'top_p': 0.95,
        'max_length': 128,
        'top_k': 1,
        'do_sample': True,
        'history': [],
    }
    if request.temperature is not None:
        kwargs.update(temperature=request.temperature)
    if request.top_p is not None:
        kwargs.update(top_p=request.top_p)
    if request.top_k is not None:
        kwargs.update(top_k=request.top_k)
    if request.max_tokens is not None:
        kwargs.update(max_length=request.max_tokens)
    if request.do_sample is not None:
        kwargs.update(do_sample=request.do_sample)

    prompt = request.prompt

    if engine_type == 'fastllm':
        response, _ = model.chat(tokenizer, prompt, **kwargs)
    else:
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=256, top_k=1)
        response = tokenizer.decode(outputs[0])

    choice = Choise(text=response)
    return CompletionResponse(model=request.model, choices=[choice])


class CodeGeeX2(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        temperature = kwargs.get('temperature', 0.95)
        top_p = kwargs.get('top_p', 0.7)
        max_tokens = kwargs.get('max_tokens', 8192)
        do_sample = kwargs.get('do_sample', False)

        # support engine_type: hf, fastllm
        self.engine_type = kwargs.get('engine_type', 'hf')

        self.default_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'do_sample': do_sample
        }

        if self.engine_type == 'llm':
            from fastllm_pytools import llm

            flm_model_path = os.path.join(pretrain_path, 'model.flm')
            # support tensor parallel
            llm_devices = [f'cuda:{d}' for d in devices]
            llm.set_device_map(llm_devices)

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrain_path, use_fast=False, trust_remote_code=True)

            if os.path.exists(flm_model_path):
                self.model = llm.model(flm_model_path)
            else:
                model = AutoModel.from_pretrained(pretrain_path,
                                                  trust_remote_code=True)
                model = model.eval()
                model = model.half()
                dtype = 'float16' if precision == 'fp16' else 'float32'
                self.model = llm.from_hf(model,
                                         self.tokenizer,
                                         dtype=dtype)
                self.model.save(flm_model_path)
        else:
            self._load(pretrain_path,
                       precision,
                       devices,
                       gpu_memory,
                       True,
                       auto_configure_device_map=auto_configure_device_map,
                       use_dispatch=True)

    def completion(self, kwargs):
        pass

    def predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = CompletionRequest.parse_obj(req_dict)
        resp = create_completion(self.model, self.tokenizer, request,
                                 self.engine_type)
        torch_gc(self.devices)
        return resp.dict()
