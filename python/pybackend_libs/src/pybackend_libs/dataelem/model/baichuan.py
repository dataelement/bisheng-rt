import copy
import time
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


def create_chat_completion(model, tokenizer, request: ChatCompletionRequest):

    if request.messages[-1].role != 'user':
        raise Exception('Invalid request')

    messages = []
    system_content = ''
    for m in request.messages:
        if m.role == 'system':
            system_content += m.content
            continue

        messages.append({'role': m.role, 'content': m.content})

    if system_content:
        messages[-1]['content'] = system_content + messages[-1]['content']

    messages = messages[::-1]
    with torch.no_grad():
        response = model.chat(tokenizer, messages)

    choice_data = ChatCompletionResponseChoice(index=0,
                                               message=ChatMessage(
                                                   role='assistant',
                                                   content=response),
                                               finish_reason='stop')

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


class BaichuanChat(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        temperature = kwargs.get('temperature', 0.3)
        top_p = kwargs.get('top_p', 0.85)
        max_tokens = kwargs.get('max_tokens', 4096)
        do_sample = kwargs.get('do_sample', False)
        self.default_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_new_tokens': max_tokens,
            'do_sample': do_sample
        }

        # self.model, self.tokenizer = None, None
        self._load(pretrain_path, precision, devices, gpu_memory)
        self.generation_config.update(**self.default_params)
        self.model.generation_config = self.generation_config

    def chat(self, **kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, request)
        torch_gc(self.devices)
        return resp

    def completion(self, **kwargs):
        pass
