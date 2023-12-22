import copy
import time

import torch

from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice,
                  ChatCompletionResponseStreamChoice, ChatMessage,
                  DeltaMessage, torch_gc)

# import time
# from typing import Any, Dict, List, Literal, Optional, Union


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

    gen_config = {}
    if request.max_tokens is not None:
        gen_config.update(max_new_tokens=request.max_tokens)

    if request.top_p is not None:
        gen_config.update(top_p=request.top_p)

    if request.temperature is not None:
        gen_config.update(temperature=request.temperature)

    if request.do_sample is not None:
        gen_config.update(do_sample=request.do_sample)

    if gen_config:
        generation_config = copy.copy(model.generation_config)
        generation_config.update(**gen_config)
    else:
        generation_config = model.generation_config

    with torch.no_grad():
        response = model.chat(
            tokenizer, messages,
            generation_config=generation_config)

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

    def predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, request)
        torch_gc(self.devices)
        return resp.dict()

    def stream_predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)

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

        gen_config = {}
        if request.max_tokens is not None:
            gen_config.update(max_new_tokens=request.max_tokens)

        if request.top_p is not None:
            gen_config.update(top_p=request.top_p)

        if request.temperature is not None:
            gen_config.update(temperature=request.temperature)

        if request.do_sample is not None:
            gen_config.update(do_sample=request.do_sample)

        if gen_config:
            new_gen_config = copy.copy(self.model.generation_config)
            new_gen_config.update(**gen_config)
        else:
            new_gen_config = self.model.generation_config

        created = int(time.time())

        prev_len = 0
        tokens = 0
        with torch.no_grad():
            for response in self.model.chat(self.tokenizer, messages,
                                            stream=True,
                                            generation_config=new_gen_config):
                delta_resp = response[prev_len:]
                finish_reason = None
                if len(delta_resp) == 0:
                    finish_reason = 'stop'

                tokens += 1
                prev_len += len(delta_resp)
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role='assistant', content=delta_resp),
                    finish_reason=finish_reason)

                yield ChatCompletionResponse(model=request.model,
                                             choices=[choice_data],
                                             object='chat.completion',
                                             created=created).dict()

        torch_gc(self.devices)

    def completion(self, **kwargs):
        pass
