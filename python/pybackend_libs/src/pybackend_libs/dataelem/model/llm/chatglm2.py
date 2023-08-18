import copy

import torch

from .chatglm2_utils import auto_configure_device_map
from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


def create_chat_completion(model, tokenizer, request: ChatCompletionRequest):

    if request.messages[-1].role != 'user':
        raise Exception('Invalid request')

    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == 'system':
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if (prev_messages[i].role == 'user'
                    and prev_messages[i + 1].role == 'assistant'):
                history.append(
                    [prev_messages[i].content, prev_messages[i + 1].content])
    kwargs = {
        'temperature': request.temperature,
        'top_p': request.top_p,
        'max_length': request.max_tokens,
        'do_sample': request.do_sample
    }

    with torch.no_grad():
        response, hist = model.chat(tokenizer,
                                    query,
                                    history=history,
                                    **kwargs)

    choice_data = ChatCompletionResponseChoice(index=0,
                                               message=ChatMessage(
                                                   role='assistant',
                                                   content=response),
                                               finish_reason='stop')

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


class ChatGLM2(BaseLLM):
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

        self.default_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'do_sample': do_sample
        }
        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   True,
                   auto_configure_device_map=auto_configure_device_map,
                   use_dispatch=True)

    def chat(self, **kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, request)
        torch_gc(self.devices)
        return resp

    def completion(self, **kwargs):
        pass
