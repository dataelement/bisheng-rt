import copy

import torch

from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


def create_chat_completion(model, tokenizer, request: ChatCompletionRequest):

    if request.messages[-1].role != 'user':
        raise Exception('Invalid request')

    query = request.messages[-1].content
    # system = 'You are a helpful assistant.'

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

    chat_kwargs = {}
    if request.top_p is not None:
        chat_kwargs.update(top_p=request.top_p)

    if request.temperature is not None:
        chat_kwargs.update(temperature=request.temperature)

    if request.do_sample:
        chat_kwargs.update(do_sample=request.do_sample)

    with torch.no_grad():
        response, _ = model.chat(tokenizer,
                                 query,
                                 history=history,
                                 **chat_kwargs)

    choice_data = ChatCompletionResponseChoice(index=0,
                                               message=ChatMessage(
                                                   role='assistant',
                                                   content=response),
                                               finish_reason='stop')

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


class InternLMChat(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        top_p = kwargs.get('top_p', 0.5)
        max_tokens = kwargs.get('max_tokens', 8192)
        do_sample = kwargs.get('do_sample', False)

        self.default_params = {
            'top_p': top_p,
            'max_new_tokens': max_tokens,
            'do_sample': do_sample
        }

        load_params = {}
        if precision == 'bf16':
            load_params = {'bf16': True}

        use_safetensors = bool(kwargs.get('use_safetensors', '0'))
        use_dispatch = bool(kwargs.get('use_dispatch', '0'))
        load_params.update(use_dispatch=use_dispatch)

        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_safetensors=use_safetensors,
                   **load_params)
        self.generation_config.update(**self.default_params)
        self.model.generation_config = self.generation_config

    def predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, request)
        torch_gc(self.devices)
        return resp.dict()

    def completion(self, kwargs):
        pass
