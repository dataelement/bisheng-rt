import copy
import time

import torch

from .chatglm2_utils import auto_configure_device_map
from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice,
                  ChatCompletionResponseStreamChoice, ChatMessage,
                  DeltaMessage, torch_gc)


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
                   use_dispatch=False)

    def predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, request)
        torch_gc(self.devices)
        return resp.dict()

    def completion(self, kwargs):
        pass

    def stream_predict(self, kwargs):
        # not support function call
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)

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
                        [prev_messages[i].content,
                         prev_messages[i + 1].content])

        kwargs = {
            'temperature': request.temperature,
            'top_p': request.top_p,
            'max_length': request.max_tokens,
            'do_sample': request.do_sample
        }

        stop = []
        if request.stop is not None:
            stop = request.stop
            if isinstance(stop, str):
                stop = [request.stop]

        with torch.no_grad():

            prompt = self.tokenizer.build_prompt(query, history=history)
            inputs_tokens = len(self.tokenizer.tokenizer.encode(prompt)) + 2

            prev_len = 0
            tokens = 0
            created = int(time.time())
            for resp, hist in self.model.stream_chat(self.tokenizer,
                                                     query,
                                                     history=history,
                                                     **kwargs):
                tokens += 1
                delta_resp = resp[prev_len:]
                prev_len += len(delta_resp)
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role='assistant', content=delta_resp),
                    finish_reason=None)

                yield ChatCompletionResponse(model=request.model,
                                             choices=[choice_data],
                                             object='chat.completion',
                                             created=created).dict()
            finish_reason = 'stop'
            if tokens + inputs_tokens >= request.max_tokens:
                finish_reason = 'length'

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role='assistant', content=''),
                finish_reason=finish_reason)

            yield ChatCompletionResponse(model=request.model,
                                         choices=[choice_data],
                                         object='chat.completion',
                                         created=created).dict()

        torch_gc(self.devices)
