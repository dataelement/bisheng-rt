# flake8: noqa
import copy
from functools import partial

import torch

from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice,
                  ChatCompletionResponseStreamChoice, ChatMessage,
                  DeltaMessage, torch_gc)
from .qwen_utils import (_TEXT_COMPLETION_CMD, add_extra_stop_words,
                         auto_configure_device_map2, parse_messages,
                         parse_response, trim_stop_words)

# from pybackend_libs.dataelem.utils import Timer
# from typing import Any, Dict, List, Literal, Optional, Union


# completion mode, not chat mode
def text_complete_last_message(model, tokenizer,
    history, stop_words_ids, gen_kwargs):
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'
    for i, (query, response) in enumerate(history):
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f'\n{im_start}user\n{query}{im_end}'
        prompt += f'\n{im_start}assistant\n{response}{im_end}'
    prompt = prompt[: -len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(
        input_ids, stop_words_ids=stop_words_ids, **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors='ignore')
    assert output.startswith(prompt)
    output = output[len(prompt):]
    output = trim_stop_words(output, ['<|endoftext|>', im_end])
    # print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output


def create_chat_completion(model, tokenizer, request: ChatCompletionRequest):
    gen_kwargs = {}
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if 'Observation:' not in stop_words:
            stop_words.append('Observation:')

    query, history = parse_messages(request.messages, request.functions)
    stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(model, tokenizer, history,
            stop_words_ids=stop_words_ids, gen_kwargs=gen_kwargs)
    else:
        response, _ = model.chat(
            tokenizer,
            query,
            history=history,
            stop_words_ids=stop_words_ids,
            **gen_kwargs
        )
        # print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")

    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


class QwenChat(BaseLLM):
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

        num_layers = int(kwargs.get('num_layers', '40'))
        device_map_func = partial(auto_configure_device_map2,
                                  num_trans_layers=num_layers,
                                  devices=devices)
        use_safetensors = kwargs .get('use_safetensors', '1') == '1'
        use_dispatch = kwargs.get('use_dispatch', '0') == '1'

        load_params.update(use_dispatch=use_dispatch)

        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_safetensors=use_safetensors,
                   auto_configure_device_map=device_map_func,
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

    def stream_predict(self, kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        if request.stream:
            if request.functions:
                raise Exception(
                    'Invalid request: Function calling is not yet implemented for stream mode.',
                )

        gen_kwargs = {}
        if request.temperature is not None:
            if request.temperature < 0.01:
                gen_kwargs['top_k'] = 1  # greedy decoding
            else:
                # Not recommended. Please tune top_p instead.
                gen_kwargs['temperature'] = request.temperature
        if request.top_p is not None:
            gen_kwargs['top_p'] = request.top_p

        # chat_stream内部使用的transformers_stream_generator暂时不支持do_sample=False
        gen_kwargs['do_sample'] = True

        stop_words = add_extra_stop_words(request.stop)
        query, history = parse_messages(request.messages, request.functions)

        current_length = 0
        stop_words_ids = [self.tokenizer.encode(s) for s in stop_words] if stop_words else None
        if stop_words:
            # TODO: It's a little bit tricky to trim stop words in the stream mode.
            raise Exception(
                'Invalid request: custom stop words are not yet supported for stream mode.',
            )
        response_generator = self.model.chat_stream(
            self.tokenizer,
            query,
            history=history,
            stop_words_ids=stop_words_ids,
            **gen_kwargs
        )
        for new_response in response_generator:
            if len(new_response) == current_length:
                continue

            new_text = new_response[current_length:]
            current_length = len(new_response)

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(
                model=request.model,
                choices=[choice_data],
                object='chat.completion.chunk'
            ).dict()
            yield chunk

        torch_gc(self.devices)

    def completion(self, kwargs):
        pass
