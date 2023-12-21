# flake8: noqa
import copy
import time

import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from .chatglm2_utils import auto_configure_device_map
from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice,
                  ChatCompletionResponseStreamChoice, ChatMessage,
                  DeltaMessage, torch_gc)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor,
        scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def stream_chat(
    model, tokenizer, query: str,
    history=None,
    role: str = 'user',
    past_key_values=None,
    max_length: int = 8192,
    do_sample=True,
    top_p=0.8,
    temperature=0.8,
    logits_processor=None,
    return_past_key_values=False,
    stop=[],
    **kwargs):

    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()

    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command('<|user|>'),
                    tokenizer.get_command('<|observation|>')]
    for word in stop:
        tokens = tokenizer.tokenizer.encode(word)
        # be careful, read more about sp tokenization
        # the stop word must mapped to single token, otherwise it is wrong!
        eos_token_id.append(tokens[-1])

    eos_token_id = list(set(eos_token_id))

    gen_kwargs = {'max_length': max_length,
                  'do_sample': do_sample, 'top_p': top_p,
                  'temperature': temperature,
                  'logits_processor': logits_processor, **kwargs}
    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)
    inputs = inputs.to(model.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if model.transformer.pre_seq_len is not None:
            past_length -= model.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat(
            (attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    history.append({'role': role, 'content': query})
    for outputs in model.stream_generate(
            **inputs,
            past_key_values=past_key_values,
            eos_token_id=eos_token_id,
            return_past_key_values=return_past_key_values,
            **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs['input_ids'][0]):-1]
        response = tokenizer.decode(outputs)
        if response and response[-1] != '�':
            response, new_history = model.process_response(response, history)
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history


def create_chat_completion(model, tokenizer, request: ChatCompletionRequest):

    if request.messages[-1].role != 'user':
        raise Exception('Invalid request')

    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    history = []
    if len(prev_messages) > 0:
        for msg in prev_messages:
            history.append(dict(msg))

    kwargs = {
        'temperature': request.temperature,
        'top_p': request.top_p,
        'max_length': request.max_tokens,
        'do_sample': request.do_sample
    }

    with torch.no_grad():
        _, hist = model.chat(tokenizer,
                             query,
                             history=history,
                             **kwargs)
        response = hist[-1]

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role=response['role'],
            content=response['content'],
            metadata=response['metadata']),
        finish_reason='stop')

    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


class ChatGLM3(BaseLLM):
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

    def stream_predict(self, kwargs):
        # not support function call
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)

        if request.messages[-1].role != 'user':
            raise Exception('Invalid request')

        query = request.messages[-1].content

        prev_messages = request.messages[:-1]
        history = []
        if len(prev_messages) > 0:
            for msg in prev_messages:
                history.append(dict(msg))

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

        SYS_TOKENS_LEN = 2
        with torch.no_grad():

            inputs = self.tokenizer.build_chat_input(
                query, history=history, role='user')
            inputs_tokens = inputs['input_ids'].size()[-1]

            prev_len = 0
            tokens = 0
            created = int(time.time())
            for resp, hist in stream_chat(self.model, self.tokenizer,
                                          query,
                                          history=history,
                                          stop=stop,
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
                                             created=created)
            finish_reason = 'stop'
            if tokens + SYS_TOKENS_LEN + inputs_tokens >= request.max_tokens:
                finish_reason = 'length'

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role='assistant', content=''),
                finish_reason=finish_reason)

            yield ChatCompletionResponse(model=request.model,
                                         choices=[choice_data],
                                         object='chat.completion',
                                         created=created)

        torch_gc(self.devices)

    def completion(self, kwargs):
        pass
