# flake8: noqa
import copy

import torch
from threading import Thread
from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice,
                  ChatCompletionResponseStreamChoice, ChatMessage,
                  DeltaMessage, torch_gc)
from .qwen_utils import (parse_response, trim_stop_words)
from transformers import TextIteratorStreamer

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
    return output


def create_chat_completion(model, tokenizer, device, request: ChatCompletionRequest):

    text = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

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


class Qwen1_5Chat(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'

        max_tokens = kwargs.get('max_tokens', 8192)
        do_sample = kwargs.get('do_sample', True)

        self.default_params = {
            'max_new_tokens': max_tokens,
            'do_sample': do_sample
        }

        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_safetensors=True,
                   auto_configure_device_map=True)
        self.generation_config.update(**self.default_params)
        self.model.generation_config = self.generation_config

    def predict(self, kwargs):

        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)
        request = ChatCompletionRequest.parse_obj(req_dict)
        resp = create_chat_completion(self.model, self.tokenizer, self.default_device, request)
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
        text = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.default_device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()
        for new_text in streamer:

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
