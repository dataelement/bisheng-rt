import copy
import time

import torch

from .llama2_utils import LlamaTokenizerHelper
from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


class Llama2Chat(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        self.default_device = f'cuda:{devices[0]}'
        temperature = kwargs.get('temperature', 0.9)
        top_p = kwargs.get('top_p', 0.6)
        # max length: 4096
        max_tokens = kwargs.get('max_tokens', 4096)
        self.default_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_length': max_tokens,
            'do_sample': False
        }

        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_safetensors=True)
        self.generation_config.update(**self.default_params)
        self.model.generation_config = self.generation_config
        self.tokenizer_helper = LlamaTokenizerHelper(self.tokenizer)

    def chat(self, **kwargs):
        messages = kwargs.get('messages')
        model_name = kwargs.get('model')
        input_ids = self.tokenizer_helper.chat_completion([messages])
        input0_len = input_ids.size()[1]

        # prompt = messages[-1]['content']
        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids.to(self.default_device), **self.default_params)

        new_tokens = generate_ids[0, input0_len:]
        response = self.tokenizer.decode(new_tokens,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

        choice_data = ChatCompletionResponseChoice(index=0,
                                                   message=ChatMessage(
                                                       role='assistant',
                                                       content=response),
                                                   finish_reason='stop')

        result = ChatCompletionResponse(model=model_name,
                                        choices=[choice_data],
                                        object='chat.completion')

        torch_gc(self.devices)
        return result

    def completion(self, **kwargs):
        pass
