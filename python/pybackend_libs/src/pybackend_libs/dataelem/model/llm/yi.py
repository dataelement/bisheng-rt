# import copy
# import time

import torch

from .llm import (BaseLLM, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


class YiBase(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices

        self.default_device = f'cuda:{devices[0]}'
        # max length: 8192
        max_tokens = kwargs.get('max_tokens', 8192)
        repetition_penalty = kwargs.get('repetition_penalty', 1.3)
        no_repeat_ngram_size = kwargs.get('no_repeat_ngram_size', 5)
        temperature = kwargs.get('temperature', 0.7)
        top_k = kwargs.get('top_k', 40)
        top_p = kwargs.get('top_p', 0.8)

        self.default_params = {
            'max_length': max_tokens,
            'do_sample': True,
            'repetition_penalty': repetition_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
        }

        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_safetensors=True)
        self.generation_config.update(**self.default_params)
        self.model.generation_config = self.generation_config

    def predict(self, kwargs):
        messages = kwargs.get('messages')
        model_name = kwargs.get('model')

        prompt = messages[-1]['content']
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        input0_len = input_ids.size()[1]
        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids.to(self.default_device), **self.default_params)

        new_tokens = generate_ids[0, input0_len:]
        response = self.tokenizer.decode(new_tokens,
                                         skip_special_tokens=True)

        choice_data = ChatCompletionResponseChoice(index=0,
                                                   message=ChatMessage(
                                                       role='assistant',
                                                       content=response),
                                                   finish_reason='stop')

        result = ChatCompletionResponse(model=model_name,
                                        choices=[choice_data],
                                        object='chat.completion')

        torch_gc(self.devices)
        return result.dict()

    def completion(self, kwargs):
        pass
