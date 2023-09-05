import copy
import time
from typing import Any, Dict, List, Literal, Optional, Union

from .llm import (BaseLLM, ChatCompletionRequest, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc)


class XverseChat(BaseLLM):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'

        temperature = kwargs.get('temperature', 0.3)
        top_p = kwargs.get('top_p', 0.85)
        max_tokens = kwargs.get('max_tokens', 2048)
        do_sample = kwargs.get('do_sample', False)
        self.default_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_length': max_tokens,
            'do_sample': do_sample
        }

        # self.model, self.tokenizer = None, None
        self._load(pretrain_path,
                   precision,
                   devices,
                   gpu_memory,
                   use_generate_config=False)

    def chat(self, **kwargs):
        req_dict = copy.copy(self.default_params)
        req_dict.update(kwargs)

        query = req_dict['messages'][-1].content
        inputs = self.tokenizer(query, return_tensors='pt').input_ids
        inputs = inputs.to(self.default_device)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs,
                max_length=req_dict['max_length'],
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1)

        response = self.tokenizer.batch_decode(generated_ids,
                                               skip_special_tokens=True)[0]

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
