# import copy
# import time

import torch
import copy
from .llm import (BaseLLM, ChatCompletionResponse,
                  ChatCompletionResponseChoice, ChatMessage, torch_gc,
                  ChatCompletionRequest, ChatCompletionResponseStreamChoice,
                  DeltaMessage)
from transformers import TextStreamer, TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria

from threading import Thread

class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = (
            [2, 6, 7, 8],
        )  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

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
        # input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, 
                                                tokenize=True, 
                                                add_generation_prompt=True, 
                                                return_tensors='pt')
        generate_ids = self.model.generate(
            input_ids.to(self.default_device), **self.default_params)

        # new_tokens = generate_ids[0, input0_len:]
        response = self.tokenizer.decode(generate_ids[0][input_ids.shape[1]:],
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

    
    def stream_predict(self, kwargs):
        
        messages = kwargs.get('messages')
        model_name = kwargs.get('model')


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

        current_length = 0

        model_inputs = self.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
        )

        stop = StopOnTokens()
        generate_kwargs = {
            "input_ids": model_inputs,
            "streamer": streamer,
            "do_sample": True,
            "stopping_criteria": StoppingCriteriaList([stop]),
        }
        generate_kwargs.update(self.default_params)

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        for new_token in streamer:
            if new_token != "":
                # yield new_token

                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=new_token),
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
