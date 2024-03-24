import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .conversation import get_gen_prompt

# from vllm.utils import random_uuid


class GenerateParams(BaseModel):
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: int = 16
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True


class Messages2Prompt(object):
    def __init__(self, model_type):
        self.model_type = model_type

    def run(self, messages):
        messages = [m.dict() for m in messages]
        return get_gen_prompt(self.model_type, messages)


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    do_sample: Optional[bool] = False
    sampling_parameters: Optional[Dict[Any, Any]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: 'chatcmpl-1016')
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    object: str = 'chat.completion'
    model: str
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    usage: UsageInfo = UsageInfo()


class VLLMModel(object):
    def __init__(self, **parameters):
        devices = parameters.get('devices')
        # important, mark which devices to be used
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'

        self.verbose = bool(int(parameters.get('verbose', '0')))

        if self.verbose:
            print('Cuda support:', torch.cuda.is_available(), ':',
                  torch.cuda.device_count(), 'devices')

        model_type = parameters.get('model_type')
        model_path = parameters.get('pretrain_path')
        pymodel_params = json.loads(parameters.get('pymodel_params', '{}'))
        disable_log_requests = pymodel_params.pop('disable_log_requests',
                                                  'true')
        max_num_seqs = pymodel_params.pop('max_num_seqs', 256)
        max_num_batched_tokens = pymodel_params.pop('max_num_batched_tokens',
                                                    None)
        gpu_memory_utilization = pymodel_params.pop('gpu_memory_utilization',
                                                    0.9)
        max_model_len = pymodel_params.pop('max_model_len', 2048)
        block_size = pymodel_params.pop('block_size', 16)
        swap_space = pymodel_params.pop('swap_space', 4)

        dtype = pymodel_params.pop('dtype', 'auto')
        tensor_parallel_size = len(devices.split(','))

        vllm_engine_config = {
            'model': model_path,
            'tokenizer': model_path,
            'disable_log_requests': disable_log_requests,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens,
            'tensor_parallel_size': tensor_parallel_size,
            'dtype': dtype,
            'trust_remote_code': True,
            'block_size': block_size,
            'swap_space': swap_space,
            'max_model_len': max_model_len,
        }
        if self.verbose:
            print('vllm_engine_config', vllm_engine_config)

        model_type = model_type.replace('vLLM', '')
        self.messages_to_prompt = Messages2Prompt(model_type)

        # Create an AsyncLLMEngine from the config from JSON
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**vllm_engine_config))

        gen_params = {}
        gen_params.update(top_p=pymodel_params.pop('top_p', 1.0))
        gen_params.update(temperature=pymodel_params.pop('temperature', 0.7))
        gen_params.update(max_tokens=pymodel_params.pop('max_tokens', 256))
        gen_params.update(stop=pymodel_params.pop('stop', []))

        self.generate_params = GenerateParams(**gen_params)

    async def generate(self, inp, request_id):
        request = ChatCompletionRequest.parse_obj(inp)
        prompt = self.messages_to_prompt.run(request.messages)
        # stream = request.stream
        # For the locale limit, cjk characters can not be printed in sysstd
        # if self.verbose:
        #     print('prompt', [prompt])

        sampling_parameters = request.sampling_parameters
        gen_params = self.generate_params.copy(update=sampling_parameters)
        if request.top_p is not None:
            gen_params.top_p = request.top_p

        if request.temperature is not None:
            gen_params.temperature = request.temperature

        if request.max_tokens is not None:
            gen_params.max_tokens = request.max_tokens

        # support stop
        if request.stop is not None:
            stop = request.stop
            if isinstance(request.stop, str):
                stop = [request.stop]

            gen_params.stop = list(set(gen_params.stop + stop))

        sampling_params = SamplingParams(**gen_params.dict())
        return self.llm_engine.generate(prompt, sampling_params, request_id)

    def make_response(self,
                      vllm_output,
                      previous_texts,
                      stream=True,
                      model_name='chat_llm'):
        choices = []
        has_finish = not stream
        for output in vllm_output.outputs:
            i = output.index
            output_text = output.text[len(previous_texts[i]):]
            if not stream:
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role='assistant', content=output_text),
                    finish_reason=output.finish_reason)
            else:
                if output.finish_reason is not None:
                    has_finish = True
                choice_data = ChatCompletionResponseStreamChoice(
                    index=output.index,
                    delta=DeltaMessage(role='assistant', content=output_text),
                    finish_reason=output.finish_reason)

            choices.append(choice_data)

        if has_finish:
            num_prompt_tokens = len(vllm_output.prompt_token_ids)
            num_generated_tokens = sum(
                len(output.token_ids) for output in vllm_output.outputs)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
        else:
            usage = UsageInfo()

        object_str = 'chat.completion.chunk' if stream else 'chat.completion'
        resp = ChatCompletionResponse(model=model_name,
                                      choices=choices,
                                      object=object_str,
                                      usage=usage)
        return resp.dict()

    def get_n(self):
        return self.generate_params.n

    def predict(self, kwargs):
        raise Exception('not implemented')

    def completion(self, kwargs):
        raise Exception('not implemented')
