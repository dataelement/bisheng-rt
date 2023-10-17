import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

from .conversation import get_gen_prompt


class GenerateParams(BaseModel):
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
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
    id: str = Field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
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

        self.verbose = bool(parameters.get('verbose', '0'))

        if self.verbose:
            print('Cuda support:', torch.cuda.is_available(), ':',
                  torch.cuda.device_count(), 'devices')

        model_type = parameters.get('model_type')
        model_path = parameters.get('model_path')
        pymodel_params = json.loads(parameters.get('pymodel_params', '{}'))
        disable_log_requests = pymodel_params.pop('disable_log_requests',
                                                  'true')
        max_num_seqs = pymodel_params.pop('max_num_seqs', 256)
        max_num_batched_tokens = pymodel_params.pop('max_num_batched_tokens',
                                                    None)
        gpu_memory_utilization = pymodel_params.pop('gpu_memory_utilization',
                                                    0.9)
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
        if self.verbose:
            print('prompt', [prompt])

        sampling_parameters = request.sampling_parameters
        gen_params = self.generate_params.copy(update=sampling_parameters)
        sampling_params = SamplingParams(**gen_params.dict())

        async for output in self.llm_engine.generate(prompt, sampling_params,
                                                     request_id):
            yield output

    def make_response(self, vllm_output):
        choices = []
        for output in vllm_output.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(vllm_output.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in vllm_output.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        resp = ChatCompletionResponse(model=self.model_name,
                                      choices=choices,
                                      object='chat.completion',
                                      usage=usage)
        return resp.dict()

    def predict(self, kwargs):
        raise Exception('not implemented')

    def completion(self, kwargs):
        raise Exception('not implemented')
