import json
import multiprocessing
import os
# import time
from typing import Dict, List, Literal, Optional, Union

import llama_cpp
from llama_cpp import Llama
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function', 'observation']
    content: str
    function_call: Optional[Dict] = None
    tools: Optional[List[dict]] = None
    metadata: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    do_sample: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class GenerateParams(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = []
    model: str


class LlamaCppModel(object):
    def __init__(self, **parameters):
        # self.verbose = bool(int(parameters.get('verbose', '0')))

        model_path = parameters.get('pretrain_path')
        pymodel_params = json.loads(parameters.get('pymodel_params', '{}'))
        # dtype = pymodel_params.pop('dtype', 'auto')
        chat_format = pymodel_params.pop('chat_format', 'llama-2')
        n_ctx = pymodel_params.pop('n_ctx', 512)
        n_batch = pymodel_params.pop('n_batch', 512)
        n_threads = pymodel_params.pop('n_threads', None)
        n_threads_batch = pymodel_params.pop('n_threads_batch', None)
        if n_threads is None:
            n_threads = max(multiprocessing.cpu_count() // 2, 1)

        if n_threads_batch is None:
            n_threads_batch = max(multiprocessing.cpu_count() // 2, 1)

        cache_size = pymodel_params.pop('cache_size', '2g')
        cache_size = int(cache_size.split('g')[0]) * 1e8
        numa = pymodel_params.pop('numa', False)

        model_ftype = pymodel_params.pop('model_ftype', 'q4_0')
        model_filename = f'ggml-model-{model_ftype}.gguf'
        model_file = os.path.join(model_path, model_filename)

        # chat_format: llama-2, alpaca, qwen, baichuan-2, baichuan
        self.model = Llama(
            model_path=model_file,
            chat_format=chat_format,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            numa=numa)

        cache = llama_cpp.LlamaRAMCache(capacity_bytes=cache_size)
        self.model.set_cache(cache)

        gen_params = {}
        gen_params.update(top_p=pymodel_params.pop('top_p', 0.95))
        gen_params.update(top_k=pymodel_params.pop('top_k', 40))
        gen_params.update(temperature=pymodel_params.pop('temperature', 0.2))
        gen_params.update(max_tokens=pymodel_params.pop('max_tokens', 2048))
        gen_params.update(stop=pymodel_params.pop('stop', []))

        model_id = os.path.basename(model_path)
        gen_params.update(model=model_id)

        self.generate_params = GenerateParams(**gen_params)

    def predict(self, inp):
        request = ChatCompletionRequest.parse_obj(inp)
        gen_params = {}
        if request.temperature is not None:
            gen_params['temperature'] = request.temperature

        if request.stream is not None:
            gen_params['stream'] = request.stream

        if request.max_tokens is not None:
            gen_params['max_tokens'] = request.max_tokens

        if request.top_p is not None:
            gen_params['top_p'] = request.top_p

        if request.stop is not None:
            stop = request.stop
            if isinstance(stop, str):
                stop = [stop]

            gen_params['stop'] = list(set(self.generate_params['stop'] + stop))

        gen_params = self.generate_params.copy(update=gen_params).dict()

        messages = [m.dict() for m in request.messages]
        result = self.model.create_chat_completion(messages, **gen_params)

        return result

    def stream_predict(self, inp):
        request = ChatCompletionRequest.parse_obj(inp)
        gen_params = {}
        if request.temperature is not None:
            gen_params['temperature'] = request.temperature

        gen_params['stream'] = True

        if request.max_tokens is not None:
            gen_params['max_tokens'] = request.max_tokens

        if request.top_p is not None:
            gen_params['top_p'] = request.top_p

        if request.stop is not None:
            stop = request.stop
            if isinstance(stop, str):
                stop = [stop]

            gen_params['stop'] = list(set(self.generate_params['stop'] + stop))

        gen_params = self.generate_params.copy(update=gen_params).dict()

        messages = [m.dict() for m in request.messages]
        result = self.model.create_chat_completion(messages, **gen_params)

        return result

    def completion(self, kwargs):
        raise Exception('not implemented')
