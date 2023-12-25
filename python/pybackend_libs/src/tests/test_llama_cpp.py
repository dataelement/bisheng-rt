# flake8: noqa
import multiprocessing
import time
from typing import (Callable, Coroutine, Dict, Iterator, List, Optional, Tuple,
                    Union)

import llama_cpp
from llama_cpp import Llama
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing_extensions import Literal, TypedDict

# Disable warning for model and model_alias settings
BaseSettings.model_config['protected_namespaces'] = ()

class Settings(BaseSettings):
    model: str = Field(
        description='The path to the model to use for generating completions.'
    )
    model_alias: Optional[str] = Field(
        default=None,
        description='The alias of the model to use for generating completions.',
    )
    # Model Params
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description='The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.',
    )
    main_gpu: int = Field(
        default=0,
        ge=0,
        description='Main GPU to use.',
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description='Split layers across multiple GPUs in proportion.',
    )
    vocab_only: bool = Field(
        default=False, description='Whether to only return the vocabulary.'
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_mmap_supported(),
        description='Use mmap.',
    )
    use_mlock: bool = Field(
        default=llama_cpp.llama_mlock_supported(),
        description='Use mlock.',
    )
    # Context Params
    seed: int = Field(
        default=llama_cpp.LLAMA_DEFAULT_SEED, description='Random seed. -1 for random.'
    )
    n_ctx: int = Field(default=2048, ge=1, description='The context size.')
    n_batch: int = Field(
        default=512, ge=1, description='The batch size to use per eval.'
    )
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description='The number of threads to use.',
    )
    n_threads_batch: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=0,
        description='The number of threads to use when batch processing.',
    )
    rope_scaling_type: int = Field(default=llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED)
    rope_freq_base: float = Field(default=0.0, description='RoPE base frequency')
    rope_freq_scale: float = Field(
        default=0.0, description='RoPE frequency scaling factor'
    )
    yarn_ext_factor: float = Field(default=-1.0)
    yarn_attn_factor: float = Field(default=1.0)
    yarn_beta_fast: float = Field(default=32.0)
    yarn_beta_slow: float = Field(default=1.0)
    yarn_orig_ctx: int = Field(default=0)
    mul_mat_q: bool = Field(
        default=True, description='if true, use experimental mul_mat_q kernels'
    )
    f16_kv: bool = Field(default=True, description='Whether to use f16 key/value.')
    logits_all: bool = Field(default=True, description='Whether to return logits.')
    embedding: bool = Field(default=True, description='Whether to use embeddings.')
    # Sampling Params
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description='Last n tokens to keep for repeat penalty calculation.',
    )
    # LoRA Params
    lora_base: Optional[str] = Field(
        default=None,
        description='Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.',
    )
    lora_path: Optional[str] = Field(
        default=None,
        description='Path to a LoRA file to apply to the model.',
    )
    # Backend Params
    numa: bool = Field(
        default=False,
        description='Enable NUMA support.',
    )
    # Chat Format Params
    chat_format: str = Field(
        default='llama-2',
        description='Chat format to use.',
    )
    clip_model_path: Optional[str] = Field(
        default=None,
        description='Path to a CLIP model to use for multi-modal chat completion.',
    )
    # Cache Params
    cache: bool = Field(
        default=False,
        description='Use a cache to reduce processing times for evaluated prompts.',
    )
    cache_type: Literal['ram', 'disk'] = Field(
        default='ram',
        description='The type of cache to use. Only used if cache is True.',
    )
    cache_size: int = Field(
        default=2 << 30,
        description='The size of the cache in bytes. Only used if cache is True.',
    )
    # Misc
    verbose: bool = Field(
        default=True, description='Whether to print debug information.'
    )
    # Server Params
    host: str = Field(default='localhost', description='Listen address')
    port: int = Field(default=8000, description='Listen port')
    interrupt_requests: bool = Field(
        default=True,
        description='Whether to interrupt requests when a new request is received.',
    )

class Timer(object):

    def __init__(self):
        self.tic()
        self.elapses = []

    def tic(self):
        self.tstart = time.time()

    def toc(self, reset=True, memorize=True):
        elapse = round(time.time() - self.tstart, 3)
        if memorize:
            self.elapses.append(elapse)

        if reset:
            self.tic()

    def get(self):
        n = round(sum(self.elapses), 3)
        elapses = self.elapses + [
            n,
        ]
        return elapses


def create_model(settings: Optional[Settings]=None, chat_handler=None):
    if settings is None:
        settings = Settings()

    llama = llama_cpp.Llama(
        model_path=settings.model,
        # Model Params
        n_gpu_layers=settings.n_gpu_layers,
        main_gpu=settings.main_gpu,
        tensor_split=settings.tensor_split,
        vocab_only=settings.vocab_only,
        use_mmap=settings.use_mmap,
        use_mlock=settings.use_mlock,
        # Context Params
        seed=settings.seed,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        n_threads=settings.n_threads,
        n_threads_batch=settings.n_threads_batch,
        rope_scaling_type=settings.rope_scaling_type,
        rope_freq_base=settings.rope_freq_base,
        rope_freq_scale=settings.rope_freq_scale,
        yarn_ext_factor=settings.yarn_ext_factor,
        yarn_attn_factor=settings.yarn_attn_factor,
        yarn_beta_fast=settings.yarn_beta_fast,
        yarn_beta_slow=settings.yarn_beta_slow,
        yarn_orig_ctx=settings.yarn_orig_ctx,
        mul_mat_q=settings.mul_mat_q,
        f16_kv=settings.f16_kv,
        logits_all=settings.logits_all,
        embedding=settings.embedding,
        # Sampling Params
        last_n_tokens_size=settings.last_n_tokens_size,
        # LoRA Params
        lora_base=settings.lora_base,
        lora_path=settings.lora_path,
        # Backend Params
        numa=settings.numa,
        # Chat Format Params
        chat_format=settings.chat_format,
        chat_handler=chat_handler,
        # Misc
        verbose=settings.verbose,
    )
    if settings.cache:
        if settings.cache_type == 'disk':
            if settings.verbose:
                print(f'Using disk cache with size {settings.cache_size}')
            cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
        else:
            if settings.verbose:
                print(f'Using ram cache with size {settings.cache_size}')
            cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)

        cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
        llama.set_cache(cache)

    return llama


def test1():

  model_path = ('/public/bisheng/model_repository/llama2-7b-chat-hf-4b-gguf/'
                'ggml-model-Q4_0.gguf')

  settings = Settings(model=model_path, chat_format='llama-2')
  llama = create_model(settings, chat_handler=None)

  msgs1 =  [
    {'role': 'system',
     'content': 'You are an assistant who perfectly describes images.'},
    {'role': 'user', 'content': 'Describe this image in detail please.'}
  ]

  msgs2 =  [
    {'role': 'system',
     'content': 'You are an assistant who perfectly describes images.'},
    {'role': 'user', 'content': 'Describe the world war II.'}
  ]

  # warmup
  for _ in range(3):
    llama.create_chat_completion(messages=msgs1)

  # speed
  timer = Timer()
  total_tokens = 0
  for _ in range(5):
    res = llama.create_chat_completion(messages=msgs2)
    total_tokens += res['usage']['total_tokens']
    timer.toc()

  ts = timer.get()
  tps = total_tokens / ts[-1]
  print('elapse', ts, tps)


test1()
