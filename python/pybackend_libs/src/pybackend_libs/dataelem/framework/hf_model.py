from typing import Any, Dict, List, Literal, Optional, Union

import torch
from accelerate import (dispatch_model, infer_auto_device_map,
                        init_empty_weights)
from pydantic import BaseModel, Field
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer)
from transformers.generation.utils import GenerationConfig


def torch_gc(devices):
    if torch.cuda.is_available():
        for device_id in devices:
            with torch.cuda.device(f'cuda:{device_id}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def torch_seed(seed=1947):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


class HFModel(object):
    def load(self,
             pretrain_path,
             precision,
             devices,
             gpu_memory,
             use_auto_model=False,
             use_safetensors=False,
             auto_configure_device_map=None,
             use_dispatch=False,
             **kwargs):
        torch_seed()

        memory_per_device = int(int(gpu_memory) / len(devices))
        memory = f'{memory_per_device}GiB'
        max_memory = {int(device_id): memory for device_id in devices}

        auto_model_cls = AutoModel if use_auto_model else AutoModelForCausalLM

        if not use_auto_model:
            self.generation_config = GenerationConfig.from_pretrained(
                pretrain_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path,
                                                       use_fast=False,
                                                       trust_remote_code=True)
        with init_empty_weights():
            config = AutoConfig.from_pretrained(pretrain_path,
                                                trust_remote_code=True)
            model = auto_model_cls.from_config(config,
                                               torch_dtype=torch.float16,
                                               trust_remote_code=True)

        model.tie_weights()
        no_split_modules = model._no_split_modules
        if auto_configure_device_map is None:
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_modules)
        else:
            device_map = auto_configure_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_modules)

        if use_dispatch:
            model = auto_model_cls.from_pretrained(
                pretrain_path,
                trust_remote_code=True,
                use_safetensors=use_safetensors,
                **kwargs).half()
            self.model = dispatch_model(model, device_map=device_map)
        else:
            self.model = auto_model_cls.from_pretrained(
                pretrain_path,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_safetensors=use_safetensors,
                **kwargs)

        self.model.eval()
