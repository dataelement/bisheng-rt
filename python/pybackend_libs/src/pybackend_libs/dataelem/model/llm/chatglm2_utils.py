import os
from typing import Dict, Optional, Tuple, Union

import numpy as np


def auto_configure_device_map(model, max_memory,
                              no_split_module_classes) -> Dict[str, int]:

    devices = sorted(list(max_memory.keys()))
    num_gpus = len(devices)

    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2

    device_first = devices[0]
    device_last = devices[-1]
    device_map = {
        'transformer.embedding.word_embeddings': device_first,
        'transformer.rotary_pos_emb': device_first,
        'transformer.encoder.final_layernorm': device_last,
        'transformer.output_layer': device_last,
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = devices[gpu_target]
        used += 1

    return device_map
