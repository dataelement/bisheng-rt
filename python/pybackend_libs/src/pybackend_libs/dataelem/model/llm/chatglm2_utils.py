# import os
from typing import Dict

# import numpy as np


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

    # refer https://github.com/THUDM/ChatGLM2-6B/issues/421

    device_first = devices[0]
    device_last = devices[-1]
    device_map = {
        'transformer.embedding.word_embeddings': device_first,
        'transformer.rotary_pos_emb': device_first,
        'transformer.encoder.final_layernorm': device_last,
        'transformer.output_layer': device_last,
        'transformer.prefix_encoder.embedding': device_first,
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
