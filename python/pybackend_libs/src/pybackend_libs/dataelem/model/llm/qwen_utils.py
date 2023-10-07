# import os
from typing import Dict

import numpy as np


def auto_configure_device_map2(model,
                               max_memory,
                               no_split_module_classes,
                               num_trans_layers=32,
                               devices=None) -> Dict[str, int]:
    # refer https://github.com/QwenLM/Qwen/blob/main/utils.py
    num_gpus = len(max_memory.keys())
    per_gpu_layers = (num_trans_layers + 2) / num_gpus

    used_devices = [int(d) for d in devices]
    device_map = {
        'transformer.wte': used_devices[0],
        'transformer.ln_f': used_devices[0],
        'lm_head': used_devices[num_gpus - 1]
    }

    used = 1
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus - 1 else 1
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = used_devices[gpu_target]
        used += 1

    return device_map


def auto_configure_device_map(model,
                              max_memory,
                              no_split_module_classes,
                              num_trans_layers=32) -> Dict[str, int]:
    # num_trans_layers = 32
    devices = sorted(list(max_memory.keys()))
    num_gpus = len(devices)
    # per_gpu_layers = num_trans_layers / num_gpus

    layer_names = [
        'transformer.wte',
        'transformer.drop',
    ]
    for i in range(num_trans_layers):
        layer_names.append(f'transformer.h.{i}')

    layer_names.extend(['transformer.ln_f', 'lm_head'])

    layer_params = [[layer, 0] for layer in layer_names]

    for param_name, param in model.named_parameters():
        size = np.prod(list(param.shape))
        layer_index = None
        if 'transformer.h.' in param_name:
            layer_index = int(param_name.split('.')[2])
            layer_index += 2

        else:
            for i, layer_name in enumerate(layer_names):
                if param_name.startswith(layer_name):
                    layer_index = i
                    break

        layer_params[layer_index][1] += size

    total_n = np.sum([t[1] for t in layer_params])
    per_device_cnt = int(np.ceil(total_n / num_gpus))

    groups = []
    group = []
    curr_cnt = 0
    for name, cnt in layer_params:
        curr_cnt += cnt
        if curr_cnt >= per_device_cnt:
            group.append(name)
            groups.append(group)
            group = []
            curr_cnt = 0
        else:
            group.append(name)

    if group:
        groups.append(group)

    assert len(groups) == num_gpus, f'{len(groups)}!={num_gpus}'

    device_map = {}
    for group, device in zip(groups, devices):
        for name in group:
            device_map[name] = device

    return device_map
