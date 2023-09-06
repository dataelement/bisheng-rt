# import os
from typing import Dict

import numpy as np


def auto_configure_device_map(model, max_memory,
                              no_split_module_classes) -> Dict[str, int]:

    num_trans_layers = 32
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
