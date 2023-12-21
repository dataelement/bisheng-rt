# flake8: noqa
# import os
import copy
import json
import re
from typing import Dict

import numpy as np

from .llm import ChatCompletionResponseChoice, ChatMessage


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


def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""


REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""


_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != 'user' for m in messages):
        raise Exception(
            f'Invalid request: Expecting at least one user message.',
        )

    messages = copy.deepcopy(messages)
    default_system = 'You are a helpful assistant.'
    system = ''
    if messages[0].role == 'system':
        system = messages.pop(0).content.lstrip('\n').rstrip()
        if system == default_system:
            system = ''

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get('name', '')
            name_m = func_info.get('name_for_model', name)
            name_h = func_info.get('name_for_human', name)
            desc = func_info.get('description', '')
            desc_m = func_info.get('description_for_model', desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info['parameters'], ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = '\n\n'.join(tools_text)
        tools_name_text = ', '.join(tools_name_text)
        system += '\n\n' + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip('\n').rstrip()

    dummy_thought = {
        'en': '\nThought: I now know the final answer.\nFinal answer: ',
        'zh': '\nThought: 我会作答了。\nFinal answer: ',
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if content:
            content = content.lstrip('\n').rstrip()
        if role == 'function':
            if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                raise Exception(
                    f'Invalid request: Expecting role assistant before role function.',
                )
            messages[-1].content += f'\nObservation: {content}'
            if m_idx == len(_messages) - 1:
                messages[-1].content += '\nThought:'
        elif role == 'assistant':
            if len(messages) == 0:
                raise Exception(
                    f'Invalid request: Expecting role user before role assistant.',
                )
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r'[\u4e00-\u9fff]+', last_msg)) > 0
            if func_call is None:
                if functions:
                    content = dummy_thought['zh' if last_msg_has_zh else 'en'] + content
            else:
                f_name, f_args = func_call['name'], func_call['arguments']
                if not content:
                    if last_msg_has_zh:
                        content = f'Thought: 我可以使用 {f_name} API。'
                    else:
                        content = f'Thought: I can use {f_name}.'
                content = f'\n{content}\nAction: {f_name}\nAction Input: {f_args}'
            if messages[-1].role == 'user':
                messages.append(
                    ChatMessage(role='assistant', content=content.lstrip('\n').rstrip())
                )
            else:
                messages[-1].content += content
        elif role == 'user':
            messages.append(
                ChatMessage(role='user', content=content.lstrip('\n').rstrip())
            )
        else:
            raise Exception(
                f'Invalid request: Incorrect role {role}.'
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == 'user':
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise Exception('Invalid request')

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
            usr_msg = messages[i].content.lstrip('\n').rstrip()
            bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f'{system}\n\nQuestion: {usr_msg}'
                system = ''
            for t in dummy_thought.values():
                t = t.lstrip('\n')
                if bot_msg.startswith(t) and ('\nAction: ' in bot_msg):
                    bot_msg = bot_msg[len(t) :]
            history.append([usr_msg, bot_msg])
        else:
            raise Exception(
                'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f'{system}\n\nQuestion: {query}'
    return query, history


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


def parse_response(response):
    func_name, func_args = '', ''
    i = response.rfind('\nAction:')
    j = response.rfind('\nAction Input:')
    k = response.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + '\nObservation:'  # Add it back.
        k = response.rfind('\nObservation:')
        func_name = response[i + len('\nAction:') : j].strip()
        func_args = response[j + len('\nAction Input:') : k].strip()
    if func_name:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role='assistant',
                content=response[:i],
                function_call={'name': func_name, 'arguments': func_args},
            ),
            finish_reason='function_call',
        )
        return choice_data
    z = response.rfind('\nFinal Answer: ')
    if z >= 0:
        response = response[z + len('\nFinal Answer: ') :]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )
    return choice_data
