import asyncio
import json

from pybackend_libs.dataelem.model.vllm.vllm_model import (VLLMModel,
                                                           get_gen_prompt)


def test_message_to_prompt():
    model_types = [
        'Qwen7bChat', 'BaichuanChat', 'Baichuan2Chat', 'Chatglm2', 'Llama2',
        'InternlmChat'
    ]
    messages = [{
        'role': 'system',
        'content': '你是人工智能助手'
    }, {
        'role': 'user',
        'content': '你能做什么'
    }, {
        'role': 'assistant',
        'content': '我可以回答各种问题'
    }, {
        'role': 'user',
        'content': '你是谁'
    }]
    for t in model_types:
        print(t, [get_gen_prompt(t, messages)])


async def run(model, inp, request_id):
    seq = 0
    previous_texts = ['']
    async for vllm_output in model.generate(inp, request_id):
        # for output in vllm_output.outputs:
        #     print(seq, output.index, output.text, output.finish_reason)
        # seq += 1

        resp = model.make_response(vllm_output,
                                   previous_texts,
                                   model_name='qwen-7b-chat')
        for output in vllm_output.outputs:
            previous_texts[output.index] = output.text
        print(seq, resp)
        seq += 1


def test_vllm_qwen7b_chat():
    pymodel_params = {
        'gpu_memory_utilization': 0.6,
        'temperature': 0.0,
        'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>']
    }

    params = {
        'pretrain_path': '/home/public/llm/Qwen-7B-Chat',
        'devices': '3,8',
        'gpu_memory': '30',
        'model_type': 'vLLMQwen7bChat',
        'pymodel_params': json.dumps(pymodel_params),
        'verbose': '1'
    }

    model = VLLMModel(**params)
    inp = {
        'model': 'qwen-7b',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    # task = asyncio.create_task(run()))
    asyncio.run(run(model, inp, '0001'))


def test_vllm_qwen14b_chat():
    pymodel_params = {
        'gpu_memory_utilization': 0.55,
        'temperature': 0.0,
        'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>']
    }

    params = {
        'pretrain_path': '/home/public/llm/Qwen-14B-Chat',
        'devices': '3,5,6,8',
        'gpu_memory': '50',
        'model_type': 'vLLMQwen7bChat',
        'pymodel_params': json.dumps(pymodel_params),
        'verbose': '1'
    }

    model = VLLMModel(**params)
    inp = {
        'model': 'qwen-14b',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    # task = asyncio.create_task(run()))
    asyncio.run(run(model, inp, '0001'))


def test_vllm_chatglm2_chat():
    # https://github.com/vllm-project/vllm/pull/649
    # not merge into main yet, 2023.10.19, wait release.
    pymodel_params = {
        'gpu_memory_utilization': 0.6,
        'temperature': 0.0,
        'stop': ['<eos>', '<pad>', '<bos>']
    }

    params = {
        'pretrain_path': '/home/public/llm/chatglm2-6b',
        'devices': '3,8',
        'gpu_memory': '30',
        'model_type': 'vLLMChatglm2',
        'pymodel_params': json.dumps(pymodel_params),
        'verbose': '1'
    }

    model = VLLMModel(**params)
    inp = {
        'model': 'chatglm2-6b',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    asyncio.run(run(model, inp, '0001'))


# test_message_to_prompt()
# test_vllm_qwen7b_chat()
test_vllm_qwen14b_chat()
# test_vllm_chatglm2_chat()
