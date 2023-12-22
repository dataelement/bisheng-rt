import asyncio
import json

from pybackend_libs.dataelem.model.vllm.vllm_model import (VLLMModel,
                                                           get_gen_prompt)


def test_message_to_prompt():
    model_types = [
        'Qwen7bChat', 'BaichuanChat', 'Baichuan2Chat', 'Chatglm2', 'Llama2',
        'InternlmChat', 'Chatglm3',
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


async def run(model, inp, request_id, model_name='qwen-7b-chat'):
    seq = 0
    previous_texts = ['']
    async for vllm_output in await model.generate(inp, request_id):
        # for output in vllm_output.outputs:
        #     print(seq, output.index, output.text, output.finish_reason)
        # seq += 1

        resp = model.make_response(vllm_output,
                                   previous_texts,
                                   model_name=model_name)
        for output in vllm_output.outputs:
            previous_texts[output.index] = output.text
        print(seq, resp)
        seq += 1


def test_vllm_qwen7b_chat():
    pymodel_params = {
        'gpu_memory_utilization': 0.8,
        'temperature': 0.0,
        'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>']
    }

    params = {
        'pretrain_path': '/public/bisheng/model_repository/Qwen-7B-Chat',
        'devices': '4,5',
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
    asyncio.run(run(model, inp, '0001', 'qwen-7b-chat'))


def test_vllm_qwen14b_chat():
    pymodel_params = {
        'gpu_memory_utilization': 0.9,
        'temperature': 0.0,
        'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>']
    }

    params = {
        'pretrain_path': '/public/bisheng/model_repository/Qwen-14B-Chat',
        'devices': '4,5',
        'gpu_memory': '40',
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
    asyncio.run(run(model, inp, '0001', 'qwen-14b-chat'))


def test_vllm_chatglm2_chat():
    # https://github.com/vllm-project/vllm/pull/649
    # not merge into main yet, 2023.10.19, wait release.
    pymodel_params = {
        'gpu_memory_utilization': 0.9,
        'temperature': 0.0,
        'stop': ['<eos>', '<pad>', '<bos>']
    }

    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm2-6b',
        'devices': '4',
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

    asyncio.run(run(model, inp, '0001', 'chatglm2'))


def test_vllm_baichuan2_13b_chat():
    pymodel_params = {
        'temperature': 0.0,
        'gpu_memory_utilization': 0.9,
        'stop': ['<reserved_107>', '<reserved_106>'],
    }

    params = {
        'pretrain_path': '/public/bisheng/model_repository/Baichuan2-13B-Chat',
        'devices': '4,5',
        'gpu_memory': 40,
        'precision': 'fp16',
        'max_tokens': 256,
        'model_type': 'vLLMBaichuan2Chat',
        'pymodel_params': json.dumps(pymodel_params),
        'verbose': '1'
    }

    print('params', [json.dumps(pymodel_params)])
    # return

    model = VLLMModel(**params)
    inp = {
        'model': 'Baichuan2-13B-Chat',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    asyncio.run(run(model, inp, '0001', 'baichuan2-13b-chat'))


def test_vllm_chatglm3_chat():
    pymodel_params = {
        'temperature': 0.0,
        'gpu_memory_utilization': 0.9,
        'stop': ['<|user|>', '<|observation|>', '</s>'],
    }

    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm3-6b',
        'devices': '4',
        'gpu_memory': 20,
        'precision': 'fp16',
        'max_tokens': 256,
        'model_type': 'vLLMChatglm3',
        'pymodel_params': json.dumps(pymodel_params),
        'verbose': '1'
    }

    print('params', [json.dumps(pymodel_params)])
    # return

    model = VLLMModel(**params)
    inp = {
        'model': 'chatglm3',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    asyncio.run(run(model, inp, '0001', 'chatglm3'))


# test_message_to_prompt()
# test_vllm_qwen7b_chat()
# test_vllm_qwen14b_chat()
# test_vllm_chatglm2_chat()
# test_vllm_baichuan2_13b_chat()
# test_vllm_chatglm3_chat()
