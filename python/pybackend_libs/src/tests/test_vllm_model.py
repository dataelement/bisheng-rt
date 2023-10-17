import asyncio
import json

from pybackend_libs.dataelem.model.vllm.vllm_model import VLLMModel


async def run(model, inp, request_id):
    seq = 0
    async for vllm_output in model.generate(inp, request_id):
        for output in vllm_output.outputs:
            print(seq, output.index, output.text, output.finish_reason)
        seq += 1


def test_vllm_chat():
    pymodel_params = {
        'gpu_memory_utilization': 0.6,
        'temperature': 0.0,
        'stop': ['<|im_end|>', '<|im_start|>']
    }

    params = {
        'model_path': '/home/public/llm/Qwen-7B-Chat',
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

    # outp = model.predict(inp)
    # print(outp)


test_vllm_chat()
