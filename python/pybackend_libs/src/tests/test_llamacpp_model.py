# flake8: noqa
import json

from pybackend_libs.dataelem.model.llama_cpp.lc_model import LlamaCppModel


def test_qwen_int4():
    pymodel_params = {
        'n_threads': 20,
        'n_threads_batch':20,
        'chat_format': 'qwen',
        'top_p': 0.8,
        'max_tokens': 2048
    }

    params = {
        'pretrain_path':
            '/public/bisheng/model_repository/Qwen-1_8B-Chat-4b-gguf',
        'devices': '',
        'gpu_memory': None,
        'precision': 'fp16',
        'max_tokens': 256,
        'pymodel_params': json.dumps(pymodel_params)
    }

    model = LlamaCppModel(**params)


    inp = {
        'model': 'qwen-1_8b',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁'
        }],
    }

    outp = model.predict(inp)
    print(outp)


test_qwen_int4()
