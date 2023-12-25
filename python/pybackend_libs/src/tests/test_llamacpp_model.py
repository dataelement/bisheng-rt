# flake8: noqa
import json

from pybackend_libs.dataelem.model.llama_cpp.lc_model import LlamaCppModel


def test_qwen_int4():
    pymodel_params = {
        'n_threads': 20,
        'n_threads_batch':20,
        'chat_format': 'qwen',
        'top_p': 0.8,
        'max_tokens': 2048,
        'model_ftype': 'q4_0',
        'stop': ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '\n\n\n']
    }

    params = {
        'pretrain_path':
            '/public/bisheng/model_repository/Qwen-1_8B-Chat-4b-gguf',
        # 'pretrain_path':
        #     '/public/bisheng/model_repository/Qwen-7B-Chat-4b-gguf',
        'pymodel_params': json.dumps(pymodel_params)
    }

    model = LlamaCppModel(**params)


    inp = {
        'model': 'qwen-1_8b',
        'messages': [{
            'role': 'system',
            'content': '你是一个作家'
        }, {
            'role': 'user',
            # 'content': '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素'
            'content': 'what can you do?'
        }],
    }

    # for outp in model.stream_predict(inp):
    #     print(outp)

    outp = model.predict(inp)
    print(outp)


test_qwen_int4()
