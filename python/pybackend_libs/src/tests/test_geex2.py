from pybackend_libs.dataelem.model import CodeGeeX2


def test_code_geex2():
    params = {
        'pretrain_path': '/home/public/llm/codegeex2-6b',
        'devices': '5,6',
        'gpu_memory': 20,
        'max_tokens': 1024,
        'engine_type': 'hf',
    }

    model = CodeGeeX2(**params)
    inp = {
        'model':
        'codegeex2_6b',
        'prompt': '# language: Python\n# write a bubble sort function\n',
    }
    outp = model.predict(inp)
    print(outp['choices'][0]['text'])


test_code_geex2()
