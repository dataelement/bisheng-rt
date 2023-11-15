from pybackend_libs.dataelem.model.llm.yi import YiBase


def test_yi_6B_base():
    params = {
        'pretrain_path': '/home/public/llm/Yi-6B',
        'devices': '8',
        'gpu_memory': 20,
        'max_tokens': 8192,
    }

    model = YiBase(**params)
    inp = {
        'model': 'yi-6b',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '今天天气怎么样?',
        }],
    }

    outp = model.predict(inp)
    print(outp)


test_yi_6B_base()
