from pybackend_libs.dataelem.model import XverseChat


def test_xverse_13b_chat():
    params = {
        'pretrain_path': '/home/public/llm/XVERSE-13B-Chat',
        'devices': '5,6',
        'gpu_memory': 30,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = XverseChat(**params)
    inp = {
        'model':
        'xverse_13b_chat',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手。'
        }, {
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '我是数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁创建的'
        }],
    }
    outp = model.predict(inp)
    print(outp)


test_xverse_13b_chat()
