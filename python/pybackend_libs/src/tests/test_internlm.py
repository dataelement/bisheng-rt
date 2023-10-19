from pybackend_libs.dataelem.model import InternLMChat


def test_internlm_7b_chat():
    params = {
        'pretrain_path': '/home/public/llm/internlm-chat-7b-8k',
        'devices': '5,6',
        'gpu_memory': 20,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = InternLMChat(**params)
    inp = {
        'model':
        'internlm_7b_chat',
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


def test_internlm_20b_chat():
    params = {
        'pretrain_path': '/home/public/llm/internlm-20b-chat/',
        'devices': '3,4,5,6',
        'gpu_memory': 60,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = InternLMChat(**params)
    inp = {
        'model':
        'internlm_20b_chat',
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


test_internlm_7b_chat()
# test_internlm_20b_chat()
