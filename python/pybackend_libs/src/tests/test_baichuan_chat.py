from pybackend_libs.dataelem.model import BaichuanChat


def test_baichuan_13b_chat():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/home/hanfeng/projects/models/Baichuan-13B-Chat',
        'devices': '7,8',
        'gpu_memory': 40,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = BaichuanChat(**params)
    inp = {
        'model':
        'baichua_13b_chat',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '我是数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁创造的'
        }],
    }
    outp = model.predict(inp)
    print(outp)


test_baichuan_13b_chat()
