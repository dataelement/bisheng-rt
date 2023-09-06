from pybackend_libs.dataelem.model import QwenChat


def test_qwen_7b_chat():
    params = {
        'pretrain_path': '/home/public/projects/models/Qwen-7B-Chat',
        'devices': '8',
        'gpu_memory': 20,
        'max_tokens': 256,
    }

    model = QwenChat(**params)
    inp = {
        'model':
        'qwen-7b',
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


test_qwen_7b_chat()
