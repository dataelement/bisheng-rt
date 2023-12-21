from pybackend_libs.dataelem.model.llm.qwen import QwenChat


def test_qwen_7b_chat():
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Qwen-7B-Chat',
        'devices': '4',
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


def test_qwen_14b_chat():
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Qwen-14B-Chat',
        'devices': '4,5',
        'gpu_memory': 40,
        'max_tokens': 256,
    }

    model = QwenChat(**params)
    inp = {
        'model':
        'qwen-14b',
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


# test_qwen_7b_chat()
test_qwen_14b_chat()
