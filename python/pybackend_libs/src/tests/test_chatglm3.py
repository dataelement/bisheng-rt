from pybackend_libs.dataelem.model.llm.chatglm3 import ChatGLM3


def test_chatglm3_6b():
    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm3-6b',
        'devices': '4',
        'gpu_memory': 20,
        'max_tokens': 8192,
    }

    model = ChatGLM3(**params)
    inp = {
        'model':
        'chatglm_6b',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你可以做什么'
        }],
    }
    outp = model.predict(inp)
    print(outp)


def test_chatglm3_6b_stream():
    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm3-6b',
        'devices': '0',
        'gpu_memory': 20,
        'max_tokens': 64,
    }

    model = ChatGLM3(**params)
    inp = {
        'model':
        'chatglm_6b',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '我可以做什么'
        }],
        'stop': '我可以',
    }
    for outp in model.stream_predict(inp):
        print(outp)


# test_chatglm3_6b()
test_chatglm3_6b_stream()
