from pybackend_libs.dataelem.model.llm.chatglm2 import ChatGLM2


def test_chatglm2_6b():
    # params = {'pretrain_path': '/home/public/llm/chatglm2-6b'}
    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm2-6b',
        'devices': '4,5',
        'gpu_memory': 20,
        'max_tokens': 256,
    }

    model = ChatGLM2(**params)
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

    outp = model.predict(inp)
    print(outp)


def test_chatglm2_6b_stream():
    # params = {'pretrain_path': '/home/public/llm/chatglm2-6b'}
    params = {
        'pretrain_path': '/public/bisheng/model_repository/chatglm2-6b',
        'devices': '6,7',
        'gpu_memory': 20,
        'max_tokens': 128,
    }

    model = ChatGLM2(**params)
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

    for outp in model.stream_predict(inp):
        print(outp)


# test_chatglm2_6b()
test_chatglm2_6b_stream()
