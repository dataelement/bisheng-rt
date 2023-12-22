from pybackend_libs.dataelem.model.llm.baichuan import BaichuanChat


def test_baichuan_13b_chat():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Baichuan-13B-Chat',
        'devices': '4,5',
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


def test_baichuan2_13b_chat():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Baichuan2-13B-Chat',
        'devices': '4,5',
        'gpu_memory': 40,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = BaichuanChat(**params)
    inp = {
        'model':
        'baichua2_13b_chat',
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


def test_case2():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Baichuan2-13B-Chat',
        'devices': '4,5',
        'gpu_memory': 40,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = BaichuanChat(**params)
    inp1 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    inp2 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你是哪家公司研发的'
        }],
    }

    inp3 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你能做什么'
        }],
    }

    print(model.predict(inp1))
    print(model.predict(inp2))
    print(model.predict(inp3))


def test_stream():
    params = {
        'pretrain_path': '/public/bisheng/model_repository/Baichuan2-13B-Chat',
        'devices': '0,2',
        'gpu_memory': 40,
        'precision': 'fp16',
        'max_tokens': 256,
    }
    model = BaichuanChat(**params)
    inp1 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }

    inp2 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你是哪家公司研发的'
        }],
    }

    inp3 = {
        'model': 'baichua2_13b_chat',
        'messages': [{
            'role': 'user',
            'content': '你能做什么'
        }],
    }

    for outp in model.stream_predict(inp1):
        print(outp)

    for outp in model.stream_predict(inp2):
        print(outp)

    for outp in model.stream_predict(inp3):
        print(outp)


# test_baichuan_13b_chat()
# test_baichuan2_13b_chat()
# test_case2()
test_stream()
