def test_yi_chat():
    from pybackend_libs.dataelem.model.llm.yi import YiBase
    params = {
        'pretrain_path': '/opt/bisheng-rt/models/model_repository/Yi-34B-Chat',
        'devices': '0,1,2,3',
        'gpu_memory': 80,
        'max_tokens': 8192,
    }

    model = YiBase(**params)
    inp = {
        'model':
        'Yi-34B-Chat',
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
    outp = model.predict(inp)   #非流式
    print(outp)

    inp = {
        'model': 'Yi-34B-Chat',
        'stream': True,
        'messages': [{'role': 'system',
                      'content': 'You are a helpful AI assistant.'},
                     {'role': 'user',
                      'content': '今天上海天气怎么样？'}],
    }
    for outp in model.stream_predict(inp):
        print(outp)
    

test_yi_chat()