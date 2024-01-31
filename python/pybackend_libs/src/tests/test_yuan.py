def test_yuan():
    from pybackend_libs.dataelem.model.llm.yuan import YuanBase
    params = {
        'pretrain_path': '/opt/bisheng-rt/models/model_repository/Yuan2-2B-Janus-hf',
        'devices': '2,3',
        'gpu_memory': 20,
        'max_tokens': 8192,
    }

    model = YuanBase(**params)
    inp = {
        'model':
        'Yuan2-2B-Janus-hf',
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

test_yuan()