
def test_qwen1_5_14b():
    from pybackend_libs.dataelem.model.llm.qwen1_5 import  Qwen1_5Chat
    params = {
        'pretrain_path': '/opt/bisheng-rt/models/model_repository/Qwen1.5-14B-Chat',
        'devices': '1,2,3,4',
        'gpu_memory': 28,
        'max_tokens': 8192,
    }

    model = Qwen1_5Chat(**params)
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

def test_stream_qwen1_5_14b():
    from pybackend_libs.dataelem.model.llm.qwen1_5 import  Qwen1_5Chat
    params = {
        'pretrain_path': '/opt/bisheng-rt/models/model_repository/Qwen1.5-14B-Chat',
        'devices': '1,2,3,4',
        'gpu_memory': 28,
        'max_tokens': 8192,
    }

    model = Qwen1_5Chat(**params)
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
    for outp in model.stream_predict(inp):
        print(outp)