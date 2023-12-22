# flake8: noqa
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

    print('==============================')
    inp = {
        'model':
        'chatglm_6b',
        'messages': [
            {'role': 'system',
             'content': 'Answer the following questions as best as you can. You have access to the following tools:',
             'tools': [{'name': 'Calculator',
                        'description': 'Useful for when you need to answer questions about math',
                        'parameters': {
                                        'type': 'object',
                                        'properties': {
                                            'formula': {'type': 'string',
                                                        'description': 'The formula to be calculated'
                                                       }
                                        },
                                        'required': ['formula']
                                      }
                       }]
            },
            {'role': 'user', 'content': '12345679乘以54等于多少？'},
            {'role': 'assistant', 'metadata': 'Calculator', 'content': " ```python\ntool_call(formula='12345679*54')\n```"},
            {'role': 'observation', 'content': '666666666'},
            {'role': 'user', 'content': ''},
        ],
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
