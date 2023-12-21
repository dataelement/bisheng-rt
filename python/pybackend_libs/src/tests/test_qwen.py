# flake8: noqa
import json

from pybackend_libs.dataelem.model.llm.qwen import QwenChat


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


def test_qwen_function_call():
    params = {
        'pretrain_path': '/home/public/llm/Qwen-1_8B-Chat',
        'devices': '1',
        'gpu_memory': 20,
        'max_tokens': 8192,
    }

    model = QwenChat(**params)
    # inp = {
    #     'model':
    #     'qwen-1_8B',
    #     'messages': [{'role': 'system',
    #                   'content': 'You are a helpful AI assistant.'},
    #                  {'role': 'user',
    #                   'content': '今天上海天气怎么样？'}],
    #     'functions': [{'name': 'Search',
    #                    'description': 'useful for when you need to answer questions about current events. You should ask targeted questions',
    #                    'parameters': {'properties': {'__arg1': {'title': '__arg1', 'type': 'string'}}, 'required': ['__arg1'], 'type': 'object'}},
    #                   {'name': 'Calculator',
    #                    'description': 'useful for when you need to answer questions about math',
    #                    'parameters': {'properties': {'__arg1': {'title': '__arg1', 'type': 'string'}}, 'required': ['__arg1'], 'type': 'object'}}],
    # }

    inp = {
        'model':
        'qwen-1_8B',
        'messages': [
                        {'role': 'user', 'content': '你好'},
                        {'role': 'assistant', 'content': '你好！很高兴见到你。有什么我可以帮忙的吗？'},
                        {
                            'role': 'user',
                            'content': '波士顿天气如何？',
                        },
                        {
                            'role': 'assistant',
                            'content': '',
                            'function_call': {
                                'name': 'get_current_weather',
                                'arguments': '{"location": "Boston, MA"}',
                            },
                        },
                        {
                            'role': 'function',
                            'name': 'get_current_weather',
                            'content': '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
                        },
                        {
                            'role': 'user',
                            'content': '北京天气如何？',
                        },
                        {
                            'role': 'assistant',
                            'content': '',
                            'function_call': {
                                'name': 'get_current_weather',
                                'arguments': '{"location": "beijing"}',
                            },
                        },
                        {
                            'role': 'function',
                            'name': 'get_current_weather',
                            'content': '{"temperature": "25", "unit": "celsius", "description": "Sunny"}',
                        },
                    ],
        'functions': [
                        {
                            'name': 'get_current_weather',
                            'description': 'Get the current weather in a given location.',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'location': {
                                        'type': 'string',
                                        'description': 'The city and state, e.g. San Francisco, CA',
                                    },
                                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']},
                                },
                                'required': ['location'],
                            },
                        }
                    ],
    }

    outp = model.predict(inp)
    print(outp)


# def test_qwen_stream_chat():
#     params = {
#         'pretrain_path': '/home/public/llm/Qwen-1_8B-Chat',
#         'devices': '1',
#         'gpu_memory': 20,
#         'max_tokens': 8192,
#     }

#     model = QwenChat(**params)

#     inp = {
#         'model': 'qwen-1_8B',
#         'stream': True,
#         'messages': [{'role': 'system',
#                       'content': 'You are a helpful AI assistant.'},
#                      {'role': 'user',
#                       'content': '今天上海天气怎么样？'}],
#     }
#     for outp in model.stream_predict(inp):
#         print(outp)


# test_qwen_7b_chat()
test_qwen_function_call()
# test_qwen_stream_chat()
