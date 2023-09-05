from pybackend_libs.dataelem.model import Llama2Chat


def test_llama2_7b_chat():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/home/public/projects/models/Llama-2-7b-chat-hf',
        'devices': '8',
        'gpu_memory': 20,
        'precision': 'fp16',
        'max_tokens': 256,
    }

    model = Llama2Chat(**params)

    inp2 = {
        'model':
        'llama2_7b_chat',
        'messages': [
            {
                'role': 'user',
                'content': 'Hey, are you conscious? Can you talk to me?'
            },
        ]
    }

    inp1 = {
        'model':
        'llama2_7b_chat',
        'messages': [
            {
                'role': 'system',
                'content': 'Always answer with Haiku'
            },
            {
                'role': 'user',
                'content': 'I am going to Paris, what should I see?'
            },
        ]
    }

    inp = {
        'model':
        'llama2_7b_chat',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '我是数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁创造的'
        }],
    }

    outp = model.predict(inp)
    print(outp)


def test_llama2_13b_chat():
    # params = {'pretrain_path': '/home/public/llm/Baichuan-13B-Chat'}
    params = {
        'pretrain_path': '/home/public/projects/models/Llama-2-13b-chat-hf',
        'devices': '7,8',
        'gpu_memory': 30,
        'precision': 'fp16',
        'max_tokens': 512,
    }

    model = Llama2Chat(**params)

    inp = {
        'model':
        'llama2_13b_chat',
        'messages': [{
            'role': 'system',
            'content': '你是来自数据项素创建的一个人工智能助手, 请用中文回答问题'
        }, {
            'role': 'user',
            'content': '你是谁'
        }, {
            'role': 'assistant',
            'content': '我是数据项素创建的一个人工智能助手'
        }, {
            'role': 'user',
            'content': '你是谁创造的'
        }],
    }

    outp = model.predict(inp)
    print(outp)


# test_llama2_7b_chat()
test_llama2_13b_chat()
