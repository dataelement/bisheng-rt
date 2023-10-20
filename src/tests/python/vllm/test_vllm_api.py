import asyncio

from vllm_api import ChatCompletion


async def test_api():
    api_base = '192.168.106.12:19000'
    inp = {
        'model': 'Qwen-7B-Chat',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
    }
    resp = await ChatCompletion.acreate(api_base, **inp)
    print(resp)


async def test_api_stream():
    api_base = '192.168.106.12:19000'
    inp = {
        'model': 'Qwen-7B-Chat',
        'messages': [{
            'role': 'user',
            'content': '你是谁'
        }],
        'stream': True,
    }

    async for resp in await ChatCompletion.acreate(api_base, **inp):
        print(resp)


# asyncio.run(test_api())
asyncio.run(test_api_stream())
