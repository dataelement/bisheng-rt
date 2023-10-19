import json
import queue
import threading
# import uuid
from collections import defaultdict
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


prompts = [
    '以“今晚夜色真美”为开头写一篇短文，包含悬疑元素',
]


def create_request(model_name, prompt, stream=True):

    input0 = {
        'model': model_name,
        'messages': [{
            'role': 'user',
            'content': prompt
        }],
        'stream': stream
    }
    # input0_str = json.dumps(input0, ensure_ascii=False).encode("utf-8")

    inputs = []
    input0_data = np.array([json.dumps(input0)], dtype=np.object_)
    inputs.append(grpcclient.InferInput('INPUT', [1], 'BYTES'))
    inputs[-1].set_data_from_numpy(input0_data)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT'))

    return inputs, outputs


def task():
    user_data = UserData()
    model_name = 'Qwen-7B-Chat'
    # model_name = 'Qwen-14B-Chat'

    stream = True

    prompts = open('prompts.txt').readlines()
    req_cnt = len(prompts)
    request_id_offset = 10000
    with grpcclient.InferenceServerClient(url='localhost:9010',
                                          verbose=False) as triton_client:
        # Judge model is exists, framework will occur core for unknown model.
        model_ready = triton_client.is_model_ready(model_name)
        if not model_ready:
            print(f'model {model_name} is not exist in server')
            return

        # Establish stream
        triton_client.start_stream(callback=partial(callback, user_data))

        for i, prompt in enumerate(prompts):
            inputs, outputs = create_request(model_name, prompts[i], stream)
            request_id = str(request_id_offset + i)
            triton_client.async_stream_infer(
                model_name=model_name,
                inputs=inputs,
                request_id=request_id,
                outputs=outputs,
            )

        # Retrieve results...
        recv_count = 0
        result_dict = defaultdict(list)
        recv_req_cnt = 0
        while True:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                break
            else:
                this_id = data_item.get_response().id
                output = data_item.as_numpy('OUTPUT')
                resp = json.loads(output[0])
                if stream:
                    content = resp['choices'][0]['delta']['content']
                else:
                    content = resp['choices'][0]['message']['content']

                if resp['choices'][0]['finish_reason'] is not None:
                    recv_req_cnt += 1

                result_dict[this_id].append(content)

            recv_count += 1
            if recv_req_cnt == req_cnt:
                break

        for key in result_dict.keys():
            i = int(key) - request_id_offset
            a = ''.join(result_dict[key])
            print('Query: {}\nAnswer:{}'.format(prompts[i], a))


def main():
    tasks = []
    for i in range(1):
        t = threading.Thread(target=task)
        t.start()
        tasks.append(t)
    [t.join() for t in tasks]


main()
