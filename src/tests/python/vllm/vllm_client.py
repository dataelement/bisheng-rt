import json
import queue
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


def create_request(model_name, prompt):

    input0 = {
        'model': model_name,
        'messages': [{
            'role': 'user',
            'content': prompt
        }],
        'stream': True
    }
    # input0_str = json.dumps(input0, ensure_ascii=False).encode("utf-8")

    inputs = []
    input0_data = np.array([json.dumps(input0)], dtype=np.object_)
    inputs.append(grpcclient.InferInput('INPUT', [1], 'BYTES'))
    inputs[-1].set_data_from_numpy(input0_data)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT'))

    return inputs, outputs


def main():
    user_data = UserData()
    model_name = 'Qwen-7B-Chat'
    stream = True
    with grpcclient.InferenceServerClient(url='localhost:9010',
                                          verbose=False) as triton_client:
        # Establish stream
        triton_client.start_stream(callback=partial(callback, user_data))
        inputs, outputs = create_request(model_name, prompts[0])

        request_id = '10000'
        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
        )

        # Retrieve results...
        recv_count = 0
        result_dict = {}
        texts = []
        while True:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                break
            else:
                this_id = data_item.get_response().id
                if this_id not in result_dict.keys():
                    result_dict[this_id] = []

                output = data_item.as_numpy('OUTPUT')
                resp = json.loads(output[0])
                if stream:
                    texts.append(resp['choices'][0]['delta']['content'])
                else:
                    texts.append(resp['choices'][0]['message']['content'])
                print('id,resp', this_id, resp)
                if resp['choices'][0]['finish_reason'] is not None:
                    break

                result_dict[this_id].append((recv_count, data_item))

            recv_count += 1

        print('Query: {}\nAnswer:{}'.format(prompts[0], ''.join(texts)))


main()
