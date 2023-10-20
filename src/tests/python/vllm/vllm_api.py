import json
# import time
import uuid

import numpy as np
import tritonclient.grpc.aio as grpcclient

# from tritonclient.utils import InferenceServerException

MAX_TIMEOUT = 20


class ChatCompletion(object):
    OBJECT_NAME = 'chat.completions'

    @classmethod
    def create_request(cls, chat_input, request_id, model_name):
        inputs = []
        prompt_data = np.array([json.dumps(chat_input)], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput('INPUT', [1], 'BYTES'))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as e:
            raise Exception(f'Encountered an error in creating request {e}')

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('OUTPUT'))

        # Issue the asynchronous sequence inference.
        return {
            'model_name': model_name,
            'inputs': inputs,
            'outputs': outputs,
            'request_id': str(request_id),
        }

    @classmethod
    async def _acreate(
        cls,
        api_base=None,
        request_id=None,
        **params,
    ):
        model_name = params.get('model')
        stream = params.get('stream', False)
        verbose = params.pop('verbose', False)
        timeout = params.pop('timeout', MAX_TIMEOUT)
        chat_input = params

        triton_client = grpcclient.InferenceServerClient(url=api_base,
                                                         verbose=verbose)

        # Request iterator that yields the next request
        model_ready = await triton_client.is_model_ready(model_name)
        if not model_ready:
            raise Exception(f'model {model_name} is not exist in server')

        async def async_request_iterator():
            request_id = str(uuid.uuid4().hex)
            yield cls.create_request(chat_input, request_id, model_name)

        # Start streaming
        response_iterator = triton_client.stream_infer(
            inputs_iterator=async_request_iterator(),
            stream_timeout=timeout,
        )

        # Read response from the stream
        if stream:
            return (json.loads(response[0].as_numpy('OUTPUT')[0])
                    async for response in response_iterator)
        else:
            async for response in response_iterator:
                last_output = response
            output = json.loads(last_output[0].as_numpy('OUTPUT')[0])

        await triton_client.close()
        return output

    @classmethod
    async def acreate(cls, *args, **kwargs):
        return await cls._acreate(*args, **kwargs)
