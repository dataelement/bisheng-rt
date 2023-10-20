import argparse
import asyncio
import json
# import queue
import sys
from os import system

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


def create_request(prompt, stream, request_id, sampling_parameters,
                   model_name):
    input_template = {
        'model': model_name,
        'messages': [],
        'stream': stream,
        **sampling_parameters,
    }
    input0 = input_template.copy()
    input0['messages'].append({'role': 'user', 'content': prompt})

    inputs = []
    prompt_data = np.array([json.dumps(input0)], dtype=np.object_)
    try:
        inputs.append(grpcclient.InferInput('INPUT', [1], 'BYTES'))
        inputs[-1].set_data_from_numpy(prompt_data)
    except Exception as e:
        print(f'Encountered an error {e}')

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


async def main(FLAGS):
    model_name = FLAGS.model_name
    sampling_parameters = {'temperature': '0.1', 'top_p': '0.95'}
    stream = FLAGS.streaming_mode
    with open(FLAGS.input_prompts, 'r') as file:
        print(f'Loading inputs from `{FLAGS.input_prompts}`...')
        prompts = file.readlines()

    results_dict = {}

    async with grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
        # Request iterator that yields the next request
        model_ready = await triton_client.is_model_ready(model_name)
        if not model_ready:
            print(f'model {model_name} is not exist in server')
            return

        async def async_request_iterator():
            try:
                for iter in range(FLAGS.iterations):
                    for i, prompt in enumerate(prompts):
                        prompt_id = FLAGS.offset + (len(prompts) * iter) + i
                        results_dict[str(prompt_id)] = []
                        yield create_request(prompt, stream, prompt_id,
                                             sampling_parameters, model_name)
            except Exception as error:
                print(f'caught error in request iterator:  {error}')

        try:
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=FLAGS.stream_timeout,
            )
            # Read response from the stream
            async for response in response_iterator:
                result, error = response
                if error:
                    print(f'Encountered error while processing: {error}')
                else:
                    output = result.as_numpy('OUTPUT')
                    # print('output', output)
                    results_dict[result.get_response().id].append(output[0])

        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    with open(FLAGS.results_file, 'w') as file:
        for id in results_dict.keys():
            for result in results_dict[id]:
                file.write(result.decode('utf-8'))
                file.write('\n')
            file.write('\n=========\n\n')
        print(f'Storing results into `{FLAGS.results_file}`...')

    if FLAGS.verbose:
        print(f'\nContents of `{FLAGS.results_file}` ===>')
        system(f'cat {FLAGS.results_file}')

    print('PASS: vLLM example')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        required=False,
        default=False,
        help='Enable verbose output',
    )
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:19000',
        help='Server URL and it gRPC port. Default is localhost:9010.',
    )
    parser.add_argument(
        '-m',
        '--model-name',
        type=str,
        required=True,
        default=None,
        help='model name for inference',
    )

    parser.add_argument(
        '-t',
        '--stream-timeout',
        type=float,
        required=False,
        default=None,
        help='Stream timeout in seconds. Default is None.',
    )
    parser.add_argument(
        '--offset',
        type=int,
        required=False,
        default=0,
        help='Add offset to request IDs used',
    )
    parser.add_argument(
        '--input-prompts',
        type=str,
        required=False,
        default='prompts.txt',
        help='Text file with input prompts',
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=False,
        default='results.txt',
        help='The file with output results',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        required=False,
        default=1,
        help='Number of iterations through the prompts file',
    )
    parser.add_argument(
        '-s',
        '--streaming-mode',
        action='store_true',
        required=False,
        default=False,
        help='Enable streaming mode',
    )
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))
