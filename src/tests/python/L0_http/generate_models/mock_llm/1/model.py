# flake8: noqa

import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


def _get_np_input(request, name, has_batch=True):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.decoupled = self.model_config.get('model_transaction_policy', {}).get(
            'decoupled'
        )

    def execute(self, requests):
        if self.decoupled:
            return self.exec_decoupled(requests)
        else:
            return self.exec(requests)

    def exec(self, requests):
        responses = []
        for request in requests:
            inp_bytes = _get_np_input(request, 'INPUT')[0]
            inp = json.loads(inp_bytes)
            params = inp.get('parameters', {})
            rep_count = params['REPETITION'] if 'REPETITION' in params else 1
            input_np = np.asarray(inp['PROMPT'])
            stream = inp['STREAM']

            if stream:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            'STREAM only supported in decoupled mode'
                        )
                    )
                )
            else:
                resp = {'TEXT': np.repeat(input_np, rep_count, axis=1).tolist()}
                result_arr = np.array([json.dumps(resp)], dtype=np.object_)
                out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0])
                responses.append(inference_response)
        return responses

    def exec_decoupled(self, requests):
        for request in requests:

            inp_bytes = _get_np_input(request, 'INPUT')[0]
            inp = json.loads(inp_bytes)
            params = inp.get('parameters', {})
            rep_count = params['REPETITION'] if 'REPETITION' in params else 1
            fail_last = params['FAIL_LAST'] if 'FAIL_LAST' in params else False
            delay = params['DELAY'] if 'DELAY' in params else None

            input_np = np.asarray(inp['PROMPT'])
            stream = inp['STREAM']

            sender = request.get_response_sender()

            resp = {'TEXT': input_np.tolist()}
            result_arr = np.array([json.dumps(resp)], dtype=np.object_)
            out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])

            # out_tensor = pb_utils.Tensor("TEXT", input_np)
            # response = pb_utils.InferenceResponse([out_tensor])
            # If stream enabled, just send multiple copies of response
            # FIXME: Could split up response string into tokens, but this is simpler for now.

            if stream:
                for _ in range(rep_count):
                    if delay is not None:
                        time.sleep(delay)
                    if not sender.is_cancelled():
                        sender.send(response)
                    else:
                        break
                sender.send(
                    None
                    if not fail_last
                    else pb_utils.InferenceResponse(
                        error=pb_utils.TritonError('An Error Occurred')
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            # If stream disabled, just send one response
            else:
                sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        return None
