import json

import numpy as np
# import torch
import triton_python_backend_utils as pb_utils
from pybackend_libs.dataelem.model import get_model


def _get_np_input(request, name, has_batch=True):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        # model_instance_name = args['model_instance_name']
        model_config = json.loads(args['model_config'])
        self.name = model_config['name']

        self.using_decoupled = (
            pb_utils.using_decoupled_model_transaction_policy(model_config))

        self.logger.log_info(f'using_decoupled {self.using_decoupled}')

        params = model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())
        pymodel_type = parameters.pop('pymodel_type')
        # instance_groups = parameters.pop('instance_groups')
        model_path = parameters.pop('model_path')
        parameters['pretrain_path'] = model_path
        model_cate, model_cls_name = pymodel_type.split('.', 1)

        cls_type = get_model(model_cls_name)
        if cls_type is None:
            raise pb_utils.TritonModelException(
                f'{model_cls_name} is not existed')

        self.model = cls_type(**parameters)
        self.logger.log_info(f'succ to load model [{self.name}]')

    def exec(self, requests):
        def _get_np_input(request, name, has_batch=True):
            return pb_utils.get_input_tensor_by_name(request, name).as_numpy()

        def _get_optional_params(request, name):
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            return json.loads(tensor.as_numpy()[0]) if tensor else {}

        responses = []
        for request in requests:
            status_code = 200
            status_message = 'succ'
            outp = None
            try:
                inp_str = _get_np_input(request, 'INPUT')[0]
                inp = json.loads(inp_str)
                outp = self.model.predict(inp)
            except Exception as e:
                status_code = 400
                status_message = str(e)

            result = {
                'status_code': status_code,
                'status_message': status_message
            }

            if status_code == 200 and outp:
                result.update(outp)

            result_arr = np.array([json.dumps(result)], dtype=np.object_)

            out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def exec_decoupled(self, requests):
        for request in requests:
            # simple invoke
            response_sender = request.get_response_sender()
            try:
                input_data = _get_np_input(request, 'INPUT')[0]
                inp = json.loads(input_data)
                stream = inp.get('stream', False)
                if stream:
                    for out in self.model.stream_predict(inp):
                        out_arr = np.array([json.dumps(out)], dtype=np.object_)
                        out_tensor = pb_utils.Tensor('OUTPUT', out_arr)
                        inference_response = pb_utils.InferenceResponse(
                            output_tensors=[out_tensor])

                        if not response_sender.is_cancelled():
                            response_sender.send(inference_response)
                        else:
                            break
                else:
                    out = self.model.predict(inp)
                    out_arr = np.array([json.dumps(out)], dtype=np.object_)
                    out_tensor = pb_utils.Tensor('OUTPUT', out_arr)
                    inference_response = pb_utils.InferenceResponse(
                            output_tensors=[out_tensor])

                    if not response_sender.is_cancelled():
                        response_sender.send(inference_response)
            except Exception as e:
                self.logger.log_info(f'Error generating stream: {e}')
                error = pb_utils.TritonError(f'Error generating stream: {e}')
                triton_output_tensor = pb_utils.Tensor(
                    'OUTPUT', np.asarray(['N/A'], dtype=np.object_))
                response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error)
                response_sender.send(response)
            finally:
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def execute(self, requests):
        if self.using_decoupled:
            self.exec_decoupled(requests)
            return None
        else:
            return self.exec(requests)

    def finalize(self):
        self.logger.log_info(f'clean up model [{self.name}]')
