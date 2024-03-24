import json
# import os
import subprocess as sp

import numpy as np
import triton_python_backend_utils as pb_utils
from pybackend_libs.dataelem.model import get_model


def _get_np_input(request, name, has_batch=True):
    np_arr = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
    if np_arr.ndim >= 2:
        return np_arr.flatten()
    else:
        return np_arr


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


def get_gpu_memory():
    command = 'nvidia-smi --query-gpu=memory.free --format=csv'
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return memory_free_values


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        model_instance_name = args['model_instance_name']
        model_config = json.loads(args['model_config'])
        self.name = model_config['name']

        self.using_decoupled = (
            pb_utils.using_decoupled_model_transaction_policy(model_config))

        self.logger.log_info(f'using_decoupled {self.using_decoupled}')

        params = model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())
        pymodel_type = parameters.pop('pymodel_type')

        pymodel_params = parameters.pop('pymodel_params', '{}')
        pymodel_params = json.loads(pymodel_params)

        instance_groups = parameters.pop('instance_groups')
        model_path = parameters.pop('model_path')
        parameters['pretrain_path'] = model_path

        group_idx = int(model_instance_name.rsplit('_', 2)[1])
        gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
        parameters['devices'] = gpus[group_idx]

        # Do gpu memory check
        free_gpu_memories = get_gpu_memory()
        device_iarr = [int(d) for d in gpus[group_idx].split(',')]
        gpu_unit = 1024.0
        free_memories = [free_gpu_memories[i] / gpu_unit for i in device_iarr]
        gpu_memory = int(parameters.get('gpu_memory'))
        per_device_alloc_memory = gpu_memory / len(device_iarr)
        for device_id, free_memory in zip(device_iarr, free_memories):
            if free_memory < per_device_alloc_memory:
                raise pb_utils.TritonModelException(
                    f'need to allocate {per_device_alloc_memory}GB '
                    f'on GPU-{device_id}, but only have {free_memory}GB freed')

        if pymodel_params:
            parameters.update(pymodel_params)

        model_cate, model_cls_name = pymodel_type.split('.', 1)

        cls_type = get_model(model_cls_name)
        if cls_type is None:
            raise pb_utils.TritonModelException(
                f'{model_cls_name} is not existed')

        self.model = cls_type(**parameters)
        self.logger.log_info(f'succ to load model [{self.name}]')

    def exec(self, requests):
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

            result_arr = np.array(
                [[json.dumps(result, ensure_ascii=False)]],
                dtype=np.object_)

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
                        out_arr = np.array(
                            [[json.dumps(out)]], dtype=np.object_)
                        out_tensor = pb_utils.Tensor('OUTPUT', out_arr)
                        inference_response = pb_utils.InferenceResponse(
                            output_tensors=[out_tensor])

                        if not response_sender.is_cancelled():
                            response_sender.send(inference_response)
                        else:
                            break
                else:
                    out = self.model.predict(inp)
                    out_arr = np.array(
                        [[json.dumps(out, ensure_ascii=False)]],
                        dtype=np.object_)
                    out_tensor = pb_utils.Tensor('OUTPUT', out_arr)
                    inference_response = pb_utils.InferenceResponse(
                            output_tensors=[out_tensor])

                    if not response_sender.is_cancelled():
                        response_sender.send(inference_response)
            except Exception as e:
                self.logger.log_info(f'Error generating stream: {e}')
                error = pb_utils.TritonError(f'Error generating stream: {e}')
                triton_output_tensor = pb_utils.Tensor(
                    'OUTPUT', np.asarray([['N/A']], dtype=np.object_))
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
