import json
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from pybackend_libs.dataelem.model import get_model


class TritonPythonModel:
    def initialize(self, args):
        model_instance_name = args['model_instance_name']
        model_config = json.loads(args['model_config'])
        self.name = model_config['name']

        params = model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())
        pymodel_type = parameters.pop('pymodel_type')

        pymodel_params = parameters.pop('pymodel_params', '{}')
        pymodel_params = json.loads(pymodel_params)

        instance_groups = parameters.pop('instance_groups')
        model_path = parameters.pop('model_path')
        parameters['model_path'] = model_path

        group_idx = int(model_instance_name.rsplit('_', 1)[1])
        gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
        parameters['devices'] = gpus[group_idx]

        if pymodel_params:
            parameters.update(pymodel_params)

        _, model_cls_name = pymodel_type.split('.', 1)
        cls_type = get_model(model_cls_name)
        if cls_type is None:
            raise pb_utils.TritonModelException(
                f'{model_cls_name} is not existed')

        self.model = cls_type(**parameters)

    def execute(self, requests):
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

    def finalize(self):
        print(f'cleaning up model name={self.name}')
