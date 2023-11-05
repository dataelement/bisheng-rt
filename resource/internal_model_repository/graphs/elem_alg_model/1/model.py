import json

import numpy as np
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
        device_type = instance_groups.split(';')[0]
        device_type = device_type.split('=')[1]
        parameters['device_type'] = device_type
        if device_type == 'CPU':
            parameters['devices'] = '0'
        else:
            gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
            parameters['devices'] = gpus[group_idx]

        self.input_names = [x['name'] for x in model_config['input']]
        self.output_names = [x['name'] for x in model_config['output']]

        if pymodel_params:
            parameters.update(pymodel_params)

        _, model_cls_name = pymodel_type.split('.', 1)
        cls_type = get_model(model_cls_name)
        if cls_type is None:
            raise pb_utils.TritonModelException(
                f'{model_cls_name} is not existed')

        self.model = cls_type(**parameters)
        self.context = parameters

    def execute(self, requests):
        def _get_np_input(request, name, has_batch=True):
            return pb_utils.get_input_tensor_by_name(request, name).as_numpy()

        def _get_optional_params(request, name):
            tensor = pb_utils.get_input_tensor_by_name(request, name)
            return json.loads(tensor.as_numpy()[0]) if tensor else {}

        responses = []
        for request in requests:
            inputs = {}
            for name in self.input_names:
                if name == 'params':
                    inputs[name] = _get_optional_params(request, name)
                else:
                    try:
                        inputs[name] = _get_np_input(request, name)
                    except Exception:
                        inputs[name] = np.array([], dtype=np.bytes_)
            outputs = self.model.predict(self.context, inputs)

            out_tensors = []
            for i in range(len(outputs)):
                out_tensors.append(
                    pb_utils.Tensor(self.output_names[i], outputs[i]))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=out_tensors)
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print(f'cleaning up model name={self.name}')
