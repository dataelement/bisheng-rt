import json

import numpy as np
import triton_python_backend_utils as pb_utils
from pybackend_libs.dataelem.model import get_model
from torch.utils.dlpack import from_dlpack


def pb_tensor_to_numpy(pb_tensor, ret_type):
    if pb_tensor.is_cpu():
        return pb_tensor.as_numpy()
    else:
        if ret_type == 'ndarray':
            pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
            return pytorch_tensor.detach().cpu().numpy()
        else:
            return from_dlpack(pb_tensor.to_dlpack())


class GraphExecutor(object):
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

    # small difference with sess.run
    def run(self, outputs_names, inputs_names, inputs, ret_type='ndarray'):
        input_tensors = []
        for index, input_name in enumerate(inputs_names):
            in_tensor = pb_utils.Tensor(input_name, inputs[index])
            input_tensors.append(in_tensor)

        infer_request = pb_utils.InferenceRequest(
            model_name=self.model_name,
            model_version=self.model_version,
            requested_output_names=outputs_names,
            inputs=input_tensors)

        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise pb_utils.TritonModelException(
                infer_response.error().message())

        graph_outputs = []
        for index, output_name in enumerate(outputs_names):
            pb_tensor = pb_utils.get_output_tensor_by_name(
                infer_response, output_name)
            graph_outputs.append(pb_tensor_to_numpy(pb_tensor, ret_type))

        return graph_outputs


class TritonPythonModel:
    def initialize(self, args):
        # model_instance_name = args['model_instance_name']
        model_config = json.loads(args['model_config'])
        self.name = model_config['name']

        params = model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())
        pymodel_type = parameters.pop('pymodel_type')
        pymodel_params = parameters.pop('pymodel_params', '{}')
        pymodel_params = json.loads(pymodel_params)

        dep_model_name = parameters.pop('dep_model_name')
        dep_model_version = int(parameters.pop('dep_model_version', '-1'))
        self.graph_executor = GraphExecutor(
            dep_model_name, dep_model_version)
        parameters.update(has_graph_executor=True)

        # update devices for gpu enalbed alg model
        model_instance_name = args['model_instance_name']
        instance_groups = parameters.get('instance_groups', '')
        if instance_groups:
            group_idx = int(model_instance_name.rsplit('_', 1)[1])
            gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
            parameters['devices'] = gpus[group_idx]
        else:
            parameters['devices'] = ''

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

        responses = []
        for request in requests:
            try:
                input0 = _get_np_input(request, 'INPUT')
                context = json.loads(input0[0])

                context.update(graph_executor=self.graph_executor)
                result = self.model.predict(context)

                result_arr = np.array([json.dumps(result)], dtype=np.object_)
                out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0])

            except Exception as e:
                error = pb_utils.TritonError(f'Error: {e}')
                triton_output_tensor = pb_utils.Tensor(
                    'OUTPUT', np.asarray(['N/A'], dtype=np.object_))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error)

            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print(f'cleaning up model name={self.name}')
