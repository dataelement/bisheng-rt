import json
import time
from collections import defaultdict

import numpy as np
import triton_python_backend_utils as pb_utils
from alg import get_algorithm
from app import get_app


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        if 'input' in self.model_config:
            self.inputs = self.model_config['input']
            self.inputs_name = []
            self.inputs_dtype = []
            for input_ in self.inputs:
                self.inputs_name.append(input_['name'])
                self.inputs_dtype.append(
                    pb_utils.triton_string_to_numpy(input_['data_type']))
        else:
            self.inputs_name = []
            self.inputs_dtype = []
        if 'output' in self.model_config:
            self.outputs = self.model_config['output']
            self.outputs_name = []
            self.outputs_dtype = []
            for output_ in self.outputs:
                self.outputs_name.append(output_['name'])
                self.outputs_dtype.append(
                    pb_utils.triton_string_to_numpy(output_['data_type']))
        else:
            self.outputs_name = []
            self.outputs_dtype = []

        parameters = self.model_config['parameters']
        if 'app_type' in parameters and 'algorithm_type' in parameters:
            assert False, \
                'app_type and algorithm_type cannot exist at the same time.'
        elif 'algorithm_type' in parameters:
            self.alg_type = parameters['algorithm_type']['string_value']
            self.model_version = json.loads(args['model_version'])
            self.instance = get_algorithm(self.alg_type,
                                          parameters,
                                          self.inputs_name,
                                          self.outputs_name,
                                          model_version=self.model_version)
        elif 'app_type' in parameters:
            self.app_type = parameters['app_type']['string_value']
            self.instance = get_app(self.app_type, parameters,
                                    self.inputs_name, self.outputs_name)
        else:
            assert False, \
                'app_type and algorithm_type not exist in parameters.'

    async def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.time()
            context = defaultdict()
            input_tensors = request.inputs()
            for input_tensor in input_tensors:
                context[input_tensor.name()] = input_tensor.as_numpy()

            context = await self.instance.predict(context)

            output_tensors = []
            for index, name in enumerate(self.outputs_name):
                assert name in context, \
                    f'{name} not in context. Please check alg or app outputs.'
                output_tensors.append(
                    pb_utils.Tensor(
                        name,
                        np.asarray(context[name]).astype(
                            self.outputs_dtype[index])))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)
            print('total cost: %.4f' % (time.time() - t0))
        return responses

    def finalize(self):
        print('Cleaning up...')
