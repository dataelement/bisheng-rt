import json

# import torch
import triton_python_backend_utils as pb_utils


def _get_np_input(request, name, has_batch=True):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.args = args

    def execute(self, requests):
        # TODO zgq: 处理不同的路由
        pass

    def finalize(self):
        self.logger.log_info('finalize model')
