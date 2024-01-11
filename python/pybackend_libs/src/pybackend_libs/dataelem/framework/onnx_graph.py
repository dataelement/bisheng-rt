import os
from typing import Any, List

import onnxruntime


class ONNXGraph:
    def __init__(self, model_path, device):
        if device in ['cpu', 'CPU', '']:
            provider = 'CPUExecutionProvider'
            provider_options = dict()
        else:
            provider = 'CUDAExecutionProvider'
            provider_options = dict(device_id=int(device))

        # sess_opt = onnxruntime.SessionOptions()
        # sess_opt.log_severity_level = 4
        # sess_opt.enable_cpu_mem_arena = False
        # sess_opt.graph_optimization_level = (
        #     onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)

        model_file = os.path.join(model_path, 'model.onnx')
        if not os.path.exists(model_file):
            raise Exception(f'{model_file} not exists')

        self.sess = onnxruntime.InferenceSession(
            model_file,
            # sess_options=sess_opt,
            providers=[provider],
            provider_options=[provider_options])

        self.ys = [x.name for x in self.sess.get_outputs()]
        self.xs = [x.name for x in self.sess.get_inputs()]

    def run(self, inputs: List[Any]) -> List[Any]:
        assert len(inputs) == len(self.xs)
        inputs_dict = dict((x, k) for x, k in zip(self.xs, inputs))
        outputs = self.sess.run(self.ys, inputs_dict)
        return outputs
