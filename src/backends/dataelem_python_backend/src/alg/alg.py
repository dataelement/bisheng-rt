import base64
import json

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
from utils import Registry

ALGORITHM_REGISTRY = Registry('algorithm')


def get_algorithm(algorithm_type, *args, **kwargs):
    return ALGORITHM_REGISTRY.get(algorithm_type)(*args, **kwargs)


def pb_tensor_to_numpy(pb_tensor):
    if pb_tensor.is_cpu():
        return pb_tensor.as_numpy()
    else:
        pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
        return pytorch_tensor.detach().cpu().numpy()


@ALGORITHM_REGISTRY.register_module()
class AlgorithmNoGraph(object):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        self.alg_inputs = alg_inputs
        # alg input name not include params
        if 'params' in self.alg_inputs:
            self.alg_inputs.remove('params')
        # alg output name
        self.alg_ouputs = alg_ouputs

    async def infer(self, context, inputs):
        """dep model infer

        Args:
            context (dict): algorithm global information, include params
            inputs (list[np.array]): inputs of algorithm
        Returns:
            context (dict): algorithm global information, include params
            outputs (list[np.array]): outputs of algorithm
        """
        pass

    async def predict(self, context):
        """algorithm predict

        Args:
            context (dict): algorithm global information
                            (include alg input tensors)
        Returns:
            context (dict): algorithm global information
                            (include alg output tensors)
        """
        params = context.get('params', [b'{}'])
        params = json.loads(params[0].decode('utf-8'))  # python dict
        context['params'] = params

        input_list = []
        for input_name in self.alg_inputs:
            assert input_name in context, f'{input_name} not in context. ' + \
                                          'Please check request input tensor.'
            input_list.append(context[input_name])

        context, output_list = await self.infer(context, input_list)

        for index, output_name in enumerate(self.alg_ouputs):
            context[output_name] = output_list[index]
        return context


@ALGORITHM_REGISTRY.register_module()
class AlgorithmWithGraph(object):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        self.alg_inputs = alg_inputs  # alg input name
        # alg input name not include params
        if 'params' in self.alg_inputs:
            self.alg_inputs.remove('params')
        self.alg_ouputs = alg_ouputs  # alg output name

        assert 'dep_model_name' in alg_params, 'Please specify ' + \
                                               'dep_model_name in config.pbtxt'
        self.dep_model_name = alg_params['dep_model_name']['string_value']
        self.dep_model_version = kwargs.get('model_version', -1)
        self.dep_model_inputs = []  # dep_model input name
        self.dep_model_outputs = []  # dep_model output name

    def preprocess(self, context, inputs):
        """algorithm preprocess

        Args:
            context (dict): algorithm global information
            inputs (list[np.array]): inputs of preprocess (inputs of algorithm)
        Returns:
            context (dict): algorithm global information
            pre_outputs (list[np.array]): outputs of preprocess
                                          (inputs of infer)
        """
        pass

    def graph_infer(self, inputs):
        """dep model infer

        Args:
            inputs (list[np.array]): inputs of dep_model
        Returns:
            graph_outputs (list[np.array]): outputs of dep_model
        """
        assert len(inputs) == len(
            self.dep_model_inputs
        ), 'The num of infer inputs is not equal to num of dep model inputs.'
        input_tensors = []
        for index, input_ in enumerate(inputs):
            in_tensor = pb_utils.Tensor(self.dep_model_inputs[index], input_)
            input_tensors.append(in_tensor)

        infer_request = pb_utils.InferenceRequest(
            model_name=self.dep_model_name,
            model_version=self.dep_model_version,
            requested_output_names=self.dep_model_outputs,
            inputs=input_tensors)

        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise pb_utils.TritonModelException(
                infer_response.error().message())

        graph_outputs = []
        for index, output_ in enumerate(self.dep_model_outputs):
            pb_tensor = pb_utils.get_output_tensor_by_name(
                infer_response, output_)
            graph_outputs.append(pb_tensor_to_numpy(pb_tensor))

        return graph_outputs

    async def infer(self, context, inputs):
        """algorithm infer

        Args:
            context (dict): algorithm global information
            inputs (list[np.array]): inputs of infer (outputs of preprocess)
        Returns:
            context (dict): algorithm global information
            infer_outputs (list[np.array]): outputs of infer
                                            (inputs of postprocess)
        """
        # prepare for graph inputs
        graph_inputs = inputs
        # graph infer
        graph_outputs = self.graph_infer(graph_inputs)
        # prepare for postprocess inputs
        infer_outputs = graph_outputs
        return context, infer_outputs

    def postprocess(self, context, inputs):
        """algorithm postprocess

        Args:
            context (dict): algorithm global information
            inputs (list[np.array]): inputs of postprocess (outputs of infer)
        Returns:
            context (dict): algorithm global information
            post_outputs (list[np.array]): outputs of postprocess
                                           (outputs of algorithm)
        """
        pass

    async def predict(self, context):
        """algorithm predict

        Args:
            context (dict): algorithm global information
                            (include alg input tensors)
        Returns:
            context (dict): algorithm global information
                            (include alg output tensors)
        """
        params = context.get('params', [b'{}'])
        params = json.loads(params[0].decode('utf-8'))  # python dict
        context['params'] = params

        input_list = []
        for input_name in self.alg_inputs:
            assert input_name in context, f'{input_name} not in context. ' + \
                                          'Please check request input tensor.'
            input_list.append(context[input_name])

        context, pre_outputs = self.preprocess(context, input_list)
        context, infer_outputs = await self.infer(context, pre_outputs)
        context, post_outputs = self.postprocess(context, infer_outputs)

        assert len(post_outputs) == len(
            self.alg_ouputs
        ), 'Num of post_outputs not equal to num of outputs in modelconfig.'
        for index, output_name in enumerate(self.alg_ouputs):
            context[output_name] = post_outputs[index]
        return context


@ALGORITHM_REGISTRY.register_module()
class ImageDecode(AlgorithmNoGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(ImageDecode, self).__init__(alg_params, alg_inputs, alg_ouputs,
                                          **kwargs)

    async def infer(self, context, inputs):
        """ImageDecode infer

        Args:
            context (dict): ImageDecode global information
            inputs (list[np.array]): [image_b64], inputs of algorithm
        Returns:
            context (dict): ImageDecode global information
            pre_outputs (list[np.array]): [image], outputs of preprocess
        """

        # base64 decode and image decode
        image_b64 = inputs[0][0]
        image = cv2.imdecode(
            np.fromstring(base64.b64decode(image_b64), np.uint8),
            cv2.IMREAD_COLOR)
        outputs = [image.astype(np.float32)]
        return context, outputs
