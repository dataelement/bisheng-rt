import json

import triton_python_backend_utils as pb_utils
from utils import Registry

APP_REGISTRY = Registry('app')


def get_app(app_type, *args, **kwargs):
    return APP_REGISTRY.get(app_type)(*args, **kwargs)


@APP_REGISTRY.register_module()
class BaseApp(object):
    def __init__(self, app_params, app_inputs, app_ouputs, **kwargs):
        self.app_inputs = app_inputs
        # app input name not include params
        if 'params' in self.app_inputs:
            self.app_inputs.remove('params')
        # app output name
        self.app_ouputs = app_ouputs

    def alg_infer(self, inputs, alg_name, alg_version, alg_input_names,
                  alg_output_names):
        """alg infer

        Args:
            inputs (list[np.array]): inputs of alg
        Returns:
            outputs (list[np.array]): outputs of alg
        """
        assert len(inputs) == len(
            alg_input_names
        ), 'Num of infer inputs not equal to num of alg inputs in modelconfig.'
        input_tensors = []
        for index, input_ in enumerate(inputs):
            in_tensor = pb_utils.Tensor(alg_input_names[index], input_)
            input_tensors.append(in_tensor)

        infer_request = pb_utils.InferenceRequest(
            model_name=alg_name,
            model_version=alg_version,
            requested_output_names=alg_output_names,
            inputs=input_tensors)

        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise pb_utils.TritonModelException(
                infer_response.error().message())

        outputs = []
        for index, output_ in enumerate(alg_output_names):
            outputs.append(
                pb_utils.get_output_tensor_by_name(infer_response,
                                                   output_).as_numpy())

        return outputs

    async def infer(self, context, inputs):
        """app infer

        Args:
            context (dict): app global information, include params
            inputs (list[np.array]): inputs of app
        Returns:
            context (dict): app global information, include params
            outputs (list[np.array]): outputs of app
        """
        pass

    async def predict(self, context):
        """app predict

        Args:
            context (dict): app global information(include app input tensors)
        Returns:
            context (dict): app global information(include app output tensors)
        """
        params = context.get('params', [b'{}'])
        params = json.loads(params[0].decode('utf-8'))  # python dict
        context['params'] = params

        input_list = []
        for input_name in self.app_inputs:
            assert input_name in context, f'{input_name} not in context. ' + \
                                          'Please check request input tensor.'
            input_list.append(context[input_name])

        context, output_list = await self.infer(context, input_list)

        for index, output_name in enumerate(self.app_ouputs):
            context[output_name] = output_list[index]
        return context
