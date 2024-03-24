# flake8: noqa: E501
import asyncio
import json
import os
import subprocess as sp
import threading
import time
import uuid

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


def get_gpu_memory_v2():
    command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv'
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]

    gpu_unit = 1024.0
    memory_frees = []
    memory_totals = []
    for info in memory_free_info:
        free_info, total_info = info.split(',')
        memory_frees.append(int(free_info.split()[0]) / gpu_unit)
        memory_totals.append(int(total_info.split()[0]) / gpu_unit)

    return memory_frees, memory_totals


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        model_instance_name = args['model_instance_name']
        self.model_config = json.loads(args['model_config'])
        self.model_name = args['model_name']

        # assert are in decoupled mode. Currently, Triton needs to use
        # decoupled policy for asynchronously forwarding requests to
        # vLLM engine.
        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)

        params = self.model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())

        stream = parameters.pop('use_decoupled', '1') == 1
        if stream:
            assert self.using_decoupled, (
                'vLLM Triton backend must be configured '
                'to use decoupled model transaction policy')
        self.logger.log_info('using_decoupled {}'.format(self.using_decoupled))

        pymodel_type = parameters.pop('pymodel_type')
        model_path = parameters.pop('model_path')
        parameters['pretrain_path'] = model_path

        instance_groups = parameters.pop('instance_groups')
        group_idx = int(model_instance_name.rsplit('_', 2)[1])
        gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
        devices = gpus[group_idx]
        parameters.update(devices=devices)

        # Do gpu memory check, vllm is very fragile, core will occur if out of cuda memory.
        # avoid breakdown all rt service.
        free_gpu_memories, total_gpu_memories = get_gpu_memory_v2()
        device_iarr = [int(d) for d in devices.split(',')]

        free_memories = [free_gpu_memories[i] for i in device_iarr]
        gpu_memory = int(parameters.get('gpu_memory'))
        per_device_alloc_memory = gpu_memory / len(device_iarr)
        for device_id, free_memory in zip(device_iarr, free_memories):
            if free_memory < per_device_alloc_memory:
                raise pb_utils.TritonModelException(
                    f'need to allocate {per_device_alloc_memory}GB '
                    f'on GPU-{device_id}, but only have {free_memory}GB freed')

        if total_gpu_memories:
            assert len(
                set(total_gpu_memories)) == 1, ('need same gpu device on host')

        truncate_func = lambda x: float(int(x * (1000)) / (1000))
        util_ratio = per_device_alloc_memory / total_gpu_memories[0]
        util_ratio = truncate_func(util_ratio)
        if 'pymodel_params' in parameters:
            pymodel_params_info = json.loads(parameters['pymodel_params'])
            pymodel_params_info['gpu_memory_utilization'] = util_ratio
            parameters['pymodel_params'] = json.dumps(pymodel_params_info)
        else:
            pymodel_params_info = {'gpu_memory_utilization': util_ratio}
            parameters['pymodel_params'] = json.dumps(pymodel_params_info)

        model_cate, model_cls_name = pymodel_type.split('.', 1)
        parameters.update(model_type=model_cls_name)
        vllm_model_cls_name = 'VLLMModel'
        cls_type = get_model(vllm_model_cls_name)
        if cls_type is None:
            raise pb_utils.TritonModelException(
                f'{model_cls_name} is not existed')
        self.model = cls_type(**parameters)

        # output_config = pb_utils.get_output_config_by_name(self.model_config, "TEXT")
        # self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Counter to keep track of ongoing request counts
        self.ongoing_request_count = 0

        # Starting asyncio event loop to process the received requests asynchronously.
        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.Thread(target=self.engine_loop,
                                             args=(self._loop, ))
        self._shutdown_event = asyncio.Event()

        self._loop_thread.start()

    def create_task(self, coro):
        """
        Creates a task on the engine's event loop which is running on a separate thread.
        """
        assert (self._shutdown_event.is_set() is
                False), 'Cannot create tasks after shutdown has been requested'

        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def engine_loop(self, loop):
        """
        Runs the engine's event loop on a separate thread.
        """
        asyncio.set_event_loop(loop)
        try:
            self._loop.run_until_complete(self.await_shutdown())
        except Exception:
            pass

    async def await_shutdown(self):
        """
        Primary coroutine running on the engine event loop. This coroutine is responsible for
        keeping the engine alive until a shutdown is requested.
        """
        # first await the shutdown signal
        while self._shutdown_event.is_set() is False:
            await asyncio.sleep(5)

        # Wait for the ongoing_requests
        while self.ongoing_request_count > 0:
            self.logger.log_info('Awaiting remaining {} requests'.format(
                self.ongoing_request_count))
            await asyncio.sleep(5)

        for task in asyncio.all_tasks(loop=self._loop):
            if task is not asyncio.current_task():
                task.cancel()

        self.logger.log_info('Shutdown complete')

    def create_response(self, vllm_output, previous_texts=None, stream=False):
        """
        Parses the output from the vLLM engine into Triton
        response.
        """
        resp = self.model.make_response(vllm_output, previous_texts, stream,
                                        self.model_name)
        # resp_str = json.dumps(resp, ensure_ascii=False).encode('utf-8')

        resp_str = json.dumps(resp, ensure_ascii=False)
        result_arr = np.array([[json.dumps(resp, ensure_ascii=False)]],
            dtype=np.object_)
        out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0])

        return inference_response

    async def nondecoupled_generate(self, request):
        """
        Forwards single request to LLM engine and returns responses.
        """
        response = None
        self.ongoing_request_count += 1
        try:
            request_id = str(uuid.uuid4().hex)
            inp_bytes = _get_np_input(request, 'INPUT')[0]
            inp = json.loads(inp_bytes)
            previous_texts = [''] * self.model.get_n()
            async for vllm_output in await self.model.generate(
                    inp, request_id):
                last_output = vllm_output

            response = self.create_response(last_output, previous_texts, False)
        except Exception as e:
            self.logger.log_info(f'Error generating stream: {e}')
            error = pb_utils.TritonError(f'Error generating stream: {e}')
            triton_output_tensor = pb_utils.Tensor(
                'OUTPUT', np.asarray([['N/A']], dtype=np.object_))
            response = pb_utils.InferenceResponse(
                output_tensors=[triton_output_tensor], error=error)
        finally:
            self.ongoing_request_count -= 1

        return response

    async def generate(self, request):
        """
        Forwards single request to LLM engine and returns responses.
        """
        response_sender = request.get_response_sender()
        self.ongoing_request_count += 1
        try:
            request_id = str(uuid.uuid4().hex)
            inp_bytes = _get_np_input(request, 'INPUT')[0]
            inp = json.loads(inp_bytes)
            # inp = json.loads(inp_bytes.decode(encoding='utf-8'))
            # print('inp', [inp_bytes], [inp])
            stream = inp.get('stream', False)
            last_output = None
            previous_texts = [''] * self.model.get_n()
            async for vllm_output in await self.model.generate(
                    inp, request_id):
                if stream:
                    response_sender.send(
                        self.create_response(vllm_output, previous_texts,
                                             stream))
                    for output in vllm_output.outputs:
                        previous_texts[output.index] = output.text
                else:
                    last_output = vllm_output

            if not stream:
                response_sender.send(
                    self.create_response(last_output, previous_texts, stream))

        except Exception as e:
            self.logger.log_info(f'Error generating stream: {e}')
            error = pb_utils.TritonError(f'Error generating stream: {e}')
            triton_output_tensor = pb_utils.Tensor(
                'OUTPUT', np.asarray([['N/A']], dtype=np.object_))
            response = pb_utils.InferenceResponse(
                output_tensors=[triton_output_tensor], error=error)
            response_sender.send(response)
            raise e
        finally:
            response_sender.send(
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            self.ongoing_request_count -= 1

    def execute(self, requests):
        """
        Triton core issues requests to the backend via this method.

        When this method returns, new requests can be issued to the backend. Blocking
        this function would prevent the backend from pulling additional requests from
        Triton into the vLLM engine. This can be done if the kv cache within vLLM engine
        is too loaded.
        We are pushing all the requests on vllm and let it handle the full traffic.
        """
        if self.using_decoupled:
            for request in requests:
                self.create_task(self.generate(request))
            return None
        else:
            futures = []
            for request in requests:
                future = self.create_task(self.nondecoupled_generate(request))
                futures.append(future)
            return [future.result() for future in futures]

    def finalize(self):
        """
        Triton virtual method; called when the model is unloaded.
        """
        self.logger.log_info('Issuing finalize to vllm backend')
        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
