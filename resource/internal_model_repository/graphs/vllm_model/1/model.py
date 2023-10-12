# flake8: noqa: E501
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modify by DataElem, Inc. 2023, adapted for bisheng-rt

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union
import argparse
import numpy as np
import triton_python_backend_utils as pb_utils
from pybackend_libs.dataelem.model.vllm.conversation import get_gen_prompt
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
from dataclasses import dataclass

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    do_sample: Optional[bool] = False
    sampling_parameters: Optional[Dict[Any, Any]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


def _get_np_input(request, name, has_batch=True):
    return pb_utils.get_input_tensor_by_name(request, name).as_numpy()


def _get_optional_params(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return json.loads(tensor.as_numpy()[0]) if tensor else {}


class Messages2Prompt(object):
    def __init__(self, model_type):
        self.model_type = model_type

    def run(self, messages):
        messages = [m.dict() for m in messages]
        return get_gen_prompt(self.model_type, messages)

def parameters2parser(parameters):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=127.0.0.1)
    # parser.add_argument("--port", type=int, default=8000)
 
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = vllm_engine_config["model"]
    args.tokenizer = vllm_engine_config["tokenizer"]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    return engine_args



class TritonPythonModel:
    def initialize(self, args):
        print("==get into initialize")
        #self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        self.model_name = args['model_name']
        model_instance_name = args['model_instance_name']

        # assert are in decoupled mode. Currently, Triton needs to use
        # decoupled policy for asynchronously forwarding requests to
        # vLLM engine.
        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config)
        assert (
            self.using_decoupled
        ), 'vLLM Triton backend must be configured to use decoupled model transaction policy'

        # engine_args_filepath = os.path.join(
        #     args["model_repository"], _VLLM_ENGINE_ARGS_FILENAME
        # )
        # assert os.path.isfile(
        #     engine_args_filepath
        # ), (f"'{_VLLM_ENGINE_ARGS_FILENAME}' containing vllm engine args must be"
        #     f"provided in '{args['model_repository']}'")
        # with open(engine_args_filepath) as file:
        #     vllm_engine_config = json.load(file)

        params = self.model_config['parameters']
        parameters = dict((k, v['string_value']) for k, v in params.items())
        pymodel_type = parameters.pop('pymodel_type')

        pymodel_params = parameters.pop('pymodel_params', '{}')
        pymodel_params = json.loads(pymodel_params)

        instance_groups = parameters.pop('instance_groups')
        model_path = parameters.pop('model_path')
        group_idx = int(model_instance_name.rsplit('_', 1)[1])
        gpus = instance_groups.split(';', 1)[1].split('=')[1].split('|')
        devices = gpus[group_idx]

        # important, mark which devices to be used
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'

        disable_log_requests = parameters.pop('disable_log_requests', 'true')
        max_num_seqs = parameters.pop('max_num_seqs', 256)
        max_num_batched_tokens = parameters.pop('max_num_batched_tokens', None)
        gpu_memory_utilization = parameters.pop('gpu_memory_utilization', 0.5)
        dtype = parameters.pop('dtype', 'auto')
        # tp model is more fast, but gpu memory will be equally allcoated.
        # pp model can using huggingface+accelate
        tensor_parallel_size = len(devices.split(','))

        vllm_engine_config = {
            'model': os.path.join("/opt/bisheng-rt",model_path),
            'tokenizer': os.path.join("/opt/bisheng-rt",model_path),
            'tokenizer_mode': 'auto', 
            'trust_remote_code': False,
            'download_dir': None, 
            'load_format': 'auto',
            'dtype': 'auto',
            'seed': 0, 
            'max_model_len': None, 
            'worker_use_ray': False, 
            'pipeline_parallel_size': 1, 
            'tensor_parallel_size': 1, 
            'block_size': 16, 
            'swap_space': 4, 
            'gpu_memory_utilization': 0.9,
            'max_num_batched_tokens': None, 
            'max_num_seqs': 256,
            'disable_log_stats': False, 
            'revision': None, 
            'quantization': None, 
            'engine_use_ray': False, 
            'disable_log_requests': False, 
            'max_log_len': None

            # 'disable_log_requests': disable_log_requests,
            # # 'gpu_memory_utilization': gpu_memory_utilization,
            # # 'max_num_seqs': max_num_seqs,
            # 'max_num_batched_tokens': max_num_batched_tokens,
            # 'tensor_parallel_size': tensor_parallel_size,
            
            
        }
        engine_args = parameters2parser(vllm_engine_config)
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        print("vllm_engine_config:",vllm_engine_config)
        self.messages_to_prompt = Messages2Prompt(self.model_name)

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
        self._loop.run_until_complete(self.await_shutdown())

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
            #self.logger.log_info('Awaiting remaining {} requests'.format(
            #    self.ongoing_request_count))
            print("Awaiting remaining {} requests".format(self.ongoing_request_count))
            await asyncio.sleep(5)

        #self.logger.log_info('Shutdown complete')
        print("Shutdown complete")

    def get_sampling_params_dict(self, params_dict):
        """
        This functions parses the dictionary values into their
        expected format.
        """

        # params_dict = json.loads(params_json)

        # Special parsing for the supported sampling parameters
        bool_keys = ['ignore_eos', 'skip_special_tokens', 'use_beam_search']
        for k in bool_keys:
            if k in params_dict:
                params_dict[k] = bool(params_dict[k])

        float_keys = [
            'frequency_penalty', 'length_penalty', 'presence_penalty',
            'temperature', 'top_p'
        ]
        for k in float_keys:
            if k in params_dict:
                params_dict[k] = float(params_dict[k])

        int_keys = ['best_of', 'max_tokens', 'n', 'top_k']
        for k in int_keys:
            if k in params_dict:
                params_dict[k] = int(params_dict[k])

        return params_dict

    def create_response(self, vllm_output):
        """
        Parses the output from the vLLM engine into Triton
        response.
        """

        choices = []
        for output in vllm_output.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(vllm_output.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in vllm_output.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        resp = ChatCompletionResponse(model=self.model_name,
                                      choices=choices,
                                      object='chat.completion',
                                      usage=usage)

        result_arr = np.array([resp.json()], dtype=np.object_)

        out_tensor_0 = pb_utils.Tensor('OUTPUT', result_arr)
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0])

        return inference_response

    async def generate(self, request):
        """
        Forwards single request to LLM engine and returns responses.
        """
        print("==get into generate")
        response_sender = request.get_response_sender()
        self.ongoing_request_count += 1
        try:
            request_id = random_uuid()

            inp_str = _get_np_input(request, 'INPUT')[0]
            inp = json.loads(inp_str)
            request = ChatCompletionRequest.parse_obj(inp)

            prompt = self.messages_to_prompt(request.messages)
            print("prompt:",prompt)
            stream = request.stream

            params_dict = request.sampling_parameters
            sampling_params_dict = self.get_sampling_params_dict(params_dict)
            print("sampling_params_dict:",sampling_params_dict)
            sampling_params = SamplingParams(**sampling_params_dict)
            print("sampling_params:",sampling_params)

            last_output = None
            async for output in self.llm_engine.generate(
                    prompt, sampling_params, request_id):
                if stream:
                    response_sender.send(self.create_response(output))
                else:
                    last_output = output

            if not stream:
                response_sender.send(self.create_response(last_output))

        except Exception as e:
            #self.logger.log_info(f'Error generating stream: {e}')
            print(f'Error generating stream: {e}')
            error = pb_utils.TritonError(f'Error generating stream: {e}')
            triton_output_tensor = pb_utils.Tensor(
                'OUTPUT', np.asarray(['N/A'], dtype=np.object_))
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
        print("==get into execute")
        for request in requests:
            self.create_task(self.generate(request))
        return None

    def finalize(self):
        """
        Triton virtual method; called when the model is unloaded.
        """
        #self.logger.log_info('Issuing finalize to vllm backend')
        print("Issuing finalize to vllm backend")
        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
