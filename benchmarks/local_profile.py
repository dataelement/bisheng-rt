# Please pull the latest code to run the profiling.
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed


class QwenProfile:
    def __init__(self, model_path, device='cuda:0', seed=1024,
                 quant_type='bf16', **kwargs) -> None:
        self.seed = seed
        self.quant_type = quant_type
        self.use_flash_attn = kwargs.get('use_flash_attn', False)
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

        if quant_type == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                bf16=True,
                use_flash_attn=self.use_flash_attn
            ).eval()
        elif quant_type == 'fp32':
            assert self.use_flash_attn is False, 'FP32 profiling cannot be \
                performed with Flash-Attention activated.'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                fp32=True,
                use_flash_attn=False
            ).eval()
        elif quant_type == 'int4':
            # please install AutoGPTQ following the readme to use quantization
            from auto_gptq import AutoGPTQForCausalLM
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_path,
                device=device,
                trust_remote_code=True,
                use_safetensors=True,
                use_flash_attn=self.use_flash_attn
            ).eval()

        # Specify hyperparameters for generation
        self.config = GenerationConfig.from_pretrained(model_path,
                                                       trust_remote_code=True)

    def profile(self,
                max_experiment_times=1,
                context_length_per_experiment=1,
                generate_length_per_experiment=2048):
        self.config.min_length = (generate_length_per_experiment +
                                  context_length_per_experiment)
        self.config.max_new_tokens = generate_length_per_experiment

        time_costs = []
        context_str = '我' * context_length_per_experiment
        max_gpu_memory_cost = 0
        for _ in tqdm(range(max_experiment_times)):
            inputs = self.tokenizer(context_str, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            t1 = time.time()
            pred = self.model.generate(**inputs, generation_config=self.config)
            time_costs.append(time.time() - t1)
            assert pred.shape[1] == self.config.min_length
            max_gpu_memory_cost = max(max_gpu_memory_cost,
                                      torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()

        print('Average generate speed (tokens/s): {}'.format(
            (max_experiment_times * generate_length_per_experiment) /
            sum(time_costs)))
        print(f'GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB')
        print('Experiment setting: ')
        print(f'seed = {self.seed}')
        print(f'max_experiment_times = {max_experiment_times}')
        print(f'context_length_per_experiment = \
              {context_length_per_experiment}')
        print(f'generate_length_per_experiment = \
              {generate_length_per_experiment}')
        print(f'use_flash_attn = {self.use_flash_attn}')
        print(f'quant_type = {self.quant_type}')


class ChatGlmProfile:
    def __init__(self, model_path, device='cuda:0',
                 seed=1024, quant_type='bf16', **kwargs) -> None:
        self.seed = seed
        self.quant_type = quant_type
        self.use_flash_attn = kwargs.get('use_flash_attn', False)
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

        if quant_type == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
            ).bfloat16().eval()
        elif quant_type == 'fp32':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
            ).eval()

    def profile(self,
                max_experiment_times=1,
                context_length_per_experiment=1,
                generate_length_per_experiment=2048):
        min_length = (generate_length_per_experiment +
                      context_length_per_experiment)
        max_new_tokens = generate_length_per_experiment

        time_costs = []
        context_str = '我' * context_length_per_experiment
        max_gpu_memory_cost = 0
        for _ in tqdm(range(max_experiment_times)):
            inputs = self.tokenizer(context_str, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            t1 = time.time()
            pred = self.model.generate(**inputs,
                                       do_sample=True,
                                       min_length=min_length,
                                       max_new_tokens=max_new_tokens,
                                       )
            time_costs.append(time.time() - t1)
            print(pred)
            print(pred.shape)
            assert pred.shape[1] == min_length + 2
            max_gpu_memory_cost = max(max_gpu_memory_cost,
                                      torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()

        print('Average generate speed (tokens/s): {}'.format(
            (max_experiment_times * generate_length_per_experiment) /
            sum(time_costs)))
        print(f'GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB')
        print('Experiment setting: ')
        print(f'seed = {self.seed}')
        print(f'max_experiment_times = {max_experiment_times}')
        print(f'context_length_per_experiment = \
              {context_length_per_experiment}')
        print(f'generate_length_per_experiment = \
              {generate_length_per_experiment}')
        print(f'use_flash_attn = {self.use_flash_attn}')
        print(f'quant_type = {self.quant_type}')


if __name__ == '__main__':
    device = 'cuda:0'
    quant_type = 'bf16'  # fp32, bf16 or int4
    # model_path = "/home/public/llm/Qwen-7B-Chat"
    # model_profile = QwenProfile(model_path,
    #                             device=device, quant_type=quant_type)

    model_path = '/home/public/llm/chatglm2-6b'
    model_profile = ChatGlmProfile(model_path,
                                   device=device, quant_type=quant_type)

    model_profile.profile(max_experiment_times=1,
                          context_length_per_experiment=1,
                          generate_length_per_experiment=2048)
