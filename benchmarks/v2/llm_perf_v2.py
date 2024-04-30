# flake8: noqa
"""Benchmark online serving throughput for llm with simple simulous call.
"""
import argparse
import asyncio
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    stream: bool = True
    proxy: str = ''
    system_prompt: str = ''


@dataclass
class RequestFuncOutput:
    generated_text: str = ''
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith('v1/chat/completions')

    stream = request_func_input.stream
    system_prompt = request_func_input.system_prompt
    proxy_params = {}
    if request_func_input.proxy:
        proxy_params = {'proxy': request_func_input.proxy}

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search

        payload = {
          'model': request_func_input.model,
          'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': request_func_input.prompt}
           ],
          'temperature': 0.0,
          'max_tokens': request_func_input.output_len,
          'stream': stream
        }

        headers = {
            'Authorization': f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        if stream:
            headers.update({'Accept': 'text/event-stream'})

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        generated_text = ''
        ttft = 0
        st = time.perf_counter()
        latency = None
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers,
                                    **proxy_params) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        if stream:
                            chunk = chunk.decode('utf-8').lstrip('data: ')
                            if chunk == '[DONE]':
                                latency = time.perf_counter() - st
                            else:
                                body = json.loads(chunk)
                                if 'content' in body['choices'][0]['delta']:
                                    generated_text += body['choices'][0]['delta']['content']
                        else:
                            body = json.loads(chunk)
                            generated_text += body['choices'][0]['message']['content']
                            latency = time.perf_counter() - st

                    if latency is None:
                        latency = time.perf_counter() - st

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

    if pbar:
        pbar.update(1)
    return output


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float

def load_dataset(filename, sample_type='conv'):
    with open(filename) as f:
        dataset = json.load(f)

    if sample_type == 'conv':
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data['conversations'][0]['value'],
                    data['conversations'][1]['value']) for data in dataset]
    elif sample_type == 'instruct':
        dataset = [(data['instruction'] + ' ' + data['input'], data['output'])
                   for data in dataset if 'output' in data and 'input' in data]

    return dataset

def sample_requests(
    dataset_path: str,
    dataset_type: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset = load_dataset(dataset_path, dataset_type)
    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer.encode(outputs[i].generated_text))
            total_output += output_len
            total_input += input_requests[i][1]
            per_token_latencies.append(outputs[i].latency / output_len)
            ttfts.append(outputs[i].ttft)
            completed += 1

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
    )

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    stream: bool,
    proxy: str,
    system_prompt: str,
    disable_tqdm: bool,
    task_id,
    result_queue
):
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    print(f'Stream mode: {stream}')

    benchmark_start_time = time.perf_counter()

    outputs = []
    for request in input_requests:
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            stream=stream,
            proxy=proxy,
            system_prompt=system_prompt,
        )
        output = await async_request_openai_chat_completions(request_func_input, pbar=pbar)
        outputs.append(output)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    result_queue.put((task_id, outputs, benchmark_duration))


def merge_metrics(input_requests, outputs, benchmark_durations, tokenizer):
    benchmark_duration = max(benchmark_durations)
    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print(f'Successful requests: {metrics.completed}')
    print(f'Benchmark duration: {benchmark_duration:2f} s')
    print(f'Total input tokens: {metrics.total_input}')
    print(f'Total generated tokens: {metrics.total_output}')
    print(f'Request throughput: {metrics.request_throughput:.2f} requests/s')
    print(f'Input token throughput: {metrics.input_throughput:.2f} tokens/s')
    print(f'Output token throughput: {metrics.output_throughput:.2f} tokens/s')
    print(f'Mean TTFT: {metrics.mean_ttft_ms:.2f} ms')
    print(f'Median TTFT: {metrics.median_ttft_ms:.2f} ms')
    print(f'P99 TTFT: {metrics.p99_ttft_ms:.2f} ms')
    print(f'Mean TPOT: {metrics.mean_tpot_ms:.2f} ms')
    print(f'Median TPOT: {metrics.median_tpot_ms:.2f} ms')
    print(f'P99 TPOT: {metrics.p99_tpot_ms:.2f} ms')

    result = {
        'duration': benchmark_duration,
        'completed': metrics.completed,
        'total_input_tokens': metrics.total_input,
        'total_output_tokens': metrics.total_output,
        'request_inthroughput': metrics.request_throughput,
        'input_throughput': metrics.input_throughput,
        'output_throughput': metrics.output_throughput,
        'mean_ttft_ms': metrics.mean_ttft_ms,
        'median_ttft_ms': metrics.median_ttft_ms,
        'p99_ttft_ms': metrics.p99_ttft_ms,
        'mean_tpot_ms': metrics.mean_tpot_ms,
        'median_tpot_ms': metrics.median_tpot_ms,
        'p99_tpot_ms': metrics.p99_tpot_ms
    }
    return result


def thread_function(args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    outputs = loop.run_until_complete(benchmark(**args))
    loop.close()
    return outputs


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f'{args.base_url}{args.endpoint}'
    else:
        api_url = f'http://{args.host}:{args.port}{args.endpoint}'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset,
        args.dataset_type, args.num_prompts, tokenizer)

    batch_size = int(np.ceil(len(input_requests) / args.num_parallel))
    print('num input requests', len(input_requests), batch_size)
    # warmup
    threads = []
    result_queue = Queue()
    for i in range(args.num_parallel):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_input_request = input_requests[start: end]
        params = dict(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=batch_input_request,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            stream=args.use_stream,
            proxy=args.proxy,
            system_prompt=args.system_prompt,
            disable_tqdm=args.disable_tqdm,
            task_id=i,
            result_queue=result_queue,
        )
        thread = threading.Thread(target=thread_function, args=(params,))
        threads.append(thread)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    results = []
    while not result_queue.empty():
        item = result_queue.get()
        results.append(item)

    results = sorted(results, key=lambda x: x[0])
    outputs = []
    benchmark_durations = []
    for r in results:
        outputs.extend(r[1])
        benchmark_durations.append(r[2])

    print('outputs', outputs)
    assert len(outputs) == len(input_requests)
    benchmark_result = merge_metrics(input_requests, outputs, benchmark_durations, tokenizer)

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        result_json['date'] = current_dt
        result_json['backend'] = backend
        result_json['version'] = args.version
        result_json['model_id'] = model_id
        result_json['tokenizer_id'] = tokenizer_id
        result_json['best_of'] = args.best_of
        result_json['use_beam_search'] = args.use_beam_search
        result_json['num_prompts'] = args.num_prompts

        # Num parallel and warmup
        result_json['num_parallel'] = arg.num_parallel
        result_json['num_warmup'] = arg.num_warmup

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split('/')[-1]
        file_name = f'{backend}-{args.num_parallel}qps-{base_model_id}-{current_dt}.json'
        with open(file_name, 'w') as outfile:
            json.dump(result_json, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the online serving throughput.')
    parser.add_argument(
        '--backend',
        type=str,
        default='openai.chat',
        help='server protocol.',
    )

    parser.add_argument(
        '--version',
        type=str,
        default='N/A',
        help='Version of the serving backend/engine.',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Server or API base url if not using http host and port.',
    )
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument(
        '--endpoint',
        type=str,
        default='/generate',
        help='API endpoint.',
    )
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Path to the dataset.')
    parser.add_argument('--dataset-type',
                        type=str,
                        default='conv',
                        help='dataset type.')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of the model.',
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        help=
        'Name or path of the tokenizer, if not using the default model tokenizer.',
    )
    parser.add_argument(
        '--best-of',
        type=int,
        default=1,
        help='Generates `best_of` sequences per prompt and '
        'returns the best one.',
    )
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=1000,
        help='Number of prompts to process.',
    )

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code from huggingface',
    )
    parser.add_argument(
        '--disable-tqdm',
        action='store_true',
        help='Specify to disable tqdm progress bar.',
    )
    parser.add_argument(
        '--save-result',
        action='store_true',
        help='Specify to save benchmark results to a json file',
    )

    parser.add_argument('--use-stream', action='store_true')
    parser.add_argument('--proxy', type=str, default='', help='proxy url')
    parser.add_argument('--system-prompt', type=str, default='You are a helpful assistant.', help='system prompt')
    parser.add_argument(
        '--num-parallel',
        type=int,
        default=1,
        help='number of thread',
    )

    parser.add_argument(
        '--num-warmup',
        type=int,
        default=1,
        help='number of warmup',
    )

    args = parser.parse_args()
    main(args)
