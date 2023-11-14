# Please pull the latest code to run the profiling.
import argparse
import json
import os
import time

import requests
from tqdm import tqdm


class ModelProfile:
    def __init__(self, model_name, model_url, **kwargs) -> None:
        self.timeout = 10000
        self.model_name = model_name
        self.model_url = model_url.format(model_name)
        self.client = requests.Session()
        self.test_samples = os.path.join('data', 'alpaca_data_zh_51k.json')
        with open(os.path.join('data', 'alpaca_data_zh_51k.json'), 'r') as f:
            self.test_samples = json.load(f)
            self.test_samples = [
                sample for sample in self.test_samples
                if 'input' in sample and sample['input'] == '']
            # print(self.test_samples, len(self.test_samples))

    def profile(self, max_experiment_times=70, warmup_steps=20):
        time_costs = []
        total_str = 0
        for index in tqdm(range(max_experiment_times)):
            context_str = self.test_samples[index]['instruction']
            chat_messages = [{'role': 'user', 'content': context_str}]
            inp = {'model': self.model_name, 'messages': chat_messages}
            t1 = time.time()
            outp = self.client.post(url=self.model_url,
                                    json=inp, timeout=self.timeout).json()
            res_content = outp['choices'][0]['message']['content']
            print(index, res_content)
            total_str += len(res_content)
            time_costs.append(time.time() - t1)

            if index == warmup_steps - 1:
                time_costs = []
                total_str = 0

        print('Average generate speed (str/s): {}'.format(
            total_str / sum(time_costs)))
        print('Experiment setting: ')
        print(f'max_experiment_times = {len(time_costs)}')
        print(f'generate_length_per_experiment = \
              {total_str / len(time_costs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model profile.')
    parser.add_argument('--model_name', type=str, default='Qwen-7B-Chat')
    parser.add_argument('--model_url', type=str, default='')
    parser.add_argument('--max_experiment_times', type=int, default=70)
    parser.add_argument('--warmup_steps', type=int, default=20)
    args = parser.parse_args()

    model_profile = ModelProfile(args.model_name, args.model_url)
    model_profile.profile(max_experiment_times=args.max_experiment_times,
                          warmup_steps=args.warmup_steps)
