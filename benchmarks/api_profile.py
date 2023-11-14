# -*- coding: utf-8 -*-
# Please pull the latest code to run the profiling.
import time
import os
import json
import requests
from tqdm import tqdm


class ModelProfile:
    def __init__(self, model_name, model_url, **kwargs) -> None:
        self.timeout = 10000
        self.model_name = model_name
        self.model_url = model_url.format(model_name)
        self.client = requests.Session()
        self.test_samples = os.path.join("data", "alpaca_data_zh_51k.json")
        with open(os.path.join("data", "alpaca_data_zh_51k.json"), 'r') as f:
            self.test_samples = json.load(f)
            self.test_samples = [sample for sample in self.test_samples if "input" in sample and sample["input"] == ""]
            # print(self.test_samples, len(self.test_samples))
        
    def profile(self, max_experiment_times=70, warmup_steps=20):
        time_costs = []
        total_str = 0
        for index in tqdm(range(max_experiment_times)):
            context_str = self.test_samples[index]['instruction']
            chat_messages = [{'role': 'user', 'content': context_str}] 
            inp = {'model': self.model_name, 'messages': chat_messages}
            t1 = time.time()
            outp = self.client.post(url=self.model_url, json=inp, timeout=self.timeout).json()
            res_content = outp["choices"][0]["message"]["content"]
            print(index, res_content)
            total_str += len(res_content)
            time_costs.append(time.time() - t1)

            if index == warmup_steps - 1:
                time_costs = []
                total_str = 0

        print("Average generate speed (str/s): {}".format(total_str / sum(time_costs)))
        print("Experiment setting: ")
        print(f"max_experiment_times = {len(time_costs)}")
        print(f"generate_length_per_experiment = {total_str / len(time_costs)}")


if __name__ == "__main__":
    model_name = "Qwen-14B-Chat"
    model_url = "http://192.168.106.12:9005/v2.1/models/{}/infer"
    model_profile = ModelProfile(model_name, model_url)

    max_experiment_times = 70
    model_profile.profile(max_experiment_times=max_experiment_times)