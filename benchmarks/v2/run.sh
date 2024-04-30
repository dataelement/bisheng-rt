#!/bin/bash


function perf1() {
  dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
  model="/public/bisheng/model_repository/Qwen-1_8B-Chat"
  model_name="Qwen1.5-72B-Chat"

  python3 llm_perf_v2.py \
     --backend openai.chat \
     --base-url http://34.87.129.78:9300 \
     --endpoint /v1/chat/completions \
     --tokenizer $model \
     --model $model_name \
     --dataset $dataset --dataset-type instruct \
     --trust-remote-code \
     --num-prompts 10 --seed 1947 --num-parallel 2 --use-stream
}

perf1