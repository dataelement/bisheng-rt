#!/bin/bash


dataset="/public/bisheng/bisheng-test-data/rt_data/ShareGPT_V3_unfiltered_cleaned_split.json"
model="/opt/bisheng-rt/models/model_repository/Qwen-1_8B-Chat"
model_name="models/model_repository/Qwen-1_8B-Chat"
# python3 benchmark_serving.py \
#     --backend vllm \
#     --base-url http://192.168.106.20:9001 \
#     --endpoint /generate \
#     --num-prompts 100 \
#     --tokenizer $model \
#     --model $model_name \
#     --dataset $dataset \
#     --trust-remote-code \
#     --seed 0 --request-rate 100


# model_name="Qwen-1_8B-Chat"
# python3 benchmark_serving.py \
#     --backend openai \
#     --base-url http://192.168.106.20:9001 \
#     --endpoint /v1/completions \
#     --num-prompts 100 \
#     --tokenizer $model \
#     --model $model_name \
#     --dataset $dataset \
#     --trust-remote-code \
#     --seed 0 --request-rate 100.0


#model_name="Qwen-1_8B-Chat"
#dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
#python3 benchmark_serving.py \
#    --backend openai.chat \
#    --base-url http://192.168.106.20:28001 \
#    --endpoint /v1/chat/completions \
#    --tokenizer $model \
#    --model $model_name \
#    --dataset $dataset --dataset-type instruct \
#    --trust-remote-code \
#    --num-prompts 2000 --seed 2024 --request-rate 50


#model_name="Qwen-1_8B-Chat"
#dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
#python3 benchmark_serving.py \
#    --backend openai.chat \
#    --base-url http://192.168.106.20:28001 \
#    --endpoint /v1/chat/completions \
#    --tokenizer $model \
#    --model $model_name \
#    --dataset $dataset --dataset-type instruct \
#    --trust-remote-code \
#    --num-prompts 5000 --seed 1947 --request-rate 50 --use-stream


#model_name="Qwen-1_8B-Chat"
#dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
#python3 benchmark_serving.py \
#    --backend openai.chat \
#    --base-url http://192.168.106.20:9001 \
#    --endpoint /v1/chat/completions \
#    --tokenizer $model \
#    --model $model_name \
#    --dataset $dataset --dataset-type instruct \
#    --trust-remote-code \
#    --num-prompts 5000 --seed 1947 --request-rate 50 --use-stream
#

#model_name="Qwen-1_8B-Chat"
#dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
#python3 benchmark_serving.py \
#    --backend openai.chat \
#    --base-url http://192.168.106.20:9001 \
#    --endpoint /v1/chat/completions \
#    --tokenizer $model \
#    --model $model_name \
#    --dataset $dataset --dataset-type instruct \
#    --trust-remote-code \
#    --num-prompts 2000 --seed 2024 --request-rate 50

# kimi
#model_name="moonshot-v1-8k"
#dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
#export OPENAI_API_KEY=""
#python3 benchmark_serving.py \
#    --backend openai.chat \
#    --base-url https://api.moonshot.cn \
#    --endpoint /v1/chat/completions \
#    --tokenizer $model \
#    --model $model_name \
#    --dataset $dataset --dataset-type instruct \
#    --trust-remote-code \
#    --num-prompts 100 --seed 1947 --request-rate 1 --use-stream

# openai
# model_name="gpt-4-1106-preview"
# dataset="/public/bisheng/bisheng-test-data/rt_data/alpaca_data_zh_51k.json"
# export OPENAI_API_KEY=""
# python3 benchmark_serving.py \
#     --backend openai.chat \
#     --base-url https://api.openai.com \
#     --endpoint /v1/chat/completions \
#     --tokenizer $model \
#     --model $model_name \
#     --dataset $dataset --dataset-type instruct \
#     --trust-remote-code \
#     --num-prompts 3 --seed 1947 --request-rate 1 --use-stream \
#     --proxy http://118.195.232.223:39995
