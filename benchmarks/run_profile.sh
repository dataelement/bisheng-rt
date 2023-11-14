run_qwen_7b_api() {
    python api_profile.py  --model_name Qwen-7B-Chat --model_url http://192.168.106.12:9001/v2.1/models/{}/infer --max_experiment_times 70 --warmup_steps 20
}

run_qwen_14b_api() {
    python api_profile.py --model_name Qwen-14B-Chat --model_url http://192.168.106.12:9001/v2.1/models/{}/infer --max_experiment_times 70 --warmup_steps 20
}

run_qwen_7b_api
# run_qwen_14b_api