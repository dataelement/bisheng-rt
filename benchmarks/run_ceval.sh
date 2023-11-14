# wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
# mkdir data/ceval
# mv ceval-exam.zip data/ceval
# cd data/ceval; unzip ceval-exam.zip
# cd ../../

# pip install thefuzz

run_chatglm2_6b_local() {
    CUDA_VISIBLE_DEVICES=1 python evaluate_chat_ceval.py -d data/ceval/ -c /home/public/llm/chatglm2-6b --save_path outs_chat/ceval_eval_result_chatglm2_6b
} 

run_qwen_7b_local() {
    CUDA_VISIBLE_DEVICES=1 python evaluate_chat_ceval.py -d data/ceval/ -c /home/public/llm/Qwen-7B-Chat --save_path outs_chat/ceval_eval_result_qwen_7b
}

run_qwen_7b_api() {
    python api_evaluate_chat_ceval.py -d data/ceval/ --model_name Qwen-7B-Chat --model_url http://192.168.106.12:9001/v2.1/models/{}/infer --save_path outs_chat/ceval_eval_result_qwen_7b_api_hf
}

run_qwen_14b_api() {
    python api_evaluate_chat_ceval.py -d data/ceval/ --model_name Qwen-14B-Chat --model_url http://192.168.106.12:9001/v2.1/models/{}/infer --save_path outs_chat/ceval_eval_result_qwen_14b_api_hf
}

# run_chatglm2_6b_local
# run_qwen_7b_local
# run_qwen_7b_api
run_qwen_14b_api