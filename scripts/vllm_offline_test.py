#本地启动服务：（106.20容器环境中，已经手动修改默认加载的模型为qwen-7b）
python3 -m vllm.entrypoints.api_server



#请求：
curl http://127.0.0.1:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'