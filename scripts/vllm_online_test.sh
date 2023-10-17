#使用huggingface加载模型
function load_data0() {
  cat <<EOF
{
  "parameters": {
    "type": "dataelem.pymodel.huggingface_model",
    "pymodel_type": "llm.QwenChat",
    "precision": "bf16",
    "gpu_memory": "26",
    "instance_groups": "device=gpu;gpus=0,1,2,3"
  }
}
EOF
}


function load_data() {
  cat <<EOF
{
  "parameters": {
    "type": "dataelem.pymodel.vllm_model",
    "pymodel_type": "llm.Qwen-7B-Chat",
    "pymodel_params": "{\"max_tokens\": 4096}",
    "gpu_memory": "36",
    "instance_groups": "device=gpu;gpus=0,1"
  }
}
EOF
}

function load_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2/repository/models/${model}/load \
   -H 'Content-Type: application/json' \
   -d "$(load_data)"
}


function unload_model() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2/repository/models/${model}/unload \
   -H 'Content-Type: application/json' \
   -d '{}'
}
function infer_data4() {
  cat <<EOF
{

  "model": "Qwen-7B-Chat",
  "messages": [
    {"role": "user", "content": "The future of AI is"}
   ],
  "stream": true,
  "sampling_parameters":{"n":4,
                         "use_beam_search":true,
                         "temperature":0,
                         "best_of":4,
                         "top_p":1.0,
                         "top_k":-1,
                         "ignore_eos":false,
                         "max_tokens":16 
                        }
}
EOF
}

#使用buggingface预估
function infer_data5() {
  cat <<EOF
{

  "model": "Qwen-7B-Chat",
  "messages": [
    {"role": "user", "content": "The future of AI is"}
   ]
}
EOF
}

function model_infer() {
  model="$1"
  curl -v -X POST http://127.0.0.1:9001/v2.1/models/${model}/infer \
   -H 'Content-Type: application/json' \
   -d "$(infer_data4)"
}


#Qwen-7B-Chat

#huggingface:
#1.修改model_infer中infer_data4  -> infer_data5    
#2.修改load_model中load_data  ->   load_data0


#vllm:
#1.model_infer中配置为infer_data4
#2.load_model中配置为load_data  


#load_model Qwen-7B-Chat
load_model Qwen-7B-Chat  && model_infer Qwen-7B-Chat
#model_infer Qwen-7B-Chat