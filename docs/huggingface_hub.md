## 支持通用的huggingface model 上下线和gpu资源分配

支持的模型类型： 
ChatLLM, CompletionLLM, Embedding, LayoutModel, AgentModel

```
message: []
prompt: ''
texts: []
images: []
```

### Request设计
1. Json

### Response设计
1. Json



config template 设计

```
name: "dataelem.model.huggingface_model"
backend: "python"
max_batch_size: 0

parameters { key: "model_type" value: { string_value: "dataelem.model.ChatLLM" } }
parameters { key: "model_parameters" value: { string_value: "{}" } }

input [
  {
    name: "INPUT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```


1. 下载huggingface模型文件，将模型导入到/opt/model_repository/xxx_model

2. 发起load模型请求

```
curl -X post /v2/repository/models/xxx_model/load -d 
"""
{
  "parameters" : {
    "type": "dataelem.pymodel.huggingface_model",
    "pymodel_type": "llm.ChatGLM2",
    "precision": "fp16",
    "instance_groups": "device=gpu;gpus=0,1|2",
    "minimum_gpu_memory": "16g"
  }
}
"""

```

上述配置为启动两个实例，第一个实例占用gpu 0卡和1卡，第二个实例占用2卡


1. 拉起python backend的huggingface_model模型（graph路径下），实例名字是对应的xxx_model
2. 更新xxx_model的路径到parameters参数中


LoadModelsFromConfig逻辑

1. 修改models参数
```
 std::unordered_map<std::string, std::vector<const InferenceParameter*>>
      models;
```

2. 调用Poll过程
3. 调用UpdateDependencyGraph
4. 调用LoadModelByDependency


LoadUnloadModel逻辑：

0. 修改models参数
1. LoadUnloadModels
  - 调用Poll
  - 调用UpdateDependencyGraph
  - 调用LoadModelByDependency