# IContext模块

统一管理tensorrt中IBuilder、INetworkDefinition、IBuilderConfig、IExecutionContext、ICudaEngine、网络权重分配、临时变量分配，管理执行上下文。其中公有成员函数有：

## IContext()

函数定义：
```
IContext(nvinfer1::ILogger* logger, bool dynamic_shape): _logger(logger), _dynamic_shape(dynamic_shape)
```
参数：
* logger：tensorrt内部定义的logger类
* dynamic_shape：false：对应tensorrt隐式batch，true：对应tensorrt显式batch，对应的tensorrt网络创建不一样

构造函数：初始化nvinfer1::ILogger和dynamic_shape。

## setBatchSize()

函数定义：

```
void setBatchSize(int batch_size)
```
参数：
* batch_size：batch_size大小

设置tensorrt网络支持的最大batch_size大小，只有在隐式batch需要设置

## setWorkspaceSize()

函数定义：

```
void setWorkspaceSize(std::size_t workspace_ize)
```
参数：
* workspace_ize：分配最大显存空间大小

设置tensorrt网络运行时所需要的最大显存空间大小

## setFp16Mode()

函数定义：

```
void setFp16Mode()
```
是否用fp16来运行tensorrt网络

## setInt8Mode()

函数定义：

```
void setInt8Mode()
```
是否用int8来运行tensorrt网络，目前暂时不支持

## initPlugin()

函数定义：

```
void initPlugin()
```
注册自定义plugin，并初始化

## logger()

函数定义：

```
nvinfer1::ILogger& logger()
```
返回tensorrt logger实例

## loadWeightsMap()

函数定义：

```
void loadWeightsMap(std::string weight_file)
```
参数：
* weight_file：tensorrt网络权重文件

读取tensorrt网络权重文件，并保存到weights_map（std::map<std::string, nvinfer1::Weights>）中，权重都展平到一维处理

## getWeightsByName()

函数定义：

```
nvinfer1::Weights getWeightsByName(std::string name)
```
参数：
* name：权重名字

通过网络权重名字，得到对应的权重nvinfer1::Weights

## getNetWorkDefine()

函数定义：

```
nvinfer1::INetworkDefinition* getNetWorkDefine()
```
返回tensorrt指向INetworkDefinition对象的指针，如果没有INetworkDefinition对象，则创建并返回指针

## getIBuilderConfig()

函数定义：

```
nvinfer1::IBuilderConfig* getIBuilderConfig()
```
返回tensorrt指向IBuilderConfig对象的指针，如果没有IBuilderConfig对象，则创建并返回指针

## getICudaEngine()

函数定义：

```
nvinfer1::ICudaEngine* getICudaEngine()
```
返回tensorrt指向ICudaEngine对象的指针，如果没有ICudaEngine对象，则创建并返回指针

## getICudaEngineShared()

函数定义：

```
std::shared_ptr<nvinfer1::ICudaEngine> getICudaEngineShared()
```
返回tensorrt指向ICudaEngine对象的智能指针，如果没有ICudaEngine对象，则创建并返回智能指针

## getIExecutionContext()

函数定义：

```
nvinfer1::IExecutionContext* getIExecutionContext()
```
返回tensorrt指向IExecutionContext对象的指针，如果没有IExecutionContext对象，则创建并返回指针

## setOptimizationProfile()

函数定义：

```
void setOptimizationProfile(std::vector<std::vector<Dims>>& inputsProfileDims)
```
参数：
* inputsProfileDims：输入tensors的大小范围

设置输入tensor的大小范围，min/opt/max，只有在显式batch下（dynamic_shape=true）需要设置

## saveEngine()

函数定义：

```
bool saveEngine(const std::string& engine_file)
```
参数：
* engine_file：tensorrt模型保存文件

将tensorrt优化后的ICudaEngine序列化，并保存到engine_file文件中

## loadEngine()

函数定义：

```
bool IContext::loadEngine(const std::string& engine_file)
```
参数：
* engine_file：tensorrt模型保存文件

从engine_file中解序列化ICudaEngine

## createTempWeights()

函数定义：

```
nvinfer1::Weights createTempWeights(std::vector<T> vec)
```
参数：
* vec：临时变量保存在一维数组

根据创建好的一维数组，创建临时权重，并通过IContext全局bufs托管

## setInputNode()

函数定义：

```
std::vector<ITensor*> setInputNode(const std::vector<std::string>& inputNames,
                                   const std::vector<nvinfer1::Dims>& input_dims,
                                   const std::vector<nvinfer1::DataType>& types)
```
参数：
* inputNames：输入tensor的名字
* input_dims：输入tensor的维度
* types：输入tensor的类别

设置tensorrt的输入节点，并加入到网络中

## setOutputNode()

函数定义：

```
void setOutputNode(std::vector<ITensor*>& outputs, std::vector<std::string>& outputNames)
```
参数：
* outputs：tensorrt网络输出节点
* outputNames：tensorrt网络输出节点名字

将tensorrt网络输出节点和节点名字绑定起来

## infer()

函数定义：

```
bool infer(int batch_size, samplesCommon::BufferManager& _buffers,
             std::vector<void*>& inputs,
             std::vector<Dims>& dims,
             std::map<std::string, std::pair<void*,
             nvinfer1::Dims>>& outputs)
```
参数：
* batch_size：infer网络输入的batch大小
* buffers：用来管理输入输出节点在cpu、gpu上空间的开辟和释放，只适用于隐式batch
* inputs：输入数据，给定输入数据的地址
* dims：输入数据的大小
* outputs：输出数据的名字、值和大小

静态网络下（dynamic_shape=false），tensorrt网络预估接口，需要指定输入的batch大小

## infer()

函数定义：

```
bool infer(std::vector<samplesCommon::ManagedBuffer>& inputs_buffers,
             std::vector<void*>& inputs,
             std::vector<Dims>& dims,
             std::vector<samplesCommon::ManagedBuffer>& outputs_buffers,
             std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs)
```
参数：
* inputs_buffers：用来管理输入节点在cpu、gpu上空间的开辟和释放
* outputs_buffers：用来管理输出节点在cpu、gpu上空间的开辟和释放
* inputs：输入数据，给定输入数据的地址
* dims：输入数据的大小
* outputs：输出数据的名字、值和大小

动态网络下（dynamic_shape=true），tensorrt网络预估接口



