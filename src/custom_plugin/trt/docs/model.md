# Model模块

模型应用基类，east、mrcnn、transformer、trans-ctc继承此基类。其中接口函数有：

## Model()

函数定义：
```
Model(samplesCommon::SampleParams& params)
```
参数：
* params：模型应用所需要的参数（包含batchSize、fp16、dataDirs、inputTensorNames、outputTensorNames、weightsFile、engineFile）

构造函数：初始化模型应用所需要的参数。

## build()

函数定义：

```
virtual bool build()
```
模型网络创建，会初始化plugin、载入权重、创建builder、network、config

## loadModel()

函数定义：

```
virtual bool loadModel()
```
载入engine文件，解序列化ICudaEngine

## saveEngine()

函数定义：

```
virtual bool saveEngine()
```
序列化ICudaEngine，保存模型

## initBuffer()

函数定义：

```
virtual bool initBuffer() = 0
```
初始化输入输出buffer，动态静态有区别

## initIContext()

函数定义：

```
virtual bool initIContext() = 0
```
创建IExecutionContext，并初始化，动态静态有区别

## constructNetwork()

函数定义：

```
virtual bool constructNetwork() = 0
```
搭建tensorrt网络接口，这边只提供了接口，需要给出具体实现（不同模型不一样，自己实现）。可以参考east、mrcnn、transfomer、tran_ctc实现。


# SampleModel模块

继承model基类，静态模型应用类，给出了initBuffer、initIContext、infer具体实现

## infer()

函数定义：

```
virtual bool infer(int& batch_size, std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) = 0
```
参数：
* batch_size：静态shape下需要指定实际预估的batch大小，不能超过最大的batch大小
* inputs：输入数据地址
* dims：输入数据大小
* outputs：输出数据地址、大小

模型预估接口

# SampleModelDynamic模块

继承model基类，动态模型应用类，给出了initBuffer、initIContext、infer具体实现

## infer()

函数定义：

```
virtual bool infer(std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) = 0
```
参数：
* inputs：输入数据地址
* dims：输入数据大小
* outputs：输出数据地址、大小

模型预估接口

