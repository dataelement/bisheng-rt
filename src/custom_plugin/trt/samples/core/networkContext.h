#ifndef TRT_NETWORK_ICONTEXT_H
#define TRT_NETWORK_ICONTEXT_H

#include "argsParser.h"
#include "logger.h"
#include "NvInfer.h"
#include "netUtils.h"
#include "buffers.h"

#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <list>
#include <typeinfo>
#include <utility>
#include <map>
#include <cuda_fp16.h>

class IContext {
 public:
  IContext(nvinfer1::ILogger* logger, bool dynamic_shape): _logger(logger), _dynamic_shape(dynamic_shape) {

  }

  void setBatchSize(int batch_size) {
    _builder->setMaxBatchSize(batch_size);
  }

  void setWorkspaceSize(std::size_t workspace_ize) {
    _builder_config->setMaxWorkspaceSize(workspace_ize);
  }

  void setFp16Mode() {
    // _builder->setFp16Mode(true);
    _builder_config->setFlag(BuilderFlag::kFP16);
  }

  void setInt8Mode() {
  }

  void initPlugin() {
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
  }

  nvinfer1::ILogger& logger() {
    return *_logger;
  }

  void loadWeightsMap(std::string weight_file);

  nvinfer1::Weights getWeightsByName(std::string name);

  nvinfer1::IBuilder* getIBuilder();

  nvinfer1::INetworkDefinition* getNetWorkDefine();

  nvinfer1::IBuilderConfig* getIBuilderConfig();

  nvinfer1::ICudaEngine* getICudaEngine();

  std::shared_ptr<nvinfer1::ICudaEngine> getICudaEngineShared();

  nvinfer1::IExecutionContext* getIExecutionContext();

  void setOptimizationProfile(std::vector<std::vector<Dims>>& inputsProfileDims);

  bool saveEngine(const std::string& engine_file);

  bool loadEngine(const std::string& engine_file);

  template<class T>
  nvinfer1::Weights createTempWeights(std::vector<T> vec);

  std::vector<ITensor*> setInputNode(const std::vector<std::string>& inputNames,
                                     const std::vector<nvinfer1::Dims>& input_dims,
                                     const std::vector<nvinfer1::DataType>& types);

  void setOutputNode(std::vector<ITensor*>& outputs, std::vector<std::string>& outputNames);

  // 静态shape和动态shape的infer通过函数重载
  bool infer(int batch_size, samplesCommon::BufferManager& _buffers,
             std::vector<void*>& inputs,
             std::vector<Dims>& dims,
             std::map<std::string, std::pair<void*,
             nvinfer1::Dims>>& outputs);

  bool infer(std::vector<samplesCommon::ManagedBuffer>& inputs_buffers,
             std::vector<void*>& inputs,
             std::vector<Dims>& dims,
             std::vector<samplesCommon::ManagedBuffer>& outputs_buffers,
             std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs);

  void destroy() {
    // _excute_context, _cuda_engine的destroy顺序不能反，
    // 先destroy _cuda_engine再destroy _excute_context会出现内存泄露
    // 怀疑context是engine的一部分
    if (_excute_context) _excute_context->destroy();
    if (_cuda_engine) _cuda_engine->destroy();
    if (_builder_config) _builder_config->destroy();
    if (_network) _network->destroy();
    if (_builder) _builder->destroy();
  }

  ~IContext() {}

 private:
  std::map<std::string, nvinfer1::Weights> _weights_map;
  std::shared_ptr<nvinfer1::IBuilder> _builder;
  std::shared_ptr<nvinfer1::INetworkDefinition> _network;
  std::shared_ptr<nvinfer1::IBuilderConfig> _builder_config;
  std::shared_ptr<nvinfer1::ICudaEngine> _cuda_engine;
  std::shared_ptr<nvinfer1::IExecutionContext> _excute_context;
  std::list<std::vector<uint8_t>> _bufs;
  nvinfer1::ILogger* _logger;
  std::vector<std::string> _inputNames;
  std::vector<std::string> _outputNames;
  bool _dynamic_shape;
};

#endif
