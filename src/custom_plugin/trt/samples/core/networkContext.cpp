#include "networkContext.h"

void IContext::loadWeightsMap(std::string weight_file) {
  std::ifstream input(weight_file, std::ios::binary);
  assert(input.is_open() && "Unable to load weight file.");

  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    uint32_t size;
    std::string name;
    int t;
    input >> name >> std::dec >> t >> size;
    auto type = static_cast<DataType>(t);
    assert (type == nvinfer1::DataType::kFLOAT);
    std::vector<uint32_t> val(size);
    for (uint32_t x = 0; x < size; ++x) {
      input >> std::hex >> val[x];
    }
    std::vector<uint8_t> val_char(size * sizeof(uint32_t));
    memcpy((unsigned char*)val_char.data(), (unsigned char *)val.data(), size * sizeof(uint32_t));
    _bufs.push_back(val_char);
    nvinfer1::Weights wt{type, nullptr, size};
    wt.values = _bufs.back().data();
    _weights_map[name] = wt;
  }
}

nvinfer1::Weights IContext::getWeightsByName(std::string name) {
  auto flag = _weights_map.find(name);
  if (flag != _weights_map.end()) {
    return _weights_map[name];
  } else {
    std::vector<float> val;
    return createTempWeights<float>(val);
  }
}

nvinfer1::IBuilder* IContext::getIBuilder() {
  if (!_builder) {
    _builder = makeShared<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  }
  return _builder.get();
}

nvinfer1::INetworkDefinition* IContext::getNetWorkDefine() {
  if (!_network) {
    if (!_dynamic_shape) {
      _network = makeShared<nvinfer1::INetworkDefinition>(getIBuilder()->createNetworkV2(0));
    } else {
      const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
      _network = makeShared<nvinfer1::INetworkDefinition>(getIBuilder()->createNetworkV2(explicitBatch));
    }
  }
  return _network.get();
}

nvinfer1::IBuilderConfig* IContext::getIBuilderConfig() {
  if (!_builder_config) {
    _builder_config = makeShared<nvinfer1::IBuilderConfig>(getIBuilder()->createBuilderConfig());
  }
  return _builder_config.get();
}

nvinfer1::ICudaEngine* IContext::getICudaEngine() {
  if (!_cuda_engine) {
    _cuda_engine = makeShared<nvinfer1::ICudaEngine>(getIBuilder()->buildEngineWithConfig(*getNetWorkDefine(), *getIBuilderConfig()));
  }
  return _cuda_engine.get();
}

std::shared_ptr<nvinfer1::ICudaEngine> IContext::getICudaEngineShared() {
  if (!_cuda_engine) {
    _cuda_engine = makeShared<nvinfer1::ICudaEngine>(getIBuilder()->buildEngineWithConfig(*getNetWorkDefine(), *getIBuilderConfig()));
  }
  return _cuda_engine;
}

nvinfer1::IExecutionContext* IContext::getIExecutionContext() {
  if (!_excute_context) {
    _excute_context = makeShared<nvinfer1::IExecutionContext>(getICudaEngine()->createExecutionContext());
  }
  return _excute_context.get();
}

void IContext::setOptimizationProfile(std::vector<std::vector<Dims>>& inputsProfileDims) {
  auto profile = _builder->createOptimizationProfile();
  for (unsigned int i = 0; i < _inputNames.size(); i++) {
    std::vector<Dims> dims = inputsProfileDims[i];
    profile->setDimensions(_network->getInput(i)->getName(), OptProfileSelector::kMIN, dims[0]);
    profile->setDimensions(_network->getInput(i)->getName(), OptProfileSelector::kOPT, dims[1]);
    profile->setDimensions(_network->getInput(i)->getName(), OptProfileSelector::kMAX, dims[2]);
  }
  _builder_config->addOptimizationProfile(profile);
}

bool IContext::saveEngine(const std::string& engine_file) {
  std::ofstream engineFile(engine_file, std::ios::binary);
  if (!engineFile) {
    return false;
  }

  TrtUniquePtr<IHostMemory> serializedEngine{(*getICudaEngine()).serialize()};
  if (serializedEngine == nullptr) {
    return false;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
  return !engineFile.fail();
}

bool IContext::loadEngine(const std::string& engine_file) {
  std::ifstream infile(engine_file, std::ios::in|std::ios::binary);
  if(!infile) {
    return false;
  }
  infile.seekg(0, std::ios::end);
  auto size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::vector<char> buffer((size_t)size+1);
  infile.read(buffer.data(), size);
  initPlugin();
  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  _cuda_engine = makeShared<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(
                   reinterpret_cast<char*>(buffer.data()), size, nullptr));
  if (!_cuda_engine) {
    std::cout<<"Engine load failed!"<<std::endl;
    return false;
  }

  auto bindings = _cuda_engine->getNbBindings();
  for (int i = 0; i < bindings; i++) {
    if (_cuda_engine->bindingIsInput(i)) {
      _inputNames.push_back(_cuda_engine->getBindingName(i));
    } else {
      _outputNames.push_back(_cuda_engine->getBindingName(i));
    }
  }
  return true;
}

template<class T>
nvinfer1::Weights IContext::createTempWeights(std::vector<T> vec) {
  nvinfer1::DataType type;
  int size_w = vec.size();
  int n;
  if (typeid(T) == typeid(uint32_t) || typeid(T) == typeid(int32_t)) {
    n = size_w * 4;
    type = nvinfer1::DataType::kINT32;
  } else if (typeid(T) == typeid(uint8_t) || typeid(T) == typeid(int8_t)) {
    n = size_w * 1;
    type = nvinfer1::DataType::kINT8;
  } else if (typeid(T) == typeid(float)) {
    n = size_w * 4;
    type = nvinfer1::DataType::kFLOAT;
  } else if (typeid(T) == typeid(half)) {
    n = size_w * 2;
    type = nvinfer1::DataType::kHALF;
  }
  if (size_w == 0) {
    return nvinfer1::Weights{type, nullptr, 0};
  }
  std::vector<uint8_t> val(n);
  memcpy((unsigned char*)val.data(), (unsigned char *)vec.data(), n);
  _bufs.push_back(val);
  nvinfer1::Weights weights{type, nullptr, size_w};
  weights.values = _bufs.back().data();
  return weights;
}

std::vector<ITensor*> IContext::setInputNode(const std::vector<std::string>& inputNames,
    const std::vector<nvinfer1::Dims>& input_dims,
    const std::vector<nvinfer1::DataType>& types) {
  std::vector<ITensor*> outputs;
  for (unsigned int i = 0; i < inputNames.size(); i++) {
    _inputNames.push_back(inputNames[i]);
    ITensor* inputs = getNetWorkDefine()->addInput(inputNames[i].c_str(), types[i], input_dims[i]);
    assert(inputs);
    outputs.push_back(inputs);
  }
  return outputs;
}

void IContext::setOutputNode(std::vector<ITensor*>& outputs, std::vector<std::string>& outputNames) {
  for (unsigned int i = 0; i < outputNames.size(); i++) {
    _outputNames.push_back(outputNames[i]);
    outputs[i]->setName(outputNames[i].c_str());
    getNetWorkDefine()->markOutput(*outputs[i]);
  }
}

bool IContext::infer(int batch_size, samplesCommon::BufferManager& buffers,
                     std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::map<std::string, std::pair<void*,
                     nvinfer1::Dims>>& outputs) {
  for (unsigned int i = 0; i < inputs.size(); i++) {
    int index = getIExecutionContext()->getEngine().getBindingIndex(_inputNames[i].c_str());
    nvinfer1::DataType type = getIExecutionContext()->getEngine().getBindingDataType(index);
    int _nbytes = DimsCount(dims[i]);
    if (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32) {
      _nbytes *= 4;
    } else if (type ==  nvinfer1::DataType::kHALF) {
      _nbytes *= 2;
    } else if (type == nvinfer1::DataType::kINT8 || type == nvinfer1::DataType::kBOOL) {
      _nbytes *= 1;
    }
    void* hostInputBuffer = buffers.getHostBuffer(_inputNames[i]);
    std::memcpy(hostInputBuffer, inputs[i], _nbytes);
  }
  buffers.copyInputToDevice();
  auto status = getIExecutionContext()->execute(batch_size, buffers.getDeviceBindings().data());
  buffers.copyOutputToHost();
  for (unsigned int i = 0; i < _outputNames.size(); i++) {
    void* outputs_buffer = buffers.getHostBuffer(_outputNames[i]);
    nvinfer1::Dims outputs_dims = getIExecutionContext()->getEngine()
                                  .getBindingDimensions(getIExecutionContext()->getEngine()
                                      .getBindingIndex(_outputNames[i].c_str()));
    outputs[_outputNames[i]] = std::make_pair(outputs_buffer, outputs_dims);
  }
  if (!status) {
    return false;
  }
  return true;
}

bool IContext::infer(std::vector<samplesCommon::ManagedBuffer>& inputs_buffers,
                     std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::vector<samplesCommon::ManagedBuffer>& outputs_buffers,
                     std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) {
  std::vector<void*> bindings;
  for (unsigned int i = 0; i < inputs.size(); i++) {
    int index = getIExecutionContext()->getEngine().getBindingIndex(_inputNames[i].c_str());
    nvinfer1::DataType type = getIExecutionContext()->getEngine().getBindingDataType(index);
    int wordSize;
    if (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32) {
      wordSize = 4;
    } else if (type ==  nvinfer1::DataType::kHALF) {
      wordSize = 2;
    } else if (type == nvinfer1::DataType::kINT8 || type == nvinfer1::DataType::kBOOL) {
      wordSize = 1;
    }
    getIExecutionContext()->setBindingDimensions(index, dims[i]);
    inputs_buffers[i].hostBuffer.resize(dims[i]);
    inputs_buffers[i].deviceBuffer.resize(dims[i]);
    void* hostDataBuffer = inputs_buffers[i].hostBuffer.data();
    int size = DimsCount(dims[i]);
    memcpy(hostDataBuffer, inputs[i], size * wordSize);
    CHECK(cudaMemcpy(inputs_buffers[i].deviceBuffer.data(), inputs_buffers[i].hostBuffer.data(),
                     inputs_buffers[i].hostBuffer.nbBytes(), cudaMemcpyHostToDevice));
    bindings.push_back(inputs_buffers[i].deviceBuffer.data());
  }
  std::map<std::string, Dims> outputs_dims;
  for (unsigned int j = 0; j < outputs_buffers.size(); j++) {
    nvinfer1::Dims dim = getIExecutionContext()->getBindingDimensions(getIExecutionContext()->getEngine()
                         .getBindingIndex(_outputNames[j].c_str()));
    outputs_dims[_outputNames[j]] = dim;
    outputs_buffers[j].hostBuffer.resize(dim);
    outputs_buffers[j].deviceBuffer.resize(dim);
    bindings.push_back(outputs_buffers[j].deviceBuffer.data());
  }
  auto status = getIExecutionContext()->executeV2(bindings.data());
  for (unsigned int j = 0; j < outputs_buffers.size(); j++) {
    CHECK(cudaMemcpy(outputs_buffers[j].hostBuffer.data(), outputs_buffers[j].deviceBuffer.data(),
                     outputs_buffers[j].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    void* outputs_buffer = outputs_buffers[j].hostBuffer.data();
    outputs[_outputNames[j]] = std::make_pair(outputs_buffer, outputs_dims[_outputNames[j]]);
  }

  if (!status) {
    return false;
  }
  return true;
}

template nvinfer1::Weights IContext::createTempWeights<float>(std::vector<float> vec);

template nvinfer1::Weights IContext::createTempWeights<int>(std::vector<int> vec);

template nvinfer1::Weights IContext::createTempWeights<half>(std::vector<half> vec);



