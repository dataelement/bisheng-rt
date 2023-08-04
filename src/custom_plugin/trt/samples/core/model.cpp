#include "model.h"

bool Model::build() {
  _context->initPlugin();
  _context->loadWeightsMap(mParams.weightsFile);
  auto builder = _context->getIBuilder();
  if (!builder) {
    return false;
  }
  auto network = _context->getNetWorkDefine();
  if (!network) {
    return false;
  }

  auto config = _context->getIBuilderConfig();
  if (!config) {
    return false;
  }
  auto constructed = constructNetwork();
  if (!constructed) {
    return false;
  }
  return true;
}

bool Model::loadModel() {
  if (!_context->loadEngine(mParams.engineFile)) {
    return false;
  }
  return true;
}

bool Model::saveEngine() {
  if (!_context->saveEngine(mParams.engineFile)) {
    return false;
  }
  return true;
}

bool SampleModel::initIContext() {
  _context = makeUnique<IContext>(new IContext(&sample::gLogger.getTRTLogger(), false));
  return true;
}

bool SampleModel::initBuffer() {
  buffers = samplesCommon::BufferManager(
    _context->getICudaEngineShared(), mParams.batchSize);
  return true;
}

bool SampleModel::infer(int& batch_size,
                        std::vector<void*>& inputs,
                        std::vector<Dims>& dims,
                        std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) {
  auto status = _context->infer(batch_size, buffers, inputs, dims, outputs);
  if (!status) {
    return false;
  }
  return true;
}

bool SampleModelDynamic::initIContext() {
  _context = makeUnique<IContext>(new IContext(&sample::gLogger.getTRTLogger(), true));
  return true;
}

bool SampleModelDynamic::initBuffer() {
  for (unsigned int i = 0; i < mParams.inputTensorNames.size(); i++) {
    samplesCommon::ManagedBuffer buffer;
    _inputs_buffers.emplace_back(std::move(buffer));
  }
  for (unsigned int i = 0; i < mParams.outputTensorNames.size(); i++) {
    samplesCommon::ManagedBuffer buffer;
    _outputs_buffers.emplace_back(std::move(buffer));
  }
  return true;
}

bool SampleModelDynamic::infer(std::vector<void*>& inputs,
                               std::vector<Dims>& dims,
                               std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) {
  auto status = _context->infer(_inputs_buffers, inputs, dims, _outputs_buffers, outputs);
  if (!status) {
    return false;
  }
  return true;
}
