/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ocr_specialSlicePlugin.h"
#include "maskRCNNKernels.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::OcrSpecialSlice;
using nvinfer1::plugin::OcrSpecialSlicePluginCreator;

namespace {
const char* SPECIALSLICE_PLUGIN_VERSION{"1"};
const char* SPECIALSLICE_PLUGIN_NAME{"OcrSpecialSlice_TRT"};
} // namespace

PluginFieldCollection OcrSpecialSlicePluginCreator::mFC{};
std::vector<PluginField> OcrSpecialSlicePluginCreator::mPluginAttributes;

OcrSpecialSlicePluginCreator::OcrSpecialSlicePluginCreator() {

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* OcrSpecialSlicePluginCreator::getPluginName() const noexcept {
  return SPECIALSLICE_PLUGIN_NAME;
};

const char* OcrSpecialSlicePluginCreator::getPluginVersion() const noexcept {
  return SPECIALSLICE_PLUGIN_VERSION;
};

const PluginFieldCollection* OcrSpecialSlicePluginCreator::getFieldNames() noexcept {
  return &mFC;
};

IPluginV2Ext* OcrSpecialSlicePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  return new OcrSpecialSlice();
};

IPluginV2Ext* OcrSpecialSlicePluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept {
  return new OcrSpecialSlice(data, length);
};

size_t OcrSpecialSlice::getWorkspaceSize(int) const noexcept {
  return 0;
}

bool OcrSpecialSlice::supportsFormat(DataType type, PluginFormat format) const noexcept {
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
};

const char* OcrSpecialSlice::getPluginType() const noexcept {
  return "OcrSpecialSlice_TRT";
};

const char* OcrSpecialSlice::getPluginVersion() const noexcept {
  return "1";
};

IPluginV2Ext* OcrSpecialSlice::clone() const noexcept {
  return new OcrSpecialSlice(*this);
};

void OcrSpecialSlice::setPluginNamespace(const char* libNamespace) noexcept {
  mNameSpace = libNamespace;
};

const char* OcrSpecialSlice::getPluginNamespace() const noexcept {
  return mNameSpace.c_str();
}

size_t OcrSpecialSlice::getSerializationSize() const noexcept {
  return sizeof(int);
};

void OcrSpecialSlice::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mBboxesCnt);
  ASSERT(d == a + getSerializationSize());
};

OcrSpecialSlice::OcrSpecialSlice(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  mBboxesCnt = read<int>(d);
  assert(d == a + length);
};

OcrSpecialSlice::OcrSpecialSlice() {

};

int OcrSpecialSlice::initialize() noexcept {
  return 0;
};

int OcrSpecialSlice::getNbOutputs() const noexcept {
  return 1;
};

void OcrSpecialSlice::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) {

  assert(nbInputDims == 1);
  // detections: [N, anchors, (x1, y1, x2, y2, class_id, score, cos, sin)]
  assert(inputs[0].nbDims == 2 && inputs[0].d[1] == 8);
}

Dims OcrSpecialSlice::getOutputDimensions(int index, const Dims* inputDims, int nbInputs) noexcept {

  assert(index == 0);
  assert(nbInputs == 1);
  check_valid_inputs(inputDims, nbInputs);

  nvinfer1::Dims output;
  output.nbDims = inputDims[0].nbDims;
  // number of anchors
  output.d[0] = inputDims[0].d[0];
  //(x1, y1, x2, y2)
  output.d[1] = 4;

  return output;
};

int OcrSpecialSlice::enqueue(
  int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

  specialOcrSlice(stream, batch_size, mBboxesCnt, inputs[0], outputs[0]);

  return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType OcrSpecialSlice::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Only 1 input and 1 output from the plugin layer
  ASSERT(index == 0);

  // Only DataType::kFLOAT is acceptable by the plugin layer
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool OcrSpecialSlice::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool OcrSpecialSlice::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
  return false;
}

// Configure the layer with input and output data types.
void OcrSpecialSlice::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                      const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                      const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept {
  assert(nbInputs == 1);

  assert(nbOutputs == 1);

  mBboxesCnt = inputDims[0].d[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void OcrSpecialSlice::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {
}

// Detach the plugin object from its execution context.
void OcrSpecialSlice::detachFromContext() noexcept {}
