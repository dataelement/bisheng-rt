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
#include "resizeBilinearPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>

#define DEBUG 0

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::ResizeBilinear;
using nvinfer1::plugin::ResizeBilinearPluginCreator;

namespace {
const char* RESIZE_PLUGIN_VERSION{"1"};
const char* RESIZE_PLUGIN_NAME{"ResizeBilinear_TRT"};
} // namespace

PluginFieldCollection ResizeBilinearPluginCreator::mFC{};
std::vector<PluginField> ResizeBilinearPluginCreator::mPluginAttributes;

ResizeBilinearPluginCreator::ResizeBilinearPluginCreator() {
  mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* ResizeBilinearPluginCreator::getPluginName() const noexcept {
  return RESIZE_PLUGIN_NAME;
};

const char* ResizeBilinearPluginCreator::getPluginVersion() const noexcept {
  return RESIZE_PLUGIN_VERSION;
};

const PluginFieldCollection* ResizeBilinearPluginCreator::getFieldNames() noexcept {
  return &mFC;
};

IPluginV2Ext* ResizeBilinearPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "scale")) {
      assert(fields[i].type == PluginFieldType::kFLOAT32);
      mScale = *(static_cast<const float*>(fields[i].data));
    }
  }
  return new ResizeBilinear(mScale);
};

IPluginV2Ext* ResizeBilinearPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept {
  return new ResizeBilinear(data, length);
};

ResizeBilinear::ResizeBilinear(float scale)
  : mScale(scale) {
  assert(mScale > 0);
};

int ResizeBilinear::getNbOutputs() const noexcept {
  return 1;
};

Dims ResizeBilinear::getOutputDimensions(int index, const Dims* inputDims, int nbInputs) noexcept {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  for (int d = 0; d < input.nbDims; ++d) {
    if (d == input.nbDims - 2 || d == input.nbDims - 1) {
      output.d[d] = int(input.d[d] * mScale);
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
};

int ResizeBilinear::initialize() noexcept {
  return 0;
};

void ResizeBilinear::terminate() noexcept {

};

void ResizeBilinear::destroy() noexcept {

};

size_t ResizeBilinear::getWorkspaceSize(int) const noexcept {
  return 0;
}

size_t ResizeBilinear::getSerializationSize() const noexcept {
  // scale, dimensions: 3 * 2
  return sizeof(float) + sizeof(int) * 3 * 2;
};

void ResizeBilinear::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mScale);
  write(d, mInputDims.d[0]);
  write(d, mInputDims.d[1]);
  write(d, mInputDims.d[2]);
  write(d, mOutputDims.d[0]);
  write(d, mOutputDims.d[1]);
  write(d, mOutputDims.d[2]);
  ASSERT(d == a + getSerializationSize());
};

ResizeBilinear::ResizeBilinear(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  mScale = read<float>(d);
  mInputDims = Dims3();
  mInputDims.d[0] = read<int>(d);
  mInputDims.d[1] = read<int>(d);
  mInputDims.d[2] = read<int>(d);
  mOutputDims = Dims3();
  mOutputDims.d[0] = read<int>(d);
  mOutputDims.d[1] = read<int>(d);
  mOutputDims.d[2] = read<int>(d);
  ASSERT(d == a + length);
};

const char* ResizeBilinear::getPluginType() const noexcept {
  return "ResizeBilinear_TRT";
};

const char* ResizeBilinear::getPluginVersion() const noexcept {
  return "1";
};

IPluginV2Ext* ResizeBilinear::clone() const noexcept {
  return new ResizeBilinear(*this);
};

void ResizeBilinear::setPluginNamespace(const char* libNamespace) noexcept {
  mNameSpace = libNamespace;
};

const char* ResizeBilinear::getPluginNamespace() const noexcept {
  return mNameSpace.c_str();
}

bool ResizeBilinear::supportsFormat(DataType type, PluginFormat format) const noexcept {
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
};

int ResizeBilinear::enqueue(
  int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

  int nchan = mOutputDims.d[0];
  float scale = mScale;
  int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
  int2 isize = {mInputDims.d[2], mInputDims.d[1]};
  int istride = mInputDims.d[2];
  int ostride = mOutputDims.d[2];
  int ibatchstride = mInputDims.d[1] * istride;
  int obatchstride = mOutputDims.d[1] * ostride;
  dim3 block(32, 32);
  dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));

  resizeBilinear(grid, block, stream, batch_size * nchan, scale, osize, isize, static_cast<float const*>(inputs[0]), istride,
                 ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);

  return cudaGetLastError() != cudaSuccess;
};

// Return the DataType of the plugin output at the requested index
DataType ResizeBilinear::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Only 1 input and 1 output from the plugin layer
  ASSERT(index == 0);

  // Only DataType::kFLOAT is acceptable by the plugin layer
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool ResizeBilinear::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool ResizeBilinear::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
  return false;
}

// Configure the layer with input and output data types.
void ResizeBilinear::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept {
  assert(nbInputs == 1);
  mInputDims = inputDims[0];

  assert(nbOutputs == 1);
  mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ResizeBilinear::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {
}

// Detach the plugin object from its execution context.
void ResizeBilinear::detachFromContext() noexcept {}
