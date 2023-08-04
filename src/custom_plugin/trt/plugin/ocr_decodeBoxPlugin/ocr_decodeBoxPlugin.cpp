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
#include "ocr_decodeBoxPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::OcrDecodeBoxLayer;
using nvinfer1::plugin::OcrDecodeBoxLayerPluginCreator;

namespace {
const char* DETECTIONLAYER_PLUGIN_VERSION{"1"};
const char* DETECTIONLAYER_PLUGIN_NAME{"OcrDecodeBoxLayer_TRT"};
} // namespace

PluginFieldCollection OcrDecodeBoxLayerPluginCreator::mFC{};
std::vector<PluginField> OcrDecodeBoxLayerPluginCreator::mPluginAttributes;

OcrDecodeBoxLayerPluginCreator::OcrDecodeBoxLayerPluginCreator() {
  mPluginAttributes.emplace_back(PluginField("cascade_stage", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("max_side", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* OcrDecodeBoxLayerPluginCreator::getPluginName() const noexcept {
  return DETECTIONLAYER_PLUGIN_NAME;
};

const char* OcrDecodeBoxLayerPluginCreator::getPluginVersion() const noexcept {
  return DETECTIONLAYER_PLUGIN_VERSION;
};

const PluginFieldCollection* OcrDecodeBoxLayerPluginCreator::getFieldNames() noexcept {
  return &mFC;
};

IPluginV2Ext* OcrDecodeBoxLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "cascade_stage")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mCascadeStage = *(static_cast<const int*>(fields[i].data));
      std::cout << "createPlugin OcrDecodeBoxLayer_TRT cascade_stage " << mCascadeStage << "\n";
    }
    if (!strcmp(attrName, "max_side")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mMaxSide = *(static_cast<const int*>(fields[i].data));
      std::cout << "createPlugin OcrDecodeBoxLayer_TRT max_side " << mMaxSide << "\n";
    }
  }
  return new OcrDecodeBoxLayer(mCascadeStage, mMaxSide);
};

IPluginV2Ext* OcrDecodeBoxLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept {
  return new OcrDecodeBoxLayer(data, length);
};

OcrDecodeBoxLayer::OcrDecodeBoxLayer(int cascade_stage, int max_side)
  : mCascadeStage(cascade_stage)
  , mMaxSide(max_side) {
  mType = DataType::kFLOAT;
};

int OcrDecodeBoxLayer::getNbOutputs() const noexcept {
  return 1;
};

int OcrDecodeBoxLayer::initialize() noexcept {
  //@Init the mValidCnt and mDecodedBboxes for max batch size

  return 0;
};

void OcrDecodeBoxLayer::terminate() noexcept {};

void OcrDecodeBoxLayer::destroy() noexcept {
  delete this;
};

bool OcrDecodeBoxLayer::supportsFormat(DataType type, PluginFormat format) const noexcept {
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
};

const char* OcrDecodeBoxLayer::getPluginType() const noexcept {
  return "OcrDecodeBoxLayer_TRT";
};

const char* OcrDecodeBoxLayer::getPluginVersion() const noexcept {
  return "1";
};

IPluginV2Ext* OcrDecodeBoxLayer::clone() const noexcept {
  return new OcrDecodeBoxLayer(*this);
};

void OcrDecodeBoxLayer::setPluginNamespace(const char* libNamespace) noexcept {
  mNameSpace = libNamespace;
};

const char* OcrDecodeBoxLayer::getPluginNamespace() const noexcept {
  return mNameSpace.c_str();
}

size_t OcrDecodeBoxLayer::getSerializationSize() const noexcept {
  return sizeof(int) * 3 + sizeof(int);
};

void OcrDecodeBoxLayer::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mCascadeStage);
  write(d, mMaxBatchSize);
  write(d, mAnchorsCnt);
  write(d, mMaxSide);
  ASSERT(d == a + getSerializationSize());
};

OcrDecodeBoxLayer::OcrDecodeBoxLayer(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  int cascade_stage = read<int>(d);
  mMaxBatchSize = read<int>(d);
  mAnchorsCnt = read<int>(d);
  mMaxSide = read<int>(d);
  ASSERT(d == a + length);

  mCascadeStage = cascade_stage;
  mType = DataType::kFLOAT;
};

void OcrDecodeBoxLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) {
  // rpn_rois[N, anchors, 4]
  // classifier_delta_bbox[N, anchors, 4, 1, 1]
  assert(nbInputDims == 2);
  // roi
  assert(inputs[0].nbDims == 2 && inputs[0].d[1] == 4);
  // delta_bbox
  assert(inputs[1].nbDims == 4 && inputs[1].d[1] == 4);

};

size_t OcrDecodeBoxLayer::getWorkspaceSize(int batch_size) const noexcept {
  return 0;
};

Dims OcrDecodeBoxLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept {

  check_valid_inputs(inputs, nbInputDims);
  assert(index == 0);

  // [N, anchors, (x1, y1, x2, y2)]
  nvinfer1::Dims detections;

  detections.nbDims = 2;
  // number of anchors
  detections.d[0] = inputs[0].d[0];
  detections.d[1] = 4;

  return detections;
}

int OcrDecodeBoxLayer::enqueue(
  int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

  void* detections = outputs[0];

  // refine detection
  cudaError_t status = OcrDecodeBox(stream, batch_size, mAnchorsCnt, mMaxSide, mCascadeStage,
                                    DataType::kFLOAT, // mType,
                                    inputs[1],       // inputs[InDelta],
                                    inputs[0],       // inputs[ROI]
                                    detections);

  assert(status == cudaSuccess);
  return status;
};

DataType OcrDecodeBoxLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Only DataType::kFLOAT is acceptable by the plugin layer
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool OcrDecodeBoxLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool OcrDecodeBoxLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
  return false;
}

// Configure the layer with input and output data types.
void OcrDecodeBoxLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept {
  check_valid_inputs(inputDims, nbInputs);
  assert(inputDims[0].d[0] == inputDims[1].d[0]);

  mAnchorsCnt = inputDims[0].d[0];
  mType = inputTypes[0];
  mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void OcrDecodeBoxLayer::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {
}

// Detach the plugin object from its execution context.
void OcrDecodeBoxLayer::detachFromContext() noexcept {}
