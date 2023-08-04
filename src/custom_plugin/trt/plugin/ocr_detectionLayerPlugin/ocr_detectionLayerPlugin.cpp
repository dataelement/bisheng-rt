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
#include "ocr_detectionLayerPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::OcrDetectionLayer;
using nvinfer1::plugin::OcrDetectionLayerPluginCreator;

namespace {
const char* DETECTIONLAYER_PLUGIN_VERSION{"1"};
const char* DETECTIONLAYER_PLUGIN_NAME{"OcrDetectionLayer_TRT"};
} // namespace

PluginFieldCollection OcrDetectionLayerPluginCreator::mFC{};
std::vector<PluginField> OcrDetectionLayerPluginCreator::mPluginAttributes;

OcrDetectionLayerPluginCreator::OcrDetectionLayerPluginCreator() {

  mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("max_side", nullptr, PluginFieldType::kINT32, 1));


  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* OcrDetectionLayerPluginCreator::getPluginName() const noexcept {
  return DETECTIONLAYER_PLUGIN_NAME;
};

const char* OcrDetectionLayerPluginCreator::getPluginVersion() const noexcept {
  return DETECTIONLAYER_PLUGIN_VERSION;
};

const PluginFieldCollection* OcrDetectionLayerPluginCreator::getFieldNames() noexcept {
  return &mFC;
};

IPluginV2Ext* OcrDetectionLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "num_classes")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mNbClasses = *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "keep_topk")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mKeepTopK = *(static_cast<const int*>(fields[i].data));
    }
    if (!strcmp(attrName, "score_threshold")) {
      assert(fields[i].type == PluginFieldType::kFLOAT32);
      mScoreThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "iou_threshold")) {
      assert(fields[i].type == PluginFieldType::kFLOAT32);
      mIOUThreshold = *(static_cast<const float*>(fields[i].data));
    }
    if (!strcmp(attrName, "max_side")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mMaxSide = *(static_cast<const int*>(fields[i].data));
    }
  }
  return new OcrDetectionLayer(mNbClasses, mKeepTopK, mScoreThreshold, mIOUThreshold, mMaxSide);
};

IPluginV2Ext* OcrDetectionLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept {
  return new OcrDetectionLayer(data, length);
};

OcrDetectionLayer::OcrDetectionLayer(int num_classes, int keep_topk, float score_threshold, float iou_threshold, int max_side)
  : mNbClasses(num_classes)
  , mKeepTopK(keep_topk)
  , mScoreThreshold(score_threshold)
  , mIOUThreshold(iou_threshold)
  , mMaxSide(max_side) {
  mBackgroundLabel = 0;
  assert(mNbClasses > 0);
  assert(mKeepTopK > 0);
  assert(score_threshold >= 0.0f);
  assert(iou_threshold > 0.0f);

  mParam.backgroundLabelId = 0;
  mParam.numClasses = mNbClasses;
  mParam.keepTopK = mKeepTopK;
  mParam.scoreThreshold = mScoreThreshold;
  mParam.iouThreshold = mIOUThreshold;

  mType = DataType::kFLOAT;
};

int OcrDetectionLayer::getNbOutputs() const noexcept {
  return 1;
};

int OcrDetectionLayer::initialize() noexcept {
  //@Init the mValidCnt and mDecodedBboxes for max batch size
  std::vector<int> tempValidCnt(mMaxBatchSize, mAnchorsCnt);

  mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

  CUASSERT(cudaMemcpy(
             mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

  return 0;
};

void OcrDetectionLayer::terminate() noexcept {};

void OcrDetectionLayer::destroy() noexcept {
  delete this;
};

bool OcrDetectionLayer::supportsFormat(DataType type, PluginFormat format) const noexcept {
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
};

const char* OcrDetectionLayer::getPluginType() const noexcept {
  return "OcrDetectionLayer_TRT";
};

const char* OcrDetectionLayer::getPluginVersion() const noexcept {
  return "1";
};

IPluginV2Ext* OcrDetectionLayer::clone() const noexcept {
  return new OcrDetectionLayer(*this);
};

void OcrDetectionLayer::setPluginNamespace(const char* libNamespace) noexcept {
  mNameSpace = libNamespace;
};

const char* OcrDetectionLayer::getPluginNamespace() const noexcept {
  return mNameSpace.c_str();
}

size_t OcrDetectionLayer::getSerializationSize() const noexcept {
  return sizeof(int) * 3 + sizeof(float) * 2 + sizeof(int) * 2;
};

void OcrDetectionLayer::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mNbClasses);
  write(d, mKeepTopK);
  write(d, mMaxSide);
  write(d, mScoreThreshold);
  write(d, mIOUThreshold);
  write(d, mMaxBatchSize);
  write(d, mAnchorsCnt);
  ASSERT(d == a + getSerializationSize());
};

OcrDetectionLayer::OcrDetectionLayer(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  int num_classes = read<int>(d);
  int keep_topk = read<int>(d);
  int max_size = read<int>(d);
  float score_threshold = read<float>(d);
  float iou_threshold = read<float>(d);
  mMaxBatchSize = read<int>(d);
  mAnchorsCnt = read<int>(d);
  ASSERT(d == a + length);

  mNbClasses = num_classes;
  mKeepTopK = keep_topk;
  mMaxSide = max_size;
  mScoreThreshold = score_threshold;
  mIOUThreshold = iou_threshold;

  mParam.backgroundLabelId = 0;
  mParam.numClasses = mNbClasses;
  mParam.keepTopK = mKeepTopK;
  mParam.scoreThreshold = mScoreThreshold;
  mParam.iouThreshold = mIOUThreshold;

  mType = DataType::kFLOAT;
};

void OcrDetectionLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) {
  // classifier_delta_bbox[N, anchors, 4, 1, 1]
  // classifier_class[N, anchors, num_classes, 1, 1]
  // rpn_rois[N, anchors, 4]
  // classifier_cos[N, anchors, 1, 1, 1]
  // classifier_sin[N, anchors, 1, 1, 1]
  assert(nbInputDims == 5);
  // roi
  assert(inputs[0].nbDims == 2 && inputs[0].d[1] == 4);
  // score
  assert(inputs[1].nbDims == 4 && inputs[1].d[1] == mNbClasses);
  // delta_bbox
  assert(inputs[2].nbDims == 4 && inputs[2].d[1] == 4);
  // cos
  assert(inputs[3].nbDims == 4 && inputs[3].d[1] == 1);
  // sin
  assert(inputs[4].nbDims == 4 && inputs[4].d[1] == 1);
};

size_t OcrDetectionLayer::getWorkspaceSize(int batch_size) const noexcept {
  RefineDetectionWorkSpace refine(batch_size, mAnchorsCnt, mParam, mType);
  std::cout << "refine.totalSize:" << refine.totalSize << std::endl;
  return refine.totalSize;
};

Dims OcrDetectionLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept {

  check_valid_inputs(inputs, nbInputDims);
  assert(index == 0);

  // [N, anchors, (x1, y1, x2, y2, class_id, score, cos, sin)]
  nvinfer1::Dims detections;

  detections.nbDims = 2;
  // number of anchors
  detections.d[0] = mKeepTopK;
  detections.d[1] = 8;

  return detections;
}

int OcrDetectionLayer::enqueue(
  int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

  void* detections = outputs[0];

  // refine detection
  RefineDetectionWorkSpace refDetcWorkspace(batch_size, mAnchorsCnt, mParam, mType);
  cudaError_t status = OcrRefineBatchClassNMS(stream, batch_size, mAnchorsCnt,
                       DataType::kFLOAT, // mType,
                       mMaxSide,
                       mParam, refDetcWorkspace, workspace,
                       inputs[1],       // inputs[InScore]
                       inputs[2],       // inputs[InDelta],
                       mValidCnt->mPtr, // inputs[InCountValid],
                       inputs[0],       // inputs[ROI]
                       inputs[3],       // inputs[cos]
                       inputs[4],       // inputs[sin]
                       detections);

  assert(status == cudaSuccess);
  return status;
};

DataType OcrDetectionLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Only DataType::kFLOAT is acceptable by the plugin layer
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool OcrDetectionLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool OcrDetectionLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
  return false;
}

// Configure the layer with input and output data types.
void OcrDetectionLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept {
  check_valid_inputs(inputDims, nbInputs);
  assert(inputDims[0].d[0] == inputDims[1].d[0] && inputDims[1].d[0] == inputDims[2].d[0]);

  mAnchorsCnt = inputDims[2].d[0];
  mType = inputTypes[0];
  mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void OcrDetectionLayer::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {
}

// Detach the plugin object from its execution context.
void OcrDetectionLayer::detachFromContext() noexcept {}
