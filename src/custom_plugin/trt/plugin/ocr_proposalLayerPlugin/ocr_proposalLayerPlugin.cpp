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
#include "ocr_proposalLayerPlugin.h"
#include "mrcnn_config.h"
#include "plugin.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <cmath>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::OcrProposalLayer;
using nvinfer1::plugin::OcrProposalLayerPluginCreator;

namespace {
const char* PROPOSALLAYER_PLUGIN_VERSION{"1"};
const char* PROPOSALLAYER_PLUGIN_NAME{"OcrProposalLayer_TRT"};
} // namespace

PluginFieldCollection OcrProposalLayerPluginCreator::mFC{};
std::vector<PluginField> OcrProposalLayerPluginCreator::mPluginAttributes;

OcrProposalLayerPluginCreator::OcrProposalLayerPluginCreator() {

  mPluginAttributes.emplace_back(PluginField("prenms_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("max_side", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* OcrProposalLayerPluginCreator::getPluginName() const noexcept {
  return PROPOSALLAYER_PLUGIN_NAME;
};

const char* OcrProposalLayerPluginCreator::getPluginVersion() const noexcept {
  return PROPOSALLAYER_PLUGIN_VERSION;
};

const PluginFieldCollection* OcrProposalLayerPluginCreator::getFieldNames() noexcept {
  return &mFC;
};

IPluginV2Ext* OcrProposalLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "prenms_topk")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mPreNMSTopK = *(static_cast<const int*>(fields[i].data));
      std::cout << "createPlugin " << i << " " << fields[i].name << " " << mPreNMSTopK << "\n";
    }
    if (!strcmp(attrName, "keep_topk")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mKeepTopK = *(static_cast<const int*>(fields[i].data));
      std::cout << "createPlugin " << i << " " << fields[i].name << " " << mKeepTopK << "\n";
    }
    if (!strcmp(attrName, "iou_threshold")) {
      assert(fields[i].type == PluginFieldType::kFLOAT32);
      mIOUThreshold = *(static_cast<const float*>(fields[i].data));
      std::cout << "createPlugin " << i << " " << fields[i].name << " " << mIOUThreshold << "\n";
    }
    if (!strcmp(attrName, "max_side")) {
      assert(fields[i].type == PluginFieldType::kINT32);
      mMaxSide = *(static_cast<const int*>(fields[i].data));
      std::cout << "createPlugin " << i << " " << fields[i].name << " " << mMaxSide << "\n";
    }
  }
  return new OcrProposalLayer(mPreNMSTopK, mKeepTopK, mIOUThreshold, mMaxSide);
};

IPluginV2Ext* OcrProposalLayerPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept {
  return new OcrProposalLayer(data, length);
};

OcrProposalLayer::OcrProposalLayer(int prenms_topk, int keep_topk, float iou_threshold, int max_side)
  : mPreNMSTopK(prenms_topk)
  , mKeepTopK(keep_topk)
  , mIOUThreshold(iou_threshold)
  , mMaxSide(max_side) {
  mBackgroundLabel = -1;
  assert(mPreNMSTopK > 0);
  assert(mKeepTopK > 0);
  assert(iou_threshold > 0.0f);

  mParam.backgroundLabelId = -1;
  mParam.numClasses = 1;
  mParam.keepTopK = mKeepTopK;
  mParam.scoreThreshold = -1 * INFINITY;
  mParam.iouThreshold = mIOUThreshold;

  mType = DataType::kFLOAT;
  if (mMaxSide == 2560) {
    mSegments = 24;
  } else if (mMaxSide == 2048) {
    mSegments = 22;
  } else if (mMaxSide == 1600) {
    mSegments = 15;
  } else if (mMaxSide == 1056) {
    mSegments = 9;
  } else if (mMaxSide == 768) {
    mSegments = 8;
  } else {
    mSegments = 1;
  }

  generate_pyramid_anchors();
};

int OcrProposalLayer::getNbOutputs() const noexcept {
  return 1;
};

int OcrProposalLayer::initialize() noexcept {
  // Init the mValidCnt of max batch size
  std::vector<int> tempValidCnt(mMaxBatchSize, mPreNMSTopK);

  mValidCnt = std::make_shared<CudaBind<int>>(mMaxBatchSize);

  CUASSERT(cudaMemcpy(
             mValidCnt->mPtr, static_cast<void*>(tempValidCnt.data()), sizeof(int) * mMaxBatchSize, cudaMemcpyHostToDevice));

  // Init the anchors for batch size:
  mAnchorBoxesDevice = std::make_shared<CudaBind<float>>(mAnchorsCnt * 4 * mMaxBatchSize);
  int batch_offset = sizeof(float) * mAnchorsCnt * 4;
  uint8_t* device_ptr = static_cast<uint8_t*>(mAnchorBoxesDevice->mPtr);
  for (int i = 0; i < mMaxBatchSize; i++) {
    CUASSERT(cudaMemcpy(static_cast<void*>(device_ptr + i * batch_offset),
                        static_cast<void*>(mAnchorBoxesHost.data()), batch_offset, cudaMemcpyHostToDevice));
  }

  return 0;
};

void OcrProposalLayer::terminate() noexcept {};

void OcrProposalLayer::destroy() noexcept {
  delete this;
};

bool OcrProposalLayer::supportsFormat(DataType type, PluginFormat format) const noexcept {
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
};

const char* OcrProposalLayer::getPluginType() const noexcept {
  return "OcrProposalLayer_TRT";
};

const char* OcrProposalLayer::getPluginVersion() const noexcept {
  return "1";
};

IPluginV2Ext* OcrProposalLayer::clone() const noexcept {
  return new OcrProposalLayer(*this);
};

void OcrProposalLayer::setPluginNamespace(const char* libNamespace) noexcept {
  mNameSpace = libNamespace;
};

const char* OcrProposalLayer::getPluginNamespace() const noexcept {
  return mNameSpace.c_str();
};

size_t OcrProposalLayer::getSerializationSize() const noexcept {
  return sizeof(int) * 2 + sizeof(float) + sizeof(int) * 2 + sizeof(int);
};

void OcrProposalLayer::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, mPreNMSTopK);
  write(d, mKeepTopK);
  write(d, mIOUThreshold);
  write(d, mMaxBatchSize);
  write(d, mAnchorsCnt);
  write(d, mMaxSide);
  ASSERT(d == a + getSerializationSize());
};

OcrProposalLayer::OcrProposalLayer(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  int prenms_topk = read<int>(d);
  int keep_topk = read<int>(d);
  float iou_threshold = read<float>(d);
  mMaxBatchSize = read<int>(d);
  mAnchorsCnt = read<int>(d);
  mMaxSide = read<int>(d);
  ASSERT(d == a + length);

  mBackgroundLabel = -1;
  mPreNMSTopK = prenms_topk;
  mKeepTopK = keep_topk;
  mScoreThreshold = 0.0;
  mIOUThreshold = iou_threshold;

  mParam.backgroundLabelId = -1;
  mParam.numClasses = 1;
  mParam.keepTopK = mKeepTopK;
  mParam.scoreThreshold = 0.0;
  mParam.iouThreshold = mIOUThreshold;

  mType = DataType::kFLOAT;
  if (mMaxSide == 2560) {
    mSegments = 24;
  } else if (mMaxSide == 2048) {
    mSegments = 22;
  } else if (mMaxSide == 1600) {
    mSegments = 15;
  } else if (mMaxSide == 1056) {
    mSegments = 9;
  } else if (mMaxSide == 768) {
    mSegments = 8;
  } else {
    mSegments = 1;
  }

  generate_pyramid_anchors();
};

void OcrProposalLayer::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims) {
  // object_score[N, anchors, 1, 1],
  // foreground_delta[N, anchors, 4, 1],
  // anchors should be generated inside
  assert(nbInputDims == 2);
  // foreground_score
  assert(inputs[0].nbDims == 3 && inputs[0].d[1] == 1);
  // foreground_delta
  assert(inputs[1].nbDims == 3 && inputs[1].d[1] == 4);
};

size_t OcrProposalLayer::getWorkspaceSize(int mMaxBatchSize) const noexcept {
  OcrProposalWorkSpace proposal(mMaxBatchSize, mAnchorsCnt, mPreNMSTopK, mParam, mSegments, mType);
  std::cout << "proposal.totalSize:" << proposal.totalSize << std::endl;
  return proposal.totalSize;
};

Dims OcrProposalLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept {

  check_valid_inputs(inputs, nbInputDims);
  assert(index == 0);

  // [N, anchors, (x1, y1, x2, y2)]
  nvinfer1::Dims proposals;

  proposals.nbDims = 2;
  // number of keeping anchors
  proposals.d[0] = mKeepTopK;
  proposals.d[1] = 4;

  return proposals;
}

void OcrProposalLayer::generate_pyramid_anchors() {
  const auto image_dims = nvinfer1::Dims3(3, mMaxSide, mMaxSide);
  const auto& scales = MaskRCNNConfig::RPN_ANCHOR_SCALES;
  const auto& ratios = MaskRCNNConfig::RPN_ANCHOR_RATIOS;
  const auto& strides = MaskRCNNConfig::BACKBONE_STRIDES;
  auto anchor_stride = MaskRCNNConfig::RPN_ANCHOR_STRIDE;

  auto& anchors = mAnchorBoxesHost;
  assert(anchors.size() == 0);

  assert(scales.size() == strides.size());
  for (size_t s = 0; s < scales.size(); ++s) {
    float scale = scales[s];
    float stride = strides[s];

    for (float y = (stride - 1) / 2; y < image_dims.d[1]; y += anchor_stride * stride)
      for (float x = (stride - 1) / 2; x < image_dims.d[2]; x += anchor_stride * stride)
        for (float r : ratios) {
          // float sqrt_r = sqrt(r);
          // float h = scale / sqrt_r;
          // float w = scale * sqrt_r;

          float size_ratio = stride * stride / r;
          float w = round(sqrt(size_ratio));
          float h = round(w * r);
          w = w * (scale / stride) - 1;
          h = h * (scale / stride) - 1;

          anchors.insert(anchors.end(),
          {(x - w / 2), (y - h / 2), (x + w / 2) + 1, (y + h / 2) + 1});

          // if ((s == 4)){
          //     std::cout<< (x - w / 2) << " " << (y - h / 2) << " " << (x + w / 2) + 1 << " " << (y + h / 2) + 1 <<std::endl;
          // }
        }
  }

  assert(anchors.size() % 4 == 0);
}

int OcrProposalLayer::enqueue(
  int batch_size, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {

  void* proposals = outputs[0];

  // proposal
  OcrProposalWorkSpace proposalWorkspace(batch_size, mAnchorsCnt, mPreNMSTopK, mParam, mSegments, mType);
  cudaError_t status = OcrProposalRefineBatchClassNMS(stream, batch_size, mAnchorsCnt, mPreNMSTopK, mSegments,
                       mMaxSide,
                       DataType::kFLOAT, // mType,
                       mParam, proposalWorkspace, workspace,
                       inputs[0], // inputs[object_score]
                       inputs[1], // inputs[bbox_delta],
                       mValidCnt->mPtr,
                       mAnchorBoxesDevice->mPtr, // inputs[anchors]
                       proposals);

  assert(status == cudaSuccess);
  return status;
};

// Return the DataType of the plugin output at the requested index
DataType OcrProposalLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Only DataType::kFLOAT is acceptable by the plugin layer
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool OcrProposalLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool OcrProposalLayer::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
  return false;
}

// Configure the layer with input and output data types.
void OcrProposalLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                       const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                       const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept {
  check_valid_inputs(inputDims, nbInputs);
  assert(inputDims[0].d[0] == inputDims[1].d[0]);

  mAnchorsCnt = inputDims[0].d[0];
  assert(mAnchorsCnt == (int) (mAnchorBoxesHost.size() / 4));
  mMaxBatchSize = maxBatchSize;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void OcrProposalLayer::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept {
}

// Detach the plugin object from its execution context.
void OcrProposalLayer::detachFromContext() noexcept {}
