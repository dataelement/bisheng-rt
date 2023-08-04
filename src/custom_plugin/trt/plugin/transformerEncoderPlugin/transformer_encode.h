/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRT_TRANSFORMER_ENCODE_PLUGIN_H
#define TRT_TRANSFORMER_ENCODE_PLUGIN_H

#include "NvInferPlugin.h"
#include "common_strcture.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

namespace nvinfer1 {
namespace plugin {

template <typename T>
class TransformerEncodeNode {
 public:
  TransformerEncodeNode(const void* const* inputs, void* const* outputs);

  TransformerEncodeNode() = default;

 public:
  const T *encode_input;
  const T *att_mask;
  const T *ffn_mask;

  DenseWeight<T> self_conv1d0;
  DenseWeight<T> self_conv1d1;
  AttentionWeight<T> self_attention;
  LayerNormWeight<T> self_layernorm0;
  LayerNormWeight<T> self_layernorm1;
  LayerNormWeight<T> self_layernorm2;
  FFNWeight<T> ffn;
  LayerNormWeight<T> ffn_layernorm;
  LayerNormWeight<T> layernorm_final;

  T *encode_out;
};

template <typename T>
class TransformerEncodeTemp {
 public:
  TransformerEncodeTemp(const int B, const int L, const int numHeads, const int headSize, void* workspace);

  TransformerEncodeTemp() = default;

 public:
  T* im2col_buf0;
  T* conv1d_buf0;
  T* layernorm_buf0;
  T* im2col_buf1;
  T* conv1d_buf1;
  T* layernorm_buf1;
  T* add_buf0;
  T* layernorm_buf2;
  T* att_in_buf0;
  T* att_in_buf1;
  T* att_out_buf;
  T* att_matmul_buf;
  T* add_buf1;
  T* ffn_layernorm_buf;
  T* ffn_inter_buf;
  T* ffn_out_buf;
  T* add_buf2;
  T* layernorm_final_buf;
  T* q_buf0;
  T* k_buf0;
  T* v_buf0;
  T* q_buf1;
  T* k_buf1;
  T* v_buf1;
  T* qk_buf0;
  T* transpose_dst;
};

class TransformerEncodePluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  TransformerEncodePluginDynamic(const std::string name, const nvinfer1::DataType type,
                                 const int hiddenSize, const int numHeads, const bool isLastLayer);

  TransformerEncodePluginDynamic(const std::string name, const void* data, size_t length);

  // TransformerEncodePluginDynamic() = delete;

  ~TransformerEncodePluginDynamic() {};

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  nvinfer1::DataType mType;
  float mRsqrtHeadSize;
  int mHeadSize;
  int mHiddenSize;
  int mNumHeads;
  bool mIsLastLayer;
  const std::string mLayerName;
  std::string mNamespace;

  cublasHandle_t cublas;

 // protected:
  // using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  // using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  // using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  // using nvinfer1::IPluginV2DynamicExt::supportsFormat;
  // using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  // using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  // using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class TransformerEncodePluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  TransformerEncodePluginDynamicCreator();

  ~TransformerEncodePluginDynamicCreator() {};

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_TRANSFORMER_ENCODE_PLUGIN_H


