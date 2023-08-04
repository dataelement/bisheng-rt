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

#ifndef TRT_TRANSFORMER_DECODE_PLUGIN_H
#define TRT_TRANSFORMER_DECODE_PLUGIN_H

#include "NvInferPlugin.h"
#include "common_strcture.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

namespace nvinfer1 {
namespace plugin {

template <typename T>
class DecodeAttentionWeight {
 public:
  /* weights for masked_multi_head_attention */
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;

  LayerNormWeight<T> cross_layernorm;
  AttentionWeight<T> cross_attention;

  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;
};


template <typename T>
class TransformerDecodeNode {
 public:
  TransformerDecodeNode(const int num_Layer, const int hidden_unit, const void* const* inputs, void* const* outputs);

  TransformerDecodeNode() = default;

 public:
  /* weights for masked_multi_head_attention */
  const T *memory_tensor;
  const int *memory_sequence_length;

  std::vector<DecodeAttentionWeight<T>> multi_attention;

  LayerNormWeight<T> layernorm;
  const T *embedding_table;

  int *output_ids;
  int *parent_ids;
  int *sequence_length;
};

template <typename T>
class TransformerDecodeTemp {
 public:
  TransformerDecodeTemp(const int B, const int L, const int mNumLayer, const int mHiddenSize,
                        const int mBeamWidth, const int mVocabSize, void* workspace);

  TransformerDecodeTemp() = default;

  ~TransformerDecodeTemp() {
    delete [] h_finished_buf;
  }

 public:
  T* from_tensor_buf[2];
  T* K_cache_buf[2];
  T* V_cache_buf[2];
  T* K_mem_cache_buf[3];
  T* V_mem_cache_buf[3];
  T* decoder_buf;
  T* decoder_normed_result_buf;
  float* logits_buf;
  float* cum_log_buf;
  int* word_ids_buf;
  int* topk_ids_buf;
  bool* finished_buf;
  int* finished_count_buf;
  bool* h_finished_buf;
};

class TransformerDecodePluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  TransformerDecodePluginDynamic(const std::string name, const nvinfer1::DataType type,
                                 const int hiddenSize, const int numHeads, const int beamWidth,
                                 const int vocabSize, const int startId, const int endId, const int numLayer);

  TransformerDecodePluginDynamic(const std::string name, const void* data, size_t length);

  // TransformerDecodePluginDynamic() = delete;

  ~TransformerDecodePluginDynamic() {};

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
  int mHeadSize;
  int mHiddenSize;
  int mNumHeads;
  int mBeamWidth;
  int mVocabSize;
  int mStartId;
  int mEndId;
  int mNumLayer;
  const std::string mLayerName;
  std::string mNamespace;
  cublasHandle_t cublas;

 // protected:
 //  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
 //  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
 //  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
 //  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
 //  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
 //  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
 //  using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class TransformerDecodePluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  TransformerDecodePluginDynamicCreator();

  ~TransformerDecodePluginDynamicCreator() {};

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


