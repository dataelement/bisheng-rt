/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * BERT Encoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/ocr_open_attention.h"
#include "fastertransformer/common_structure.h"

namespace fastertransformer
{
template <typename T>
class EncoderInitParamOCR
{
public:
  const T *inputs;

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

  T *transformer_out;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_, template <OperationType> class MultiHeadAttention_>
class OcrEncoderTransformerTraits;

template <template <OperationType> class MultiHeadAttention_>
class OcrEncoderTransformerTraits<OperationType::FP32, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP32>
{
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <template <OperationType> class MultiHeadAttention_>
class OcrEncoderTransformerTraits<OperationType::FP16, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP16>
{
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <class Traits_>
class OcrEncoderTransformer
{
  const IAllocator &allocator_;
  typename Traits_::MultiHeadAttention *attention_;
  typedef typename Traits_::DataType DataType_;
  EncoderInitParamOCR<DataType_> param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[5];

  DataType_ *buf_;

  DataType_ *im2col_buf0_;
  DataType_ *conv1d_buf0_;
  DataType_ *layernorm_buf0_;

  DataType_ *im2col_buf1_;
  DataType_ *conv1d_buf1_;
  DataType_ *layernorm_buf1_;

  DataType_ *add_buf0_;
  DataType_ *layernorm_buf2_;

  DataType_ *att_in_buf0_;
  DataType_ *att_in_buf1_;
  DataType_ *att_out_buf_;
  DataType_ *att_matmul_buf_;

  DataType_ *add_buf1_;
  DataType_ *ffn_layernorm_buf_;
  DataType_ *ffn_inter_buf_;
  DataType_ *ffn_out_buf_;

  DataType_ *add_buf2_;
  DataType_ *layernorm_final_buf_;

  int batch_size_;
  int from_seq_len_;
  int head_num_;
  int size_per_head_;
  bool is_last_layer_;
  int buf_size_;

public:
  OcrEncoderTransformer(const IAllocator &allocator, int batch_size, int from_seq_len,
                        int head_num, int size_per_head, bool is_last_layer)
    : allocator_(allocator),
    batch_size_(batch_size),
    from_seq_len_(from_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    is_last_layer_(is_last_layer)
  {
    try
    {
      int m = batch_size_ * from_seq_len_;
      int k = head_num_ * size_per_head_;
      int n = k;

      buf_size_ = m * n;

      buf_ = reinterpret_cast<DataType_ *>(allocator_.malloc(sizeof(DataType_) * buf_size_ * 23));
      if (buf_ == nullptr)
        throw std::runtime_error(std::string("Tensorflow Allocator failed to allocate internal buffer."));

      im2col_buf0_ = buf_;
      conv1d_buf0_ = im2col_buf0_ + 3 * buf_size_;
      layernorm_buf0_ = conv1d_buf0_ + buf_size_;

      im2col_buf1_ = layernorm_buf0_ + buf_size_;
      conv1d_buf1_ = im2col_buf1_ + 3 * buf_size_;
      layernorm_buf1_ = conv1d_buf1_ + buf_size_;

      add_buf0_ = layernorm_buf1_ + buf_size_;
      layernorm_buf2_ = add_buf0_ + buf_size_;

      att_in_buf0_ = layernorm_buf2_ + buf_size_;
      att_in_buf1_ = att_in_buf0_ + buf_size_;
      att_out_buf_ = att_in_buf1_ + buf_size_;
      att_matmul_buf_ = att_out_buf_ + buf_size_;
      add_buf1_ = att_matmul_buf_ + buf_size_;

      ffn_layernorm_buf_ = add_buf1_ + buf_size_;
      ffn_inter_buf_ = ffn_layernorm_buf_ + buf_size_;
      ffn_out_buf_ = ffn_inter_buf_ + 2 * buf_size_;
      add_buf2_ = ffn_out_buf_ + buf_size_;
      layernorm_final_buf_ = add_buf2_ + buf_size_;

      attention_ = new typename Traits_::MultiHeadAttention(allocator_, batch_size_, from_seq_len_, from_seq_len_, head_num_, size_per_head_);

      if (Traits_::OpType == OperationType::FP32)
      {
        cublasAlgo_[0] = -1;
        cublasAlgo_[1] = -1;
        cublasAlgo_[2] = -1;
        cublasAlgo_[3] = -1;
        cublasAlgo_[4] = -1;
      }
      else
      {
        cublasAlgo_[0] = 99;
        cublasAlgo_[1] = 99;
        cublasAlgo_[2] = 99;
        cublasAlgo_[3] = 99;
        cublasAlgo_[4] = 99;
      }
      // std::cout<<cublasAlgo_[0]<<cublasAlgo_[1]<<cublasAlgo_[2]<<cublasAlgo_[3]<<std::endl;
    }

    catch (std::runtime_error &error)
    {
      throw error;
    }
  }
  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
  void initialize(EncoderInitParamOCR<DataType_> param)
  {
    param_ = param;
    cuda::MultiHeadInitParam<DataType_> multi_head_init_param;

    multi_head_init_param.from_tensor = att_in_buf0_;
    multi_head_init_param.to_tensor = att_in_buf1_;
    multi_head_init_param.self_attention = param.self_attention;
    multi_head_init_param.attr_mask = param.att_mask;
    multi_head_init_param.stream = param.stream;
    multi_head_init_param.cublas_handle = param.cublas_handle;
    multi_head_init_param.attr_out = att_out_buf_;

    attention_->initialize(multi_head_init_param);
  }

  /**
   * do forward
   **/
  void forward()
  {
    try
    {
      int m = batch_size_ * from_seq_len_;
      int k = head_num_ * size_per_head_;
      int n = k;

      im2col1d_gpu(param_.inputs, im2col_buf0_, k, m, from_seq_len_, 3, 1, param_.stream);

      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;
      k = head_num_ * size_per_head_ * 3;

      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.self_conv1d0.kernel,
                                    AType_, n,
                                    im2col_buf0_, BType_, k,
                                    &beta,
                                    conv1d_buf0_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      k = head_num_ * size_per_head_;
      n = k;
      add_bias_kernelLauncher<DataType_>(conv1d_buf0_, param_.self_conv1d0.bias, m, n, true, param_.stream);

      // // test output
      // check_cuda_error(cudaMemcpy(param_.transformer_out, conv1d_buf0_, sizeof(DataType_)*buf_size_, cudaMemcpyDeviceToDevice));

      layernorm_kernelLauncher(layernorm_buf0_, conv1d_buf0_, param_.self_layernorm0.gamma, param_.self_layernorm0.beta, m, n, param_.stream);

      im2col1d_gpu(layernorm_buf0_, im2col_buf1_, k, m, from_seq_len_, 3, 1, param_.stream);

      k = head_num_ * size_per_head_ * 3;
      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.self_conv1d1.kernel, AType_, n,
                                    im2col_buf1_, BType_, k,
                                    &beta,
                                    conv1d_buf1_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

      k = head_num_ * size_per_head_;
      n = k;
      add_bias_kernelLauncher<DataType_>(conv1d_buf1_, param_.self_conv1d1.bias, m, n, true, param_.stream);

      layernorm_kernelLauncher(layernorm_buf1_, conv1d_buf1_, param_.self_layernorm1.gamma, param_.self_layernorm1.beta, m, n, param_.stream);

      add_op(add_buf0_, layernorm_buf1_, param_.inputs, buf_size_, param_.stream);

      layernorm_kernelLauncher(layernorm_buf2_, add_buf0_, param_.self_layernorm2.gamma, param_.self_layernorm2.beta, m, n, param_.stream);

      check_cuda_error(cudaMemcpy(att_in_buf0_, layernorm_buf2_, sizeof(DataType_)*buf_size_, cudaMemcpyDeviceToDevice));
      check_cuda_error(cudaMemcpy(att_in_buf1_, layernorm_buf2_, sizeof(DataType_)*buf_size_, cudaMemcpyDeviceToDevice));

      attention_->forward();

      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.self_attention.attention_output_weight.kernel, AType_, n,
                                    att_out_buf_, BType_, k,
                                    &beta,
                                    att_matmul_buf_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

      add_op(add_buf1_, att_matmul_buf_, add_buf0_, buf_size_, param_.stream);

      layernorm_kernelLauncher(ffn_layernorm_buf_, add_buf1_, param_.ffn_layernorm.gamma, param_.ffn_layernorm.beta, m, n, param_.stream);

      n *= 2;
      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.ffn.intermediate_weight.kernel, AType_, n,
                                    ffn_layernorm_buf_, BType_, k,
                                    &beta,
                                    ffn_inter_buf_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));

      add_bias_kernelLauncher<DataType_>(ffn_inter_buf_, param_.ffn.intermediate_weight.bias, m, n, true, param_.stream);

      n = k;
      k *= 2;
      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.ffn.output_weight.kernel, AType_, n,
                                    ffn_inter_buf_, BType_, k,
                                    &beta,
                                    ffn_out_buf_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[4])));
      add_bias_kernelLauncher<DataType_>(ffn_out_buf_, param_.ffn.output_weight.bias, m, n, false, param_.stream);

      if(is_last_layer_)
      {
          add_mask_op(add_buf2_, add_buf1_, ffn_out_buf_, param_.ffn_mask, buf_size_, param_.stream);
          layernorm_kernelLauncher(layernorm_final_buf_, add_buf2_, param_.layernorm_final.gamma, param_.layernorm_final.beta, m, n, param_.stream);
          check_cuda_error(cudaMemcpy(param_.transformer_out, layernorm_final_buf_, sizeof(DataType_)*buf_size_, cudaMemcpyDeviceToDevice));
      }
      else
      {
          add_mask_op(param_.transformer_out, add_buf1_, ffn_out_buf_, param_.ffn_mask, buf_size_, param_.stream);
      }
    }

    catch (std::runtime_error &error)
    {
      throw error;
    }
  }

  ~OcrEncoderTransformer()
  {
    delete attention_;
    allocator_.free(buf_);
  }
};

} // namespace fastertransformer
