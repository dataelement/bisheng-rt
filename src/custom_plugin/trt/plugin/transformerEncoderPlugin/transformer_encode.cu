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
#include <cassert>
#include <cstring>
#include <iostream>
#include "NvInfer.h"
#include "bertCommon.h"
#include "transformer_encode.h"
#include "serialize.hpp"
#include "transformerKernels.h"

using namespace transformer;
using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::TransformerEncodePluginDynamic;
using nvinfer1::plugin::TransformerEncodePluginDynamicCreator;

template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
      + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar) {
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;
  int mask_offset = batch_id * seq_len * seq_len;

  float NEG_INF = -1e9f;
  if(sizeof(T) == 2)
    NEG_INF = -5e4f;

  float NN = -1e20f;
  if(sizeof(T) == 2)
    NN = -5e4f;

  __shared__ float s_sum, s_max;

  for(int i = 0; i < seq_len; ++i) {
    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

    mask_val = (1.0f - mask_val) * NEG_INF;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): NN;

    float max_val = blockReduceMax<float>(tmp);

    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

    float sum_val = blockReduceSum<float>(qk);

    if(threadIdx.x == 0) {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

    qk_offset += seq_len;
    mask_offset += seq_len;
  }
}

template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const float scalar) {
  int batch_id = blockIdx.x / head_num / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
  float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

  float NEG_INF = -1e9f;
  if(sizeof(T) == 2)
    NEG_INF = -5e4f;

  mask_val = (1.0f - mask_val) * NEG_INF;

  float NN = -1e20f;
  if(sizeof(T) == 2)
    NN = -5e4f;
  float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : NN;

  float max_val = blockReduceMax<float>(tmp);
  if(threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if(threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if(threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template<typename T>
__global__
void add_QKV_bias(T* Q, T* K, T* V, T* q_buf_, T* k_buf_, T* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block) {

  T* data_ptr;
  T* buf_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0) {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
  } else if(qkv_id == 1) {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
  } else {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i) {
    T tmp = data_ptr[threadIdx.x];

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template<typename T>
void multiHeadAttr_nofuse_kernelLauncher(
  cudaStream_t stream,
  cublasHandle_t cublas_handle,
  T* Q,
  T* K,
  T* V,
  T* q_buf_,
  T* k_buf_,
  T* v_buf_,
  T* qk_buf_,
  T* transpose_dst_,
  const T* attr_mask,
  T* dst,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  const T scalar) {

  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  dim3 grid;
  dim3 block;
  int cublasAlgo_[3];
  cudaDataType_t computeType_ = CUDA_R_32F;
  cudaDataType_t AType_ = CUDA_R_32F;
  cudaDataType_t BType_ = CUDA_R_32F;
  cudaDataType_t CType_ = CUDA_R_32F;
  if(sizeof(T) == 2) {
    computeType_ = CUDA_R_16F;
    AType_ = CUDA_R_16F;
    BType_ = CUDA_R_16F;
    CType_ = CUDA_R_16F;
    cublasAlgo_[0] = 99;
    cublasAlgo_[1] = 99;
    cublasAlgo_[2] = 99;
  } else {
    cublasAlgo_[0] = -1;
    cublasAlgo_[1] = -1;
    cublasAlgo_[2] = -1;
  }


  //if(sizeof(T) == 4)
  if(1) {
    const int word_per_block = 1;
    assert(k <= 1024);
    assert(m / word_per_block * 3 <= 65536);

    dim3 grid(m / word_per_block * 3);
    dim3 block(k);
    add_QKV_bias<T><<<grid, block, 0, stream>>>(Q, K, V, q_buf_, k_buf_, v_buf_, batch_size, seq_len, head_num,
        size_per_head, word_per_block);
  } else {
    const int word_per_block = 1;
    grid.x = batch_size * seq_len / word_per_block;
    block.x = head_num * size_per_head * word_per_block / 2;

    assert(block.x <= 1024);

    add_QKV_bias<T><<<grid, block, 0, stream>>>(Q, K, V, q_buf_, k_buf_, v_buf_, batch_size, seq_len, head_num,
        size_per_head / 2, word_per_block);
  }

  T alpha = (T)1.0f, beta = (T)0.0f;

  check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                   CUBLAS_OP_T, CUBLAS_OP_N,
                   seq_len, seq_len, size_per_head,
                   &alpha,
                   k_buf_, AType_, size_per_head, seq_len * size_per_head,
                   q_buf_, BType_, size_per_head, seq_len * size_per_head,
                   &beta,
                   qk_buf_, CType_, seq_len, seq_len * seq_len,
                   batch_size * head_num,
                   computeType_,
                   static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

  if(seq_len <= 32)
    block.x = 32;
  else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
  else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(batch_size * head_num <= 120) {
    grid.x = batch_size * head_num * seq_len;
    softmax_kernel_v2<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
  } else {
    grid.x = batch_size * head_num;
    softmax_kernel<T><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
  }

  //cudaMemcpy(dst, qk_buf_, m*seq_len*head_num*sizeof(T), cudaMemcpyDeviceToDevice);
  //return ;

  check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   size_per_head, seq_len, seq_len,
                   &alpha,
                   v_buf_, AType_, size_per_head, seq_len * size_per_head,
                   qk_buf_, BType_, seq_len, seq_len * seq_len,
                   &beta,
                   transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
                   batch_size * head_num,
                   computeType_,
                   static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  //if(sizeof(T) == 2)
  if(0) {
    const int seq_per_block = 4;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;

    assert(grid.x * seq_per_block == batch_size * head_num * seq_len);

    transpose<T><<<grid, block, 0, stream>>>(transpose_dst_, dst,
        batch_size, seq_len, head_num, size_per_head / 2);
  } else {
    const int seq_per_block = 1;
    grid.x = batch_size * head_num * seq_len / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<T><<<grid, block, 0, stream>>>(transpose_dst_, dst,
        batch_size, seq_len, head_num, size_per_head);
  }
}

template<typename T>
void OpenMultiHeadAttention(cublasHandle_t& cublas, const int B, const int L, const int numHeads,
                            const int headSize, const float rsqrtHeadSize, TransformerEncodeNode<T>& nodes,
                            TransformerEncodeTemp<T>& temps, cudaStream_t stream) {
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t AType = CUDA_R_32F;
  cudaDataType_t BType = CUDA_R_32F;
  cudaDataType_t CType = CUDA_R_32F;
  int cublasAlgo_[3];
  if(sizeof(T) == 4) {
    cublasAlgo_[0] = -1;
    cublasAlgo_[1] = -1;
    cublasAlgo_[2] = -1;
  } else {
    cublasAlgo_[0] = 99;
    cublasAlgo_[1] = 99;
    cublasAlgo_[2] = 99;
    computeType = CUDA_R_16F;
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
  }

  int m = B * L;
  int k = numHeads * headSize;
  int n = k;

  T alpha = (T)1.0f, beta = (T)0.0f;
  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_attention.query_weight.kernel, AType, n,
                                temps.att_in_buf0, BType, k,
                                &beta,
                                temps.q_buf0, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_attention.key_weight.kernel, AType, n,
                                temps.att_in_buf1, BType, k,
                                &beta,
                                temps.k_buf0, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_attention.value_weight.kernel, AType, n,
                                temps.att_in_buf1, BType, k,
                                &beta,
                                temps.v_buf0, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  multiHeadAttr_nofuse_kernelLauncher<T>(
    stream,
    cublas,
    temps.q_buf0,
    temps.k_buf0,
    temps.v_buf0,
    temps.q_buf1,
    temps.k_buf1,
    temps.v_buf1,
    temps.qk_buf0,
    temps.transpose_dst,
    nodes.att_mask,
    temps.att_out_buf,
    B,
    L,
    numHeads,
    headSize,
    rsqrtHeadSize);
}

template <typename T>
cudaError_t transEncode(cublasHandle_t& cublas, const int B, const int L, const int numHeads, const int headSize,
                const float rsqrtHeadSize, bool is_last_layer, TransformerEncodeNode<T>& nodes,
                TransformerEncodeTemp<T>& temps, cudaStream_t stream) {
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t AType = CUDA_R_32F;
  cudaDataType_t BType = CUDA_R_32F;
  cudaDataType_t CType = CUDA_R_32F;
  int cublasAlgo_[5];
  if(sizeof(T) == 4) {
    cublasAlgo_[0] = -1;
    cublasAlgo_[1] = -1;
    cublasAlgo_[2] = -1;
    cublasAlgo_[3] = -1;
    cublasAlgo_[4] = -1;
  } else {
    cublasAlgo_[0] = 99;
    cublasAlgo_[1] = 99;
    cublasAlgo_[2] = 99;
    cublasAlgo_[3] = 99;
    cublasAlgo_[4] = 99;
    computeType = CUDA_R_16F;
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
  }

  cudaError_t status = cudaSuccess;
  int m = B * L;
  int k = numHeads * headSize;
  int n = k;
  const size_t wordSize = sizeof(T);
  const size_t buf_size = B * L * numHeads * headSize * wordSize;

  // printf("encode_input: \n");
  // print_first_k((T *) nodes.encode_input, 50, stream);
  im2col1d_gpu<T>(nodes.encode_input, temps.im2col_buf0, k, m, L, 3, 1, stream);

  T alpha = (T)1.0f;
  T beta = (T)0.0f;
  k = numHeads * headSize * 3;

  // printf("im2col_buf0: \n");
  // print_first_k((T *) temps.im2col_buf0, 50, stream);
  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_conv1d0.kernel, AType, n,
                                temps.im2col_buf0, BType, k,
                                &beta,
                                temps.conv1d_buf0, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  k = numHeads * headSize;
  n = k;
  add_bias_kernelLauncher<T>(temps.conv1d_buf0, nodes.self_conv1d0.bias, m, n, true, stream);

  layernorm_kernelLauncher<T>(temps.layernorm_buf0, temps.conv1d_buf0, nodes.self_layernorm0.gamma,
                              nodes.self_layernorm0.beta, m, n, stream);

  // printf("layernorm_buf0: \n");
  // print_first_k((T *) temps.layernorm_buf0, 50, stream);
  im2col1d_gpu<T>(temps.layernorm_buf0, temps.im2col_buf1, k, m, L, 3, 1, stream);

  k = numHeads * headSize * 3;
  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_conv1d1.kernel, AType, n,
                                temps.im2col_buf1, BType, k,
                                &beta,
                                temps.conv1d_buf1, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

  k = numHeads * headSize;
  n = k;
  add_bias_kernelLauncher<T>(temps.conv1d_buf1, nodes.self_conv1d1.bias, m, n, true, stream);

  layernorm_kernelLauncher(temps.layernorm_buf1, temps.conv1d_buf1, nodes.self_layernorm1.gamma,
                           nodes.self_layernorm1.beta, m, n, stream);

  // printf("layernorm_buf1: \n");
  // print_first_k((T *) temps.layernorm_buf1, 50, stream);
  add_op<T>(temps.add_buf0, temps.layernorm_buf1, nodes.encode_input, buf_size, stream);

  layernorm_kernelLauncher<T>(temps.layernorm_buf2, temps.add_buf0, nodes.self_layernorm2.gamma,
                              nodes.self_layernorm2.beta, m, n, stream);

  check_cuda_error(cudaMemcpy(temps.att_in_buf0, temps.layernorm_buf2, buf_size, cudaMemcpyDeviceToDevice));
  check_cuda_error(cudaMemcpy(temps.att_in_buf1, temps.layernorm_buf2, buf_size, cudaMemcpyDeviceToDevice));

  OpenMultiHeadAttention(cublas, B, L, numHeads, headSize, rsqrtHeadSize, nodes, temps, stream);

  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.self_attention.attention_output_weight.kernel, AType, n,
                                temps.att_out_buf, BType, k,
                                &beta,
                                temps.att_matmul_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  add_op(temps.add_buf1, temps.att_matmul_buf, temps.add_buf0, buf_size, stream);
  layernorm_kernelLauncher(temps.ffn_layernorm_buf, temps.add_buf1, nodes.ffn_layernorm.gamma,
                           nodes.ffn_layernorm.beta, m, n, stream);
  // printf("att_matmul_buf: \n");
  // print_first_k((T *) temps.att_matmul_buf, 50, stream);

  n *= 2;
  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.ffn.intermediate_weight.kernel, AType, n,
                                temps.ffn_layernorm_buf, BType, k,
                                &beta,
                                temps.ffn_inter_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));

  add_bias_kernelLauncher<T>(temps.ffn_inter_buf, nodes.ffn.intermediate_weight.bias, m, n, true, stream);
  // printf("ffn_inter_buf: \n");
  // print_first_k((T *) temps.ffn_inter_buf, 50, stream);

  n = k;
  k *= 2;
  check_cuda_error(cublasGemmEx(cublas,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                nodes.ffn.output_weight.kernel, AType, n,
                                temps.ffn_inter_buf, BType, k,
                                &beta,
                                temps.ffn_out_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo_[4])));
  add_bias_kernelLauncher<T>(temps.ffn_out_buf, nodes.ffn.output_weight.bias, m, n, false, stream);
  // printf("ffn_out_buf: \n");
  // print_first_k((T *) temps.ffn_out_buf, 50, stream);

  if(is_last_layer) {
    add_mask_op(temps.add_buf2, temps.add_buf1, temps.ffn_out_buf, nodes.ffn_mask, buf_size, stream);
    layernorm_kernelLauncher(temps.layernorm_final_buf, temps.add_buf2, nodes.layernorm_final.gamma,
                             nodes.layernorm_final.beta, m, n, stream);
    check_cuda_error(cudaMemcpy(nodes.encode_out, temps.layernorm_final_buf, sizeof(T)*buf_size, cudaMemcpyDeviceToDevice));
  } else {
    add_mask_op(nodes.encode_out, temps.add_buf1, temps.ffn_out_buf, nodes.ffn_mask, buf_size, stream);
  }

  return status;
}

template <typename T>
TransformerEncodeNode<T>::TransformerEncodeNode(const void* const* inputs, void* const* outputs) {
  encode_input = static_cast<const T*>(inputs[0]);
  att_mask = static_cast<const T*>(inputs[1]);
  ffn_mask = static_cast<const T*>(inputs[2]);
  self_conv1d0.kernel = static_cast<const T*>(inputs[3]);
  self_conv1d0.bias = static_cast<const T*>(inputs[4]);
  self_layernorm0.gamma = static_cast<const T*>(inputs[5]);
  self_layernorm0.beta = static_cast<const T*>(inputs[6]);
  self_conv1d1.kernel = static_cast<const T*>(inputs[7]);
  self_conv1d1.bias = static_cast<const T*>(inputs[8]);
  self_layernorm1.gamma = static_cast<const T*>(inputs[9]);
  self_layernorm1.beta = static_cast<const T*>(inputs[10]);
  self_layernorm2.gamma = static_cast<const T*>(inputs[11]);
  self_layernorm2.beta = static_cast<const T*>(inputs[12]);
  self_attention.query_weight.kernel = static_cast<const T*>(inputs[13]);
  self_attention.key_weight.kernel = static_cast<const T*>(inputs[14]);
  self_attention.value_weight.kernel = static_cast<const T*>(inputs[15]);
  self_attention.attention_output_weight.kernel = static_cast<const T*>(inputs[16]);
  ffn_layernorm.gamma = static_cast<const T*>(inputs[17]);
  ffn_layernorm.beta = static_cast<const T*>(inputs[18]);
  ffn.intermediate_weight.kernel = static_cast<const T*>(inputs[19]);
  ffn.intermediate_weight.bias = static_cast<const T*>(inputs[20]);
  ffn.output_weight.kernel = static_cast<const T*>(inputs[21]);
  ffn.output_weight.bias = static_cast<const T*>(inputs[22]);
  layernorm_final.gamma = static_cast<const T*>(inputs[23]);
  layernorm_final.beta = static_cast<const T*>(inputs[24]);
  encode_out = static_cast<T*>(outputs[0]);
}

template <typename T>
TransformerEncodeTemp<T>::TransformerEncodeTemp(const int B, const int L, const int numHeads,
    const int headSize, void* workspace) {
  const size_t buf_size1 = B * L * numHeads * headSize;
  const size_t buf_size2 = B * L * L * numHeads;
  im2col_buf0 = static_cast<T*>(workspace);
  conv1d_buf0 = im2col_buf0 + buf_size1 * 3;
  layernorm_buf0 = conv1d_buf0 + buf_size1;
  im2col_buf1 = layernorm_buf0 + buf_size1;
  conv1d_buf1 = im2col_buf1 + buf_size1 * 3 ;
  layernorm_buf1 = conv1d_buf1 + buf_size1;
  add_buf0 = layernorm_buf1 + buf_size1;
  layernorm_buf2 = add_buf0 + buf_size1;
  att_in_buf0 = layernorm_buf2 + buf_size1;
  att_in_buf1 = att_in_buf0 + buf_size1;
  att_out_buf = att_in_buf1 + buf_size1;
  att_matmul_buf = att_out_buf + buf_size1;
  add_buf1 = att_matmul_buf + buf_size1;
  ffn_layernorm_buf = add_buf1 + buf_size1;
  ffn_inter_buf = ffn_layernorm_buf + buf_size1;
  ffn_out_buf = ffn_inter_buf + buf_size1 * 2;
  add_buf2 = ffn_out_buf + buf_size1;
  layernorm_final_buf = add_buf2 + buf_size1;
  q_buf0 = layernorm_final_buf + buf_size1;
  k_buf0 = q_buf0 + buf_size1;
  v_buf0 = k_buf0 + buf_size1;
  q_buf1 = v_buf0 + buf_size1;
  k_buf1 = q_buf1 + buf_size1;
  v_buf1 = k_buf1 + buf_size1;
  qk_buf0 = v_buf1 + buf_size1;
  transpose_dst = qk_buf0 + buf_size2;
}

namespace {
static const char* TRANSFORMER_ENCODE_PLUGIN_VERSION{"1"};
static const char* TRANSFORMER_ENCODE_PLUGIN_NAME{"TransformerEncodePluginDynamic"};
} // namespace

TransformerEncodePluginDynamic::TransformerEncodePluginDynamic(const std::string name, const DataType type,
    const int hiddenSize, const int numHeads, const bool isLastLayer)
  : mLayerName(name)
  , mHiddenSize(hiddenSize)
  , mNumHeads(numHeads)
  , mType(type)
  , mIsLastLayer(isLastLayer) {
  assert(hiddenSize % numHeads == 0);
  mHeadSize = hiddenSize / numHeads;
  mRsqrtHeadSize = 1.f / sqrt(float(mHeadSize));
}

TransformerEncodePluginDynamic::TransformerEncodePluginDynamic(const std::string name, const void* data, size_t length)
  : mLayerName(name) {
  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mNumHeads);
  deserialize_value(&data, &length, &mHeadSize);
  deserialize_value(&data, &length, &mRsqrtHeadSize);
  deserialize_value(&data, &length, &mHiddenSize);
  deserialize_value(&data, &length, &mIsLastLayer);
}

nvinfer1::IPluginV2DynamicExt* TransformerEncodePluginDynamic::clone() const noexcept {
  auto ret = new TransformerEncodePluginDynamic(mLayerName, mType, mHiddenSize, mNumHeads, mIsLastLayer);
  ret->initialize();
  return ret;
}

DimsExprs TransformerEncodePluginDynamic::getOutputDimensions(
  int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
  assert(outputIndex == 0);
  return inputs[0];
}

bool TransformerEncodePluginDynamic::supportsFormatCombination(
  int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  const auto* in_out_tensor = inOut + pos;
  return (mType==in_out_tensor->type);

  // const auto* in1 = inOut;
  // const auto* in2 = inOut+1;
  // const auto* in3 = inOut+2;
  // const auto* out = inOut + nbInputs;
  // if(pos == 0)
  // {
  //     return (mType==in1->type);
  // }
  // else if(pos == 1)
  // {
  //     return (mType==in2->type);
  // }
  // else if(pos == 2)
  // {
  //     return (mType==in3->type);
  // }
  // else if(pos == 3)
  // {
  //     return (mType==in1->type && mType==out->type);
  // }
  // return false;
}

void TransformerEncodePluginDynamic::configurePlugin(
  const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  assert(nbInputs == 25);
  assert(nbOutputs == 1);
  const PluginTensorDesc& inDesc = in[0].desc;
  TRT_UNUSED inDesc;
  const PluginTensorDesc& outDesc = out->desc;
  TRT_UNUSED outDesc;

  //std::cout<<"mType:"<<(int)mType<<" inDesc.type:"<<(int)inDesc.type<<" outDesc.type:"<<(int)outDesc.type<<std::endl;

  assert(mType == inDesc.type);
  assert(mType == outDesc.type);
  assert(inDesc.dims.d[0] == outDesc.dims.d[0]);
  assert(inDesc.dims.d[1] == outDesc.dims.d[1]);
  assert(inDesc.dims.d[2] == outDesc.dims.d[2]);
}

size_t TransformerEncodePluginDynamic::getWorkspaceSize(
  const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  const int B = inputs->dims.d[0];
  const int L = inputs->dims.d[1];

  const size_t wordSize = bert::getElementSize(mType);
  // 一开始给中间变量分配好内存空间
  const size_t ws = 31 * B * L * mHiddenSize * wordSize + B * L * L * mNumHeads * wordSize;
  std::cout<<"encode workspace:"<< ws << " B:" << B << " L:" << L << " wordSize:"<< wordSize << " hidden_units:"<< mHiddenSize << std::endl;
  return ws;
}

// IPluginV2Ext Methods
DataType TransformerEncodePluginDynamic::getOutputDataType(
  int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  assert(index == 0);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* TransformerEncodePluginDynamic::getPluginType() const noexcept {
  return TRANSFORMER_ENCODE_PLUGIN_NAME;
}

const char* TransformerEncodePluginDynamic::getPluginVersion() const noexcept {
  return TRANSFORMER_ENCODE_PLUGIN_VERSION;
}

int TransformerEncodePluginDynamic::getNbOutputs() const noexcept {
  return 1;
}

int TransformerEncodePluginDynamic::initialize() noexcept {
  cublasCreate(&cublas);
  return 0;
}

void TransformerEncodePluginDynamic::terminate() noexcept {
  CHECK(cublasDestroy(cublas));
}

size_t TransformerEncodePluginDynamic::getSerializationSize() const noexcept {
  return sizeof(DataType) + sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mRsqrtHeadSize) + sizeof(mHiddenSize) + sizeof(mIsLastLayer);
}

void TransformerEncodePluginDynamic::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, mType);
  serialize_value(&buffer, mNumHeads);
  serialize_value(&buffer, mHeadSize);
  serialize_value(&buffer, mRsqrtHeadSize);
  serialize_value(&buffer, mHiddenSize);
  serialize_value(&buffer, mIsLastLayer);
}

void TransformerEncodePluginDynamic::destroy() noexcept {
  delete this;
}

void TransformerEncodePluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* TransformerEncodePluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

int TransformerEncodePluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  const int B = inputDesc->dims.d[0];
  const int L = inputDesc->dims.d[1];

  cudaError_t status = cudaSuccess;
  if (mType == DataType::kFLOAT) {
    TransformerEncodeNode<float> nodes(inputs, outputs);
    TransformerEncodeTemp<float> temps(B, L, mNumHeads, mHeadSize, workspace);
    status = transEncode<float>(
               cublas, B, L, mNumHeads, mHeadSize, mRsqrtHeadSize, mIsLastLayer, nodes, temps, stream);
  } else {
    TransformerEncodeNode<half> nodes(inputs, outputs);
    TransformerEncodeTemp<half> temps(B, L, mNumHeads, mHeadSize, workspace);
    status = transEncode<half>(
               cublas, B, L, mNumHeads, mHeadSize, mRsqrtHeadSize, mIsLastLayer, nodes, temps, stream);
  }
  assert(status == cudaSuccess);
  return status;
}

PluginFieldCollection TransformerEncodePluginDynamicCreator::mFC{};
std::vector<PluginField> TransformerEncodePluginDynamicCreator::mPluginAttributes;

TransformerEncodePluginDynamicCreator::TransformerEncodePluginDynamicCreator() {
  mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("isLastLayer", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TransformerEncodePluginDynamicCreator::getPluginName() const noexcept {
  return TRANSFORMER_ENCODE_PLUGIN_NAME;
}

const char* TransformerEncodePluginDynamicCreator::getPluginVersion() const noexcept {
  return TRANSFORMER_ENCODE_PLUGIN_VERSION;
}

const PluginFieldCollection* TransformerEncodePluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* TransformerEncodePluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  int typeId = -1;
  int hiddenSize = 0;
  int numHeads = 0;
  bool isLastLayer = false;

  // std::cout << "fc->nbFields:" << fc->nbFields << std::endl;
  // std::cout << "name:" << name << std::endl;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("hidden_size") == 0) {
      hiddenSize = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("num_heads") == 0) {
      numHeads = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("isLastLayer") == 0) {
      isLastLayer = *static_cast<const bool*>(fc->fields[i].data);
    }
  }
  if (typeId < 0) {
    std::cout << "TransformerEncode: Invalid TypeId " << typeId << std::endl;
  }
  if (hiddenSize <= 0) {
    std::cout << "TransformerEncode: Invalid hiddenSize " << hiddenSize << std::endl;
  }
  if (numHeads <= 0) {
    std::cout << "TransformerEncode: Invalid numHeads " << numHeads << std::endl;
  }
  if (isLastLayer < 0) {
    std::cout << "TransformerEncode: Invalid isLastLayer " << isLastLayer << std::endl;
  }

  // std::cout << "typeId:" << typeId << "hidden_size:" << hiddenSize << "num_heads:" << numHeads << "isLastLayer:" << isLastLayer << std::endl;
  DataType type = static_cast<DataType>(typeId);
  TransformerEncodePluginDynamic* p = new TransformerEncodePluginDynamic(name, type, hiddenSize, numHeads, isLastLayer);
  return p;
}

IPluginV2* TransformerEncodePluginDynamicCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call TransformerEncodePluginDynamic::destroy()
  return new TransformerEncodePluginDynamic(name, serialData, serialLength);
}

void TransformerEncodePluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* TransformerEncodePluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

// REGISTER_TENSORRT_PLUGIN(TransformerEncodePluginDynamicCreator);

