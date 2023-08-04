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
#include <vector>
#include "NvInfer.h"
#include "bertCommon.h"
#include "transformer_decode.h"
// #include "common.h"
#include "serialize.hpp"
#include "transformerKernels.h"

using namespace transformer;

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::TransformerDecodePluginDynamic;
using nvinfer1::plugin::TransformerDecodePluginDynamicCreator;

template <typename T>
__global__
void add_bias_relu(T* out, const T* bias, int m, int n) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m) {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > 0.0f ? val : 0.0f);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__
void add_bias_relu(half* out, const half* bias, int m, int n) {
  half2 val, reg_bias;
  int row_id = blockIdx.x;
  int ite = n / blockDim.x / 2;
  int tid = threadIdx.x;

  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;
  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias_ptr[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m) {
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
      val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = val;
      row_id += gridDim.x;
    }
  }
}

template <typename T>
__global__
void masked_attention_kernel(T* query_buf,
                             T* key_cache, T* value_cache, T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar) {
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  // int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id];
  __syncthreads();

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite) {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head) {
      key_cache[ite * offset + qkv_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f;
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();


  if(tid < size_per_head) {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite) {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1) {
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template<typename T>
void masked_multi_head_attention(cublasHandle_t& cublas_handle, int* cublasAlgo, const cudaDataType_t computeType,
                                 const cudaDataType_t AType, const cudaDataType_t BType, const cudaDataType_t CType, const int batch_size, const int hidden_units,
                                 const int head_num, const int size_per_head, const T* from_tensor, T* key_cache, T* value_cache,
                                 AttentionWeight<T>& self_attention, T* query_buf, T* context_buf, T* decoder_output,
                                 const int step, cudaStream_t& stream) {
  int m = batch_size;
  int n = hidden_units;
  int k = hidden_units;

  T* key_buf = key_cache + (step - 1) * m * n;
  T* value_buf = value_cache + (step - 1) * m * n;

  T alpha = (T)1.0f, beta = (T)0.0f;

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                self_attention.query_weight.kernel, AType, n,
                                from_tensor, BType, k,
                                &beta,
                                query_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                self_attention.key_weight.kernel, AType, n,
                                from_tensor, BType, k,
                                &beta,
                                key_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                self_attention.value_weight.kernel, AType, n,
                                from_tensor, BType, k,
                                &beta,
                                value_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

  dim3 grid(batch_size * head_num);
  dim3 block(128);

  //suppose size_per_head <= 128
  if(step <= 64)
    block.x = 64;
  else if(step <= 128 && step > size_per_head)
    block.x = 128;
  else if(step > 128 && step <= 256)
    block.x = 256;
  else if(step > 256 && step <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head)
    block.x = size_per_head;

  assert(block.x <= 1024);

  T scalar = 1 / sqrtf(size_per_head * 1.0f);

  int shared_size = sizeof(T) * (size_per_head + step);

  masked_attention_kernel<T><<<grid, block, shared_size, stream>>>(
    query_buf, key_cache, value_cache,
    context_buf, batch_size, head_num, size_per_head, step, scalar);

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                self_attention.attention_output_weight.kernel, AType, n,
                                context_buf, BType, k,
                                &beta,
                                decoder_output, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));
}

template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf,
  T* key_cache,
  T* value_cache,
  const int* length_per_sample, T* context_buf,
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar) {
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  // int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite) {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
                 + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head) {
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f;
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head) {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite) {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head
                     + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1) {
        value_cache[value_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

/* attention with source sentence */
template<typename T>
void cross_multi_head_attention(cublasHandle_t& cublas_handle, int* cublasAlgo, const cudaDataType_t computeType,
                                const cudaDataType_t AType, const cudaDataType_t BType, const cudaDataType_t CType, const int batch_size, const int hidden_units,
                                const int head_num, const int size_per_head, const T* from_tensor, const T* memory_tensor, T* key_mem_cache,
                                T* value_mem_cache, AttentionWeight<T>& cross_attention, T* query_buf, T* context_buf,
                                T* decoder_output, const int* length, const int seq_len, const int step, cudaStream_t& stream) {
  int m = batch_size;
  int n = hidden_units;
  int k = hidden_units;

  T alpha = (T)1.0f, beta = (T)0.0f;

  //reuse the query_buf
  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                cross_attention.query_weight.kernel, AType, n,
                                from_tensor, BType, k,
                                &beta,
                                query_buf, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));

  if(step == 1) {
    m *= seq_len;
    k = hidden_units;
    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k,
                                  &alpha,
                                  cross_attention.key_weight.kernel, AType, n,
                                  memory_tensor, BType, k,
                                  &beta,
                                  key_mem_cache, CType, n,
                                  computeType,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo[1])));

    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k,
                                  &alpha,
                                  cross_attention.value_weight.kernel, AType, n,
                                  memory_tensor, BType, k,
                                  &beta,
                                  value_mem_cache, CType, n,
                                  computeType,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo[1])));
    k = hidden_units;
  }

  dim3 grid(batch_size * head_num);
  dim3 block(128);

  if(seq_len <= 64)
    block.x = 64;
  else if(seq_len <= 128 && seq_len > size_per_head)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head)
    block.x = size_per_head;

  assert(block.x <= 1024);

  T scalar = 1 / sqrtf(size_per_head * 1.0f);

  int shared_size = sizeof(T) * (size_per_head + seq_len);
  cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(
    query_buf,
    key_mem_cache,
    value_mem_cache,
    length, context_buf,
    batch_size,
    head_num, size_per_head, step, seq_len, scalar);

  m = batch_size;
  n = head_num * size_per_head;
  k = n;

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha,
                                cross_attention.attention_output_weight.kernel, AType, n,
                                context_buf, BType, k,
                                &beta,
                                decoder_output, CType, n,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));
}

template <typename T>
__global__
void decoder_norm1_kernel(const T* input, const T* gamma, const T* beta, T* output, int m, int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;

  mean = blockReduceSum<float>(local_out);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  if(tid < n)
    output[blockIdx.x * n + tid] =
      (T)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template <typename T>
__global__
void decoder_norm2_kernel(const T* input, const T* gamma, const T* beta, T* output, T* norm_output, int m, int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if(tid < n) {
    local_out = (float)(__ldg(&input[blockIdx.x * n + tid]));
    local_out += (float)(output[blockIdx.x * n + tid]);
    output[blockIdx.x * n + tid] = (T)local_out;
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < n)
    norm_output[blockIdx.x * n + tid] =
      (T)((local_out - s_mean) * s_variance * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}


template<typename T>
void decoder_norm1(
  const T* input,
  const T* gamma,
  const T* beta,
  T* output,
  int m, int n, cudaStream_t& stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if(n % 32 != 0)
    block.x = 1024;

  assert(n <= 1024);

  /* should pay attention to the rsqrt precision*/
  decoder_norm1_kernel<T><<<grid, block, 0, stream>>>(input, gamma, beta, output, m, n);
}

template<typename T>
void decoder_norm2(
  const T* input,
  const T* gamma,
  const T* beta,
  T* output,
  T* norm_output,
  int m, int n, cudaStream_t& stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  assert(n <= 1024);

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if(n % 32 != 0)
    block.x = 1024;

  /* should pay attention to the rsqrt precision*/
  decoder_norm2_kernel<T><<<grid, block, 0, stream>>>(input, gamma, beta, output, norm_output, m, n);
}

template<typename T>
void ffn(cublasHandle_t& cublas_handle, int* cublasAlgo, const cudaDataType_t computeType,
         const cudaDataType_t AType, const cudaDataType_t BType, const cudaDataType_t CType,
         const T* input, T* ffn_inner, FFNWeight<T>& ffn,
         T* output, const int m, const int inner_size, const int n, cudaStream_t& stream) {
  int m1 = m, k1 = n, n1 = inner_size;
  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n1, m1, k1,
                                &alpha,
                                ffn.intermediate_weight.kernel, AType, n1,
                                input, BType, k1,
                                &beta,
                                ffn_inner, CType, n1,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[2])));

  dim3 grid(m1);
  dim3 block(n1 / 4);

  assert(block.x <= 1024);

  add_bias_relu<T><<<grid, block, 0, stream>>>(ffn_inner, ffn.intermediate_weight.bias, m1, n1);

  int m2 = m, n2 = n, k2 = inner_size;
  check_cuda_error(cublasGemmEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n2, m2, k2,
                                &alpha,
                                ffn.output_weight.kernel, AType, n2,
                                ffn_inner, BType, k2,
                                &beta,
                                output, CType, n2,
                                computeType,
                                static_cast<cublasGemmAlgo_t>(cublasAlgo[3])));
}

template <typename T>
__global__
void add_bias_input_kernel(T* output, const T* input, const T* bias, const int m, const int n) {
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id] + __ldg(&bias[threadIdx.x]);
}

template<typename T>
void add_bias_input(T* output, const T* input, const T* ffn_bias2, const int m, const int n, cudaStream_t& stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_kernel<<<grid, block, 0, stream>>>(output, input, ffn_bias2, m, n);
}

template<typename T>
void docoder_forward(cublasHandle_t& cublas, const int B, const int L, const int numHeads, const int headSize,
                     const int beamWidth, const T *memory_tensor, const int *memory_sequence_length,
                     const T *from_tensor, T *key_cache, T *value_cache, T *key_mem_cache,
                     T *value_mem_cache, T* decoder_buf, T *decoder_output, DecodeAttentionWeight<T>& attention_weight,
                     const int step, cudaStream_t& stream) {
  cudaDataType_t computeType;
  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  int cublasAlgo[4];
  if (sizeof(T) == 4) {
    cublasAlgo[0] = CUBLAS_GEMM_DEFAULT;
    cublasAlgo[1] = CUBLAS_GEMM_DEFAULT;
    cublasAlgo[2] = CUBLAS_GEMM_DEFAULT;
    cublasAlgo[3] = CUBLAS_GEMM_DEFAULT;
    computeType = CUDA_R_32F;
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
  } else {
    cublasAlgo[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasAlgo[1] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasAlgo[2] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    cublasAlgo[3] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    computeType = CUDA_R_16F;
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
  }

  int m = B * beamWidth;
  int n = numHeads * headSize;

  int decoder_buf_size = m * n;
  T* norm_from_tensor_buf = decoder_buf;
  T* query_buf = decoder_buf + decoder_buf_size;
  T* context_buf = decoder_buf + 2 * decoder_buf_size;
  T* masked_output_buf = decoder_buf + 3 * decoder_buf_size;
  T* norm_masked_output_buf = decoder_buf + 4 * decoder_buf_size;
  T* cross_output_buf = decoder_buf + 5 * decoder_buf_size;
  T* norm_cross_output_buf = decoder_buf + 6 * decoder_buf_size;
  T* ffn_inner_buf = decoder_buf + 7 * decoder_buf_size;

  /* masked multi-head attention */
  /* layernorm(from_tensor) -> norm_from_tensor_buf */
  decoder_norm1<T>(from_tensor, attention_weight.self_layernorm.gamma, attention_weight.self_layernorm.beta,
                   norm_from_tensor_buf, m, n, stream);
  // printf("norm_from_tensor_buf: \n");
  // print_first_k((T *) norm_from_tensor_buf, 50, stream);

  masked_multi_head_attention<T>(cublas, cublasAlgo, computeType, AType, BType, CType, B * beamWidth, n,
                                 numHeads, headSize, norm_from_tensor_buf, key_cache, value_cache,
                                 attention_weight.self_attention,
                                 query_buf, context_buf, masked_output_buf, step, stream);
  // printf("masked_output_buf: \n");
  // print_first_k((T *) masked_output_buf, 50, stream);

  /* add bias to masked_output_buf
     masked_output_buf + from_tensor -> masked_output_buf
     norm(masked_output_buf) -> norm_masked_output_buf
  */
  decoder_norm2<T>(from_tensor, attention_weight.cross_layernorm.gamma, attention_weight.cross_layernorm.beta,
                   masked_output_buf, norm_masked_output_buf, m, n, stream);
  // printf("norm_masked_output_buf: \n");
  // print_first_k((T *) norm_masked_output_buf, 50, stream);

  /* cross attention with memory */
  cross_multi_head_attention<T>(cublas, cublasAlgo, computeType, AType, BType, CType, B * beamWidth, n,
                                numHeads, headSize, norm_masked_output_buf, memory_tensor, key_mem_cache,
                                value_mem_cache, attention_weight.cross_attention,
                                query_buf, context_buf, cross_output_buf, memory_sequence_length, L, step, stream);
  // printf("cross_output_buf: \n");
  // print_first_k((T *) cross_output_buf, 50, stream);

  /* cross_output_buf + bias + masked_output_buf -> cross_output_buf
     norm(cross_otuput_buf) -> normed_last_context (input for ffn)
  */
  decoder_norm2<T>(masked_output_buf, attention_weight.ffn_layernorm.gamma, attention_weight.ffn_layernorm.beta,
                   cross_output_buf, norm_cross_output_buf, m, n, stream);
  // printf("norm_cross_output_buf: \n");
  // print_first_k((T *) norm_cross_output_buf, 50, stream);

  ffn<T>(cublas, cublasAlgo, computeType, AType, BType, CType, norm_cross_output_buf, ffn_inner_buf,
         attention_weight.ffn, decoder_output, m, 2 * n, n, stream);
  // printf("decoder_output: \n");
  // print_first_k((T *) decoder_output, 50, stream);

  add_bias_input<T>(decoder_output, cross_output_buf, attention_weight.ffn.output_weight.bias, m, n, stream);
  // printf("decoder_output: \n");
  // print_first_k((T *) decoder_output, 50, stream);
}

template <typename T>
void BeamSearch_OpenNMT(
  float *log_probs, float *cum_log_probs, bool *finished,
  T **key_cache, T **value_cache,
  int *parent_ids,
  int *sequence_length,
  int *word_ids,
  int *ids,
  int *output_ids,
  const int batch_size, const int beam_width,
  const int vocab_size, const int hidden_dim, const int step,
  const int cache_size, const int decoder_layers, cudaStream_t& stream,
  const int end_id,
  int *finished_count) {
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
  // printf("broadcast_kernelLauncher: \n");
  // print_first_k((float *) cum_log_probs, 50, stream);
  /*
      User can check the broadcast_kernel by broadcast_kernel_check.
      broadcast_kernel_check will compare the results of GPU and CPU.
      Note that broadcast_kernel_check contains broadcast_kernelLauncher and uses do not need to call it again.
  */
  // broadcast_kernel_check(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);

  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
  // printf("topK: \n");
  // print_first_k((int *) ids, 50, stream);
  /*
      User can check the topK by topK_check.
      topK_check will compare the results of GPU and CPU.
      Note that topK_check contains topK and uses do not need to call it again.
  */
  // topK_kernel_check(log_probs, ids, batch_size, beam_width, vocab_size, stream);
  update(log_probs, cum_log_probs, ids, finished,
         parent_ids, sequence_length, word_ids, output_ids,
         batch_size, beam_width, vocab_size, stream,
         end_id, finished_count);
  // printf("output_ids: \n");
  // print_first_k((int *) output_ids, 50, stream);
  /*
      User can check the update by update_kernel_check.
      update_kernel_check will compare the results of GPU and CPU.
      Note that update_kernel_check contains update and uses do not need to call it again.
  */
  // update_kernel_check(log_probs, cum_log_probs, ids, finished, parent_ids, sequence_length, word_ids, output_ids,
  //                     batch_size, beam_width, vocab_size, stream, end_id, finished_count);

  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size,
                     beam_width, hidden_dim, step, cache_size,
                     decoder_layers, stream);
  /*
      User can check the update_KV_cache by update_KV_cache_kernel_check.
      update_KV_cache_kernel_check will compare the results of GPU and CPU.
      Note that update_KV_cache_kernel_check contains update_KV_cache and uses do not need to call it again.
  */
  // update_KV_cache_kernel_check(key_cache, value_cache, parent_ids, batch_size, beam_width, hidden_dim, step, cache_size, decoder_layers, stream);
}

template <typename T>
cudaError_t transDecode(cublasHandle_t& cublas, const int B, const int L, const int numHeads, const int headSize,
                const int beamWidth, const int vocabSize, const int startId, const int endId, const int numLayer,
                TransformerDecodeNode<T>& nodes, TransformerDecodeTemp<T>& temps, cudaStream_t stream) {
  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  int cublasAlgo[1] = {20};
  if (sizeof(T) == 4) {
    cublasAlgo[0] = CUBLAS_GEMM_DEFAULT;
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
  } else {
    cublasAlgo[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
  }
  cudaError_t status = cudaSuccess;

  int m = B * beamWidth;
  int k = numHeads * headSize;
  int n = vocabSize;

  // printf("memory_tensor: \n");
  // print_first_k((T *) nodes.memory_tensor, 50, stream);
  // printf("memory_sequence_length: \n");
  // print_first_k((int *) nodes.memory_sequence_length, 50, stream);

  /*
    sequence_length initialize to 0
    finished: false
    word_ids: start_id_
    [cum_log_probs (for eacm beam, the first element is 0). e.g., [0 -inf -inf -inf][0 -inf -inf -inf]
  */

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  init(temps.finished_buf, nodes.sequence_length, temps.word_ids_buf, temps.cum_log_buf, temps.topk_ids_buf,
       startId, B, beamWidth, vocabSize, stream);
  // gpu_timer.Stop();
  // printf("init: %f ", (float(gpu_timer.ElapsedMillis())));

  /*
    User can check the init by init_kernel_check.
    init_kernel_check will compare the results of GPU and CPU.
    Note that init_kernel_check contains init and uses do not need to call it again.
  */
  // init_kernel_check(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
  //                   start_id_, batch_size_, beam_width_, decoding_params.stream);

  int cache_size = B * beamWidth * L * k; // type T

  for (int step = 1; step <= L; ++step) {
    // gpu_timer.Start();
    //we use two-way buffer
    int kv_cache_id = step & 0x1;

    embedding_lookup<T>(nodes.embedding_table, temps.word_ids_buf, temps.from_tensor_buf[0],
                        B, beamWidth, k, stream);

    // printf("embedding_lookup: \n");
    // print_first_k((T *) temps.from_tensor_buf[0], 50, stream);

    sine_position_encoder<T>(temps.from_tensor_buf[0], step - 1, m, k, stream);
    // printf("sine_position_encoder: \n");
    // print_first_k((T *) temps.from_tensor_buf[0], 50, stream);

    // gpu_timer.Stop();
    // printf("sine_position_encoder: %f ", (float(gpu_timer.ElapsedMillis())));

    // gpu_timer.Start();
    int from_id, out_id;
    for (int layer = 0; layer < numLayer; ++layer) {
      /*
      For the first layer (layer-0), from_id is 0. We also stored the embedding lookup
      result in from_tensor_[0]
      */
      from_id = layer & 0x1;
      out_id = 1 - from_id;

      DecodeAttentionWeight<T> attention_weight = nodes.multi_attention[layer];
      docoder_forward<T>(cublas, B, L, numHeads, headSize, beamWidth,
                         nodes.memory_tensor, nodes.memory_sequence_length, temps.from_tensor_buf[from_id],
                         temps.K_cache_buf[kv_cache_id] + layer * cache_size,
                         temps.V_cache_buf[kv_cache_id] + layer * cache_size,
                         temps.K_mem_cache_buf[layer], temps.V_mem_cache_buf[layer], temps.decoder_buf,
                         temps.from_tensor_buf[out_id], attention_weight, step, stream);
    }
    // gpu_timer.Stop();
    // printf("docoder_forward: %f ", (float(gpu_timer.ElapsedMillis())));

    // gpu_timer.Start();
    decoder_norm1<T>(temps.from_tensor_buf[out_id], nodes.layernorm.gamma, nodes.layernorm.beta,
                     temps.decoder_normed_result_buf, m, k, stream);

    float alpha = (float)1.0f;
    float beta = (float)0.0f;

    check_cuda_error(cublasGemmEx(cublas,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, m, k,
                                  &alpha,
                                  nodes.embedding_table, AType, k,
                                  temps.decoder_normed_result_buf, BType, k,
                                  &beta,
                                  temps.logits_buf, CUDA_R_32F, n,
                                  CUDA_R_32F,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo[0])));
    // gpu_timer.Stop();
    // printf("cublasGemmEx: %f ", (float(gpu_timer.ElapsedMillis())));

    // gpu_timer.Start();
    update_logits_v2(temps.logits_buf, endId, temps.finished_buf, m, n, stream);
    // gpu_timer.Stop();
    // printf("update_logits_v2: %f ", (float(gpu_timer.ElapsedMillis())));

    // printf("update_logits_v2: \n");
    // print_first_k((int *) temps.finished_buf, 50, stream);
    /*
        User can check the update_logits by update_logits_kernel_check.
        update_logits_kernel_check will compare the results of GPU and CPU.
        Note that update_logits_kernel_check contains update_logits and uses do not need to call it again.
    */
    // gpu_timer.Start();
    BeamSearch_OpenNMT<T>(
      temps.logits_buf, temps.cum_log_buf, temps.finished_buf,
      temps.K_cache_buf,
      temps.V_cache_buf,
      nodes.parent_ids + (step - 1) * B * beamWidth,
      nodes.sequence_length,
      temps.word_ids_buf,
      temps.topk_ids_buf,
      nodes.output_ids + (step - 1) * B * beamWidth,
      B, beamWidth, vocabSize, numHeads * headSize, step, cache_size, numLayer, stream,
      endId,
      temps.finished_count_buf);
    // gpu_timer.Stop();
    // printf("BeamSearch_OpenNMT: %f ", (float(gpu_timer.ElapsedMillis())));

    // TODO
    // Find a better method to check the is_finished
    cudaMemcpy(temps.h_finished_buf, temps.finished_buf, sizeof(bool) * B * beamWidth, cudaMemcpyDeviceToHost);
    int sum = 0;
    for(int i = 0; i < B * beamWidth; i++) {
      sum += (int)temps.h_finished_buf[i];
    }
    if(sum == B * beamWidth) {
      // printf("break step: %f ", (float(step)));
      break;
    }
  }
  return status;
}

template <typename T>
TransformerDecodeNode<T>::TransformerDecodeNode(const int num_layer, const int hidden_unit,
    const void* const* inputs, void* const* outputs) {
  memory_tensor = static_cast<const T*>(inputs[0]);
  memory_sequence_length = static_cast<const int*>(inputs[1]);
  for(int i = 0; i < num_layer; i++) {
    DecodeAttentionWeight<T> attention_weight;
    attention_weight.self_layernorm.gamma = static_cast<const T*>(inputs[2]) + i * hidden_unit;
    attention_weight.self_layernorm.beta = static_cast<const T*>(inputs[3]) + i * hidden_unit;
    attention_weight.self_attention.query_weight.kernel = static_cast<const T*>(inputs[4]) + i * hidden_unit * hidden_unit;
    attention_weight.self_attention.key_weight.kernel = static_cast<const T*>(inputs[5]) + i * hidden_unit * hidden_unit;
    attention_weight.self_attention.value_weight.kernel = static_cast<const T*>(inputs[6]) + i * hidden_unit * hidden_unit;
    attention_weight.self_attention.attention_output_weight.kernel = static_cast<const T*>(inputs[7]) + i * hidden_unit * hidden_unit;
    attention_weight.cross_layernorm.gamma = static_cast<const T*>(inputs[8]) + i * hidden_unit;
    attention_weight.cross_layernorm.beta = static_cast<const T*>(inputs[9]) + i * hidden_unit;
    attention_weight.cross_attention.query_weight.kernel = static_cast<const T*>(inputs[10]) + i * hidden_unit * hidden_unit;
    attention_weight.cross_attention.key_weight.kernel = static_cast<const T*>(inputs[11]) + i * hidden_unit * hidden_unit;
    attention_weight.cross_attention.value_weight.kernel = static_cast<const T*>(inputs[12]) + i * hidden_unit * hidden_unit;
    attention_weight.cross_attention.attention_output_weight.kernel = static_cast<const T*>(inputs[13]) + i * hidden_unit * hidden_unit;
    attention_weight.ffn_layernorm.gamma = static_cast<const T*>(inputs[14]) + i * hidden_unit;
    attention_weight.ffn_layernorm.beta = static_cast<const T*>(inputs[15]) + i * hidden_unit;
    attention_weight.ffn.intermediate_weight.kernel = static_cast<const T*>(inputs[16]) + i * hidden_unit * hidden_unit * 2;
    attention_weight.ffn.intermediate_weight.bias = static_cast<const T*>(inputs[17]) + i * hidden_unit * 2;
    attention_weight.ffn.output_weight.kernel = static_cast<const T*>(inputs[18]) + i * hidden_unit * hidden_unit * 2;
    attention_weight.ffn.output_weight.bias = static_cast<const T*>(inputs[19]) + i * hidden_unit;
    multi_attention.push_back(attention_weight);
  }
  layernorm.gamma = static_cast<const T*>(inputs[20]);
  layernorm.beta = static_cast<const T*>(inputs[21]);
  embedding_table = static_cast<const T*>(inputs[22]);
  output_ids = static_cast<int*>(outputs[0]);
  parent_ids = static_cast<int*>(outputs[1]);
  sequence_length = static_cast<int*>(outputs[2]);
}

template <typename T>
TransformerDecodeTemp<T>::TransformerDecodeTemp(const int B, const int L, const int mNumLayer, const int mHiddenSize,
    const int mBeamWidth, const int mVocabSize, void* workspace) {
  const int hidden_units = mHiddenSize;
  const int from_tensor_size = B * mBeamWidth * hidden_units; // type T
  const int decoder_workspace_size = 10 * from_tensor_size; // type T
  const int decoder_normed_result_buffer_size = B * mBeamWidth * hidden_units; // type T
  const int cache_size = B * mBeamWidth * L * hidden_units; // type T
  const int logits_buf_size = B * mBeamWidth * mVocabSize; // type float
  int cum_log_buf_size = B * mBeamWidth;  // type float
  int word_ids_buf_size = B * mBeamWidth; //type int
  int finished_buf_size = B * mBeamWidth; //type bool
  int topk_ids_buf_size = B * mBeamWidth * (ceil)((mBeamWidth * mVocabSize * 1.0) / 1024.0);
  int finished_count_size = (int)(ceil(1 / 4.)) * 4; // type int
  // prevent memory misalinged address
  cum_log_buf_size = (int)(ceil(cum_log_buf_size / 4.)) * 4;
  word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
  finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;
  topk_ids_buf_size = (int)(ceil(topk_ids_buf_size / 4.)) * 4;

  from_tensor_buf[0] = static_cast<T*>(workspace);
  from_tensor_buf[1] = from_tensor_buf[0] + from_tensor_size;
  for (int i = 0; i < mNumLayer; ++i) {
    K_mem_cache_buf[i] = from_tensor_buf[1] + from_tensor_size + i * cache_size * 2;
    V_mem_cache_buf[i] = from_tensor_buf[1] + from_tensor_size + i * cache_size * 2 + cache_size;
  }
  K_cache_buf[0] = V_mem_cache_buf[mNumLayer - 1] + cache_size + 0 * cache_size * mNumLayer;
  K_cache_buf[1] = V_mem_cache_buf[mNumLayer - 1] + cache_size + 1 * cache_size * mNumLayer;
  V_cache_buf[0] = V_mem_cache_buf[mNumLayer - 1] + cache_size + 2 * cache_size * mNumLayer;
  V_cache_buf[1] = V_mem_cache_buf[mNumLayer - 1] + cache_size + 3 * cache_size * mNumLayer;
  decoder_buf = V_cache_buf[1] + cache_size * mNumLayer;
  decoder_normed_result_buf = (decoder_buf + decoder_workspace_size);
  logits_buf = (float *)(decoder_normed_result_buf + decoder_normed_result_buffer_size);
  cum_log_buf = (float *)(logits_buf + logits_buf_size);
  word_ids_buf = (int *)(cum_log_buf + cum_log_buf_size);
  finished_buf = (bool *)(word_ids_buf + word_ids_buf_size);
  topk_ids_buf = (int *)(finished_buf + finished_buf_size);
  finished_count_buf = (int *)(topk_ids_buf + topk_ids_buf_size);
  h_finished_buf = new bool[finished_buf_size];
}

namespace {
static const char* TRANSFORMER_DECODE_PLUGIN_VERSION{"1"};
static const char* TRANSFORMER_DECODE_PLUGIN_NAME{"TransformerDecodePluginDynamic"};
} // namespace

TransformerDecodePluginDynamic::TransformerDecodePluginDynamic(const std::string name, const DataType type,
    const int hiddenSize, const int numHeads, const int beamWidth,
    const int vocabSize, const int startId, const int endId,
    const int numLayer)
  : mLayerName(name)
  , mHiddenSize(hiddenSize)
  , mNumHeads(numHeads)
  , mType(type)
  , mBeamWidth(beamWidth)
  , mVocabSize(vocabSize)
  , mStartId(startId)
  , mEndId(endId)
  , mNumLayer(numLayer) {
  assert(hiddenSize % numHeads == 0);
  mHeadSize = hiddenSize / numHeads;
}

TransformerDecodePluginDynamic::TransformerDecodePluginDynamic(const std::string name, const void* data, size_t length)
  : mLayerName(name) {
  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mNumHeads);
  deserialize_value(&data, &length, &mHeadSize);
  deserialize_value(&data, &length, &mHiddenSize);
  deserialize_value(&data, &length, &mBeamWidth);
  deserialize_value(&data, &length, &mVocabSize);
  deserialize_value(&data, &length, &mStartId);
  deserialize_value(&data, &length, &mEndId);
  deserialize_value(&data, &length, &mNumLayer);
}

nvinfer1::IPluginV2DynamicExt* TransformerDecodePluginDynamic::clone() const noexcept {
  auto ret = new TransformerDecodePluginDynamic(mLayerName, mType, mHiddenSize, mNumHeads, mBeamWidth, mVocabSize,
      mStartId, mEndId, mNumLayer);
  ret->initialize();
  return ret;
}

DimsExprs TransformerDecodePluginDynamic::getOutputDimensions(
  int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
  assert(outputIndex <= 2);
  if (outputIndex == 0 || outputIndex == 1) {
    DimsExprs output(inputs[0]);
    output.d[0] = inputs[0].d[1];
    output.d[1] = inputs[1].d[0];
    output.d[2] = inputs[1].d[1];
    return output;
  } else {
    DimsExprs output(inputs[1]);
    return output;
  }
}

bool TransformerDecodePluginDynamic::supportsFormatCombination(
  int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  const auto* in_out_tensor = inOut + pos;
  // std::cout<<"pos:"<<(int)pos<<" in_out_tensor.type:"<<(int)in_out_tensor->type <<std::endl;
  if (pos == 1 || pos == 23 || pos == 24 || pos == 25) {
    return (DataType::kINT32==in_out_tensor->type);
  } else {
    return (mType==in_out_tensor->type);
  }
}

void TransformerDecodePluginDynamic::configurePlugin(
  const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
  // input[0]: shape: B*5, W, 512
  // output[0]: shape: W, B, 5
  assert(nbInputs == 23);
  assert(nbOutputs == 3);
  const PluginTensorDesc& inDesc = in[0].desc;
  TRT_UNUSED inDesc;
  const PluginTensorDesc& outDesc = out[0].desc;
  TRT_UNUSED outDesc;

  // std::cout<<"mType:"<<(int)mType<<" inDesc.type:"<<(int)inDesc.type<<" outDesc.type:"<<(int)outDesc.type<<std::endl;

  assert(mType == inDesc.type);
  assert(inDesc.dims.d[1] == outDesc.dims.d[0]);
}

size_t TransformerDecodePluginDynamic::getWorkspaceSize(
  const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  const int B = inputs[1].dims.d[0];
  const int L = inputs[0].dims.d[1];

  const size_t wordSize = bert::getElementSize(mType);
  const int hidden_units = mHiddenSize;
  const int from_tensor_size = B * mBeamWidth * hidden_units; // type T
  const int decoder_workspace_size = 10 * from_tensor_size; // type T
  const int decoder_normed_result_buffer_size = B * mBeamWidth * hidden_units; // type T
  const int cache_size = B * mBeamWidth * L * hidden_units; // type T
  const int logits_buf_size = B * mBeamWidth * mVocabSize; // type float
  int cum_log_buf_size = B * mBeamWidth;  // type float
  int word_ids_buf_size = B * mBeamWidth; //type int
  int finished_buf_size = B * mBeamWidth; //type bool
  int topk_ids_buf_size = B * mBeamWidth * (ceil)((mBeamWidth * mVocabSize * 1.0) / 1024.0);
  int finished_count_size = (int)(ceil(1 / 4.)) * 4; // type int
  // prevent memory misalinged address
  cum_log_buf_size = (int)(ceil(cum_log_buf_size / 4.)) * 4;
  word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
  finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;
  topk_ids_buf_size = (int)(ceil(topk_ids_buf_size / 4.)) * 4;

  int datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                          cache_size * 6 * mNumLayer + decoder_normed_result_buffer_size;

  // 一开始给中间变量分配好内存空间
  const size_t ws = datatype_buf_size * wordSize + (logits_buf_size + cum_log_buf_size) * sizeof(float) + \
                    word_ids_buf_size * sizeof(int) + finished_buf_size * sizeof(bool) + \
                    topk_ids_buf_size * sizeof(int) + finished_count_size * sizeof(int);

  std::cout<<"decode workspace:"<< ws << " B:" << B << " L:" << L << " wordSize:"<< wordSize << " hidden_units:"<< hidden_units << std::endl;

  return ws;
}

// IPluginV2Ext Methods
DataType TransformerDecodePluginDynamic::getOutputDataType(
  int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  assert(index <= 2);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return DataType::kINT32;
}

// IPluginV2 Methods
const char* TransformerDecodePluginDynamic::getPluginType() const noexcept {
  return TRANSFORMER_DECODE_PLUGIN_NAME;
}

const char* TransformerDecodePluginDynamic::getPluginVersion() const noexcept {
  return TRANSFORMER_DECODE_PLUGIN_VERSION;
}

int TransformerDecodePluginDynamic::getNbOutputs() const noexcept {
  return 3;
}

int TransformerDecodePluginDynamic::initialize() noexcept {
  cublasCreate(&cublas);
  return 0;
}

void TransformerDecodePluginDynamic::terminate() noexcept {
  CHECK(cublasDestroy(cublas));
}

size_t TransformerDecodePluginDynamic::getSerializationSize() const noexcept {
  return sizeof(DataType) + sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mHiddenSize) + sizeof(mBeamWidth) + sizeof(mVocabSize) + sizeof(mStartId) + sizeof(mEndId) + sizeof(mNumLayer);
}

void TransformerDecodePluginDynamic::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, mType);
  serialize_value(&buffer, mNumHeads);
  serialize_value(&buffer, mHeadSize);
  serialize_value(&buffer, mHiddenSize);
  serialize_value(&buffer, mBeamWidth);
  serialize_value(&buffer, mVocabSize);
  serialize_value(&buffer, mStartId);
  serialize_value(&buffer, mEndId);
  serialize_value(&buffer, mNumLayer);
}

void TransformerDecodePluginDynamic::destroy() noexcept {
  delete this;
}

void TransformerDecodePluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* TransformerDecodePluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

int TransformerDecodePluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  // 确定元素个数
  const int B = inputDesc[1].dims.d[0];
  const int L = inputDesc[0].dims.d[1];
  const size_t wordSize = bert::getElementSize(mType);
  // std::cout<< "B:" << B << " L:" << L << " wordSize:"<< wordSize << std::endl;

  cudaError_t status = cudaSuccess;
  if (mType == DataType::kFLOAT) {
    TransformerDecodeNode<float> nodes(mNumLayer, mHiddenSize, inputs, outputs);
    TransformerDecodeTemp<float> temps(B, L, mNumLayer, mHiddenSize, mBeamWidth, mVocabSize, workspace);
    status = transDecode<float>(
               cublas, B, L, mNumHeads, mHeadSize, mBeamWidth, mVocabSize, mStartId, mEndId, mNumLayer,
               nodes, temps, stream);
  } else if(mType == DataType::kHALF) {
    TransformerDecodeNode<half> nodes(mNumLayer, mHiddenSize, inputs, outputs);
    TransformerDecodeTemp<half> temps(B, L, mNumLayer, mHiddenSize, mBeamWidth, mVocabSize, workspace);
    status = transDecode<half>(
               cublas, B, L, mNumHeads, mHeadSize, mBeamWidth, mVocabSize, mStartId, mEndId, mNumLayer,
               nodes, temps, stream);
  } else {
    assert(false);
  }
  // gpu_timer.Stop();
  // printf("transDecode enqueue: %f ", (float(gpu_timer.ElapsedMillis())));

  assert(status == cudaSuccess);
  return status;
}

PluginFieldCollection TransformerDecodePluginDynamicCreator::mFC{};
std::vector<PluginField> TransformerDecodePluginDynamicCreator::mPluginAttributes;

TransformerDecodePluginDynamicCreator::TransformerDecodePluginDynamicCreator() {
  mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("beam_width", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("vocab_size", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("start_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("end_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("num_layer", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TransformerDecodePluginDynamicCreator::getPluginName() const noexcept {
  return TRANSFORMER_DECODE_PLUGIN_NAME;
}

const char* TransformerDecodePluginDynamicCreator::getPluginVersion() const noexcept {
  return TRANSFORMER_DECODE_PLUGIN_VERSION;
}

const PluginFieldCollection* TransformerDecodePluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* TransformerDecodePluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  int typeId = -1;
  int hiddenSize = 0;
  int numHeads = 0;
  int beamWidth = 0;
  int vocabSize = 0;
  int startId = 0;
  int endId = 0;
  int numLayer = 0;

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
    if (field_name.compare("beam_width") == 0) {
      beamWidth = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("vocab_size") == 0) {
      vocabSize = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("start_id") == 0) {
      startId = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("end_id") == 0) {
      endId = *static_cast<const int*>(fc->fields[i].data);
    }
    if (field_name.compare("num_layer") == 0) {
      numLayer = *static_cast<const int*>(fc->fields[i].data);
    }
  }
  std::cout << "typeId:" << typeId << " hidden_size:" << hiddenSize << " num_heads:" << numHeads << " beam_width:" << beamWidth \
            << " vocab_size:" << vocabSize << " start_id:" << startId << " end_id:" << endId \
            << " num_layer:" << numLayer << std::endl;
  DataType type = static_cast<DataType>(typeId);
  TransformerDecodePluginDynamic* p = new TransformerDecodePluginDynamic(name, type, hiddenSize, numHeads, beamWidth, vocabSize,
      startId, endId, numLayer);
  return p;
}

IPluginV2* TransformerDecodePluginDynamicCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call TransformerDecodePluginDynamic::destroy()
  return new TransformerDecodePluginDynamic(name, serialData, serialLength);
}

void TransformerDecodePluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char* TransformerDecodePluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

// REGISTER_TENSORRT_PLUGIN(TransformerDecodePluginDynamicCreator);

