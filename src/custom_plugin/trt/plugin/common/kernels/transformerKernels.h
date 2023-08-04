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
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <stdexcept>
#include <chrono>
#include "cublas_v2.h"

namespace transformer {

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + \
                             (_cudaGetErrorEnum(result)) + " " + file +  \
                             ":" + std::to_string(line) + " \n");
    \
  }
}

#define FINAL_MASK 0xffffffff
#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__
T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__inline__ __device__
T warpReduceMax(T val) {
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__
T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}

template <typename T>
void print_first_k(const T* buf, int size, cudaStream_t stream);

template <typename T>
void print_abs_mean(const T* buf, int size, cudaStream_t stream);

template <typename T>
void im2col1d_gpu(const T* data_im, T* data_col, const int channels, const int width,
                  const int seq_len, const int kernel_w, const int pad_w, cudaStream_t stream);

template <typename T>
void add_bias_kernelLauncher(T* out, const T* bias, int m, int n, bool is_relu, cudaStream_t stream);

template <typename T>
void layernorm_kernelLauncher(T* out, const T* input_tensor, const T* gamma, const T* beta, int m, int n, cudaStream_t stream);

template <typename T>
void add_op(T* out, const T* input0, const T* input1, int size, cudaStream_t stream);

template <typename T>
void mul_op(T* out, const T* input0, const T* input1, int size, cudaStream_t stream);

template <typename T>
void add_mask_op(T* out, const T* input0, const T* input1, const T* mask, int size, cudaStream_t stream);

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input_tensor,
    const T* bias, const T* gamma,
    const T* beta, int m, int n,
    cudaStream_t stream);

void broadcast_kernelLauncher(float* log_probs, float* cum_log_probs,
                              const int batch_size, const int beam_width,
                              const int vocab_size, cudaStream_t stream);

void topK(const float* log_probs, int* ids, const int batch_size,
          const int beam_width, const int vocab_size, cudaStream_t stream);

void update(float* log_probs, float* cum_log_probs, int* ids,
            bool* finished, int* parent_ids, int* sequence_length,
            int* word_ids, int* output_ids,
            const int batch_size, const int beam_width,
            const int vocab_size, cudaStream_t stream,
            const int end_id,
            int* finished_count);

template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids,
                      T* from_tensor, const int batch_size,
                      const int beam_width, const int hidden_units,
                      cudaStream_t stream);

void update_logits(float* logits, const float* bias, const int end_ids,
                   const bool* finished, const int m, const int n,
                   cudaStream_t stream);

void update_logits_v2(float* logits, const int end_ids,
                      const bool* finished, const int m, const int n,
                      cudaStream_t stream);

void init(bool* finished, int* sequence_length, int* word_ids,
          float* cum_log_probs, int* topk_ids_buf, const int sentence_id,
          const int batch_size, const int beam_width, const int vocabSize, cudaStream_t stream);

template <typename T>
void update_KV_cache(T** key_cache, T** value_cache, const int* beam_ids,
                     const int batch_size, const int beam_width,
                     const int hidden_dim, const int step,
                     const int cache_size, const int decoder_layers,
                     cudaStream_t stream);

template <typename T>
void sine_position_encoder(T* output, int step, int m, int n, cudaStream_t stream);

}//namespace transformer_decode
