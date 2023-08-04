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
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include "transformerKernels.h"
namespace transformer {

/* *********************************** Debug tools *********************************** */
template <typename T>
__global__
void print_abs_mean_kernel(const T* buf, int size) {
  float sum;
  for(int i = 0; i < size; i++) {
    sum += abs((float)buf[i]);
    // printf("[INFO] buf[%d] %f \n", i, buf[i]);
  }
  printf("mean: %f \n", (float) sum / (float) size);
  printf("sum: %f \n", sum);
}

template <typename T>
__global__
void print_kernel(const T* buf, int size) {
  for(int i = 0; i < size; i++) {
    printf("%f ", (float(buf[i])));
  }
  printf("\n");
}

template <typename T>
void print_first_k(const T* buf, int size, cudaStream_t stream) {
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  print_kernel<<<1, 1, 0, stream>>>(buf, size);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template <typename T>
void print_abs_mean(const T* buf, int size, cudaStream_t stream) {
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
  print_abs_mean_kernel<<<1, 1, 0, stream>>>(buf, size);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());
}

template void print_first_k(const float*, int size, cudaStream_t);
template void print_first_k(const half*, int size, cudaStream_t);
template void print_first_k(const int*, int size, cudaStream_t);

template void print_abs_mean(const float* buf, int size, cudaStream_t stream);
template void print_abs_mean(const half* buf, int size, cudaStream_t stream);
template void print_abs_mean(const int* buf, int size, cudaStream_t stream);

/* **************************** end of Debug tools *********************************** */

template <typename T>
__global__
void im2col1d_gpu_kernel(const T* data_im, T* data_col, const int n, const int channels, const int width, const int seq_len, const int kernel_w, const int pad_w, const int width_col) {
  for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
    const int w_im = index / channels;
    const int c_im = index % channels;
    const int k = w_im / seq_len;

    const T* data_im_ptr = data_im;

    T* data_col_ptr = data_col;
    data_col_ptr += w_im * kernel_w * channels + c_im;

    for (int i = 0; i < kernel_w; ++i) {
      *data_col_ptr = (w_im-pad_w+i >= k*seq_len && w_im-pad_w+i < (k+1)*seq_len) ? data_im_ptr[(w_im-pad_w+i)*channels+c_im] : 0;

      data_col_ptr += channels;
    }
  }
}

template <>
__global__
void im2col1d_gpu_kernel(const half* data_im, half* data_col, const int n, const int channels, const int width, const int seq_len, const int kernel_w, const int pad_w, const int width_col) {
  for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
    const int w_im = index / channels;
    const int c_im = index % channels;
    const int k = w_im / seq_len;

    const half* data_im_ptr = data_im;

    half* data_col_ptr = data_col;
    data_col_ptr += w_im * kernel_w * channels + c_im;

    for (int i = 0; i < kernel_w; ++i) {
      *data_col_ptr = (w_im-pad_w+i >= k*seq_len && w_im-pad_w+i < (k+1)*seq_len) ? data_im_ptr[(w_im-pad_w+i)*channels+c_im] : half(0);

      data_col_ptr += channels;
    }
  }
}

#define BLOCK 512
template <typename T>
void im2col1d_gpu(const T* data_im, T* data_col, const int channels, const int width,
                  const int seq_len, const int kernel_w, const int pad_w, cudaStream_t stream) {
  int width_col = (width + 2 * pad_w - kernel_w) + 1;
  int num_kernels = channels * width_col;

  dim3 grid((num_kernels+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  im2col1d_gpu_kernel<T><<<grid, block, 0, stream>>>(
    data_im, data_col, num_kernels, channels, width, seq_len, kernel_w,
    pad_w, width_col);
}

template <>
void im2col1d_gpu(const half* data_im, half* data_col, const int channels, const int width, const int seq_len, const int kernel_w, const int pad_w, cudaStream_t stream) {
  int width_col = (width + 2 * pad_w - kernel_w) + 1;
  int num_kernels = channels * width_col;

  dim3 grid((num_kernels+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  im2col1d_gpu_kernel<half><<<grid, block, 0, stream>>>(
    data_im, data_col, num_kernels, channels, width, seq_len, kernel_w,
    pad_w, width_col);
}

template void im2col1d_gpu<float>(const float* data_im, float* data_col,
                                  const int channels, const int width, const int seq_len, const int kernel_w, const int pad_w, cudaStream_t stream);

template void im2col1d_gpu<half>(const half* data_im, half* data_col,
                                 const int channels, const int width, const int seq_len, const int kernel_w, const int pad_w, cudaStream_t stream);

template <typename T>
__inline__ __device__
T relu(T x) {
  if(x > 0)
    return x;
  return 0;
}

template <>
__inline__ __device__
half relu(half x) {
  half tmp = __float2half_rn(0.0f);
  if(x > tmp)
    return x;
  return tmp;
}

template <typename T>
__global__
void add_bias_kernel(T* out, const T* bias, int m, int n, bool is_relu) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m) {
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = is_relu ? relu<T>(val) : val;
      row_id += gridDim.x;
    }
  }
}

template <typename T>
void add_bias_kernelLauncher(T* out, const T* bias, int m, int n, bool is_relu, cudaStream_t stream) {
  dim3 grid(m / 4);
  dim3 block(n / 4);
  assert(block.x <= 1024);
  add_bias_kernel<T><<<grid, block, 0, stream>>>(out, bias, m, n, is_relu);
}

template void add_bias_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, bool is_relu, cudaStream_t stream);

template void add_bias_kernelLauncher<half>(
  half* out, const half* bias, int m, int n, bool is_relu, cudaStream_t stream);

template <typename T>
__global__
void layernorm_kernel(T* out, const T* input, const T* gamma, const T* beta, int m, int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x) {
    local_out += (float)input[blockIdx.x * n + i];
    // local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i]);
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
      (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

// template <>
// __global__
// void layernorm_kernel(half* out, const half* input, const half* gamma, const half* beta, int m, int n)
// {
//   int tid = threadIdx.x;
//   __shared__ float s_mean;
//   __shared__ float s_variance;
//   float mean =  0.0f;
//   float variance = 0.0f;
//   float2 local_out_fp2;

//   half2* out_ptr = (half2*)out;
//   const half2* input_ptr = (const half2*)input;
//   const half2* gamma_ptr = (const half2*)gamma;
//   const half2* beta_ptr = (const half2*)beta;

//   float local_out = 0.0f;
//   int id = blockIdx.x * n / 2 + tid;
//   local_out_fp2 = __half22float2(__hadd2(out_ptr[id], input_ptr[id]));
//   local_out += local_out_fp2.x;
//   local_out += local_out_fp2.y;

//   mean = blockReduceSum<float>(local_out);
//   if(threadIdx.x == 0)
//     s_mean = mean / n;
//   __syncthreads();

//   variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
//   variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
//   variance = blockReduceSum<float>(variance);
//   if(threadIdx.x == 0)
//     s_variance = rsqrtf(variance / n + 1e-6f);
//   __syncthreads();

//   float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
//   float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
//   local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
//   local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
//   out_ptr[id] = __float22half2_rn(local_out_fp2);
// }

template <typename T>
void layernorm_kernelLauncher(T* out, const T* input, const T* gamma, const T* beta, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  layernorm_kernel<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n);
}

// template <>
// void layernorm_kernelLauncher(half* out, const half* input, const half* gamma, const half* beta, int m, int n, cudaStream_t stream)
// {
//   dim3 grid(m);
//   dim3 block(n / 2);
//   assert(n / 2 <= 1024);
//   layernorm_kernel<half><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n);
// }

template void layernorm_kernelLauncher(float* out, const float* input_tensor, const float* gamma, const float* beta, int m, int n, cudaStream_t stream);

template void layernorm_kernelLauncher(half* out, const half* input_tensor, const half* gamma, const half* beta, int m, int n, cudaStream_t stream);

template <typename T>
__global__
void add_op_kernel(T* out, const T* input0, const T* input1, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] + input1[i];
}

template <>
__global__
void add_op_kernel(half* out, const half* input0, const half* input1, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] + input1[i];
}

template <typename T>
void add_op(T* out, const T* input0, const T* input1, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  add_op_kernel<T><<<grid, block, 0, stream>>>(out, input0, input1, size);
}

template <>
void add_op(half* out, const half* input0, const half* input1, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  add_op_kernel<half><<<grid, block, 0, stream>>>(out, input0, input1, size);
}

template void add_op(float* out, const float* input0, const float* input1, int size, cudaStream_t stream);
template void add_op(half* out, const half* input0, const half* input1, int size, cudaStream_t stream);

template <typename T>
__global__
void mul_op_kernel(T* out, const T* input0, const T* input1, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] * input1[i];
}

template <>
__global__
void mul_op_kernel(half* out, const half* input0, const half* input1, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] * input1[i];
}

template <typename T>
void mul_op(T* out, const T* input0, const T* input1, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  mul_op_kernel<T><<<grid, block, 0, stream>>>(out, input0, input1, size);
}

template <>
void mul_op(half* out, const half* input0, const half* input1, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  mul_op_kernel<half><<<grid, block, 0, stream>>>(out, input0, input1, size);
}

template void mul_op(float* out, const float* input0, const float* input1, int size, cudaStream_t stream);
template void mul_op(half* out, const half* input0, const half* input1, int size, cudaStream_t stream);


template <typename T>
__global__
void add_mask_op_kernel(T* out, const T* input0, const T* input1, const T* mask, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] + input1[i] * mask[i];
}

template <>
__global__
void add_mask_op_kernel(half* out, const half* input0, const half* input1, const half* mask, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size)
    out[i] = input0[i] + input1[i] * mask[i];
}

template <typename T>
void add_mask_op(T* out, const T* input0, const T* input1, const T* mask, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  add_mask_op_kernel<T><<<grid, block, 0, stream>>>(out, input0, input1, mask, size);
}

template <>
void add_mask_op(half* out, const half* input0, const half* input1, const half* mask, int size, cudaStream_t stream) {
  dim3 grid((size+BLOCK-1)/BLOCK);
  dim3 block(BLOCK);
  add_mask_op_kernel<half><<<grid, block, 0, stream>>>(out, input0, input1, mask, size);
}

template void add_mask_op(float* out, const float* input0, const float* input1, const float* mask, int size, cudaStream_t stream);
template void add_mask_op(half* out, const half* input0, const half* input1, const half* mask, int size, cudaStream_t stream);

template <typename T>
__inline__ __device__
T gelu(T x) {
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__global__
void add_bias_act(T* out, const T* bias, int m, int n) {
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m) {
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__
void add_bias_act(half* out, const half* bias, int m, int n) {
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
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = gelu<half2>(val);
      row_id += gridDim.x;
    }
  }
}

template <typename T>
__global__
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] =
      (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <>
__global__
void add_bias_input_layernorm(half* out, const half* input, const half* bias,
                              const half* gamma, const half* beta, int m, int n) {

  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid;
  local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream) {
//  dim3 grid(m / 64);
  dim3 grid(m / 4);
  dim3 block(n / 4);
  assert(block.x <= 1024);
//  dim3 block(n);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input, const T* bias,
    const T* gamma, const T* beta, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}


template <>
void add_bias_input_layernorm_kernelLauncher(half* out, const half* input, const half* bias,
    const half* gamma, const half* beta, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n / 2);
  assert(n / 2 <= 1024);
  add_bias_input_layernorm<half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

template<typename T>
__global__
void broadcast_kernel(T* log_probs, T* cum_log_probs, const int batch_size, const int beam_width, const int vocab_size, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = tid / vocab_size;

  if(tid < N) {
    log_probs[tid] += cum_log_probs[bid];
    // 确保-inf时 topk函数不会出现问题，-inf限制为-100000，且5*6412个值都不一样，topk不会出现歧义
    // todo 可能存在风险
    log_probs[tid] = (log_probs[tid] <= -100000) ? (-100000 + tid * 0.0001) : log_probs[tid];
  }
}

void broadcast_kernelLauncher(float* log_probs, float* cum_log_probs, const int batch_size, const int beam_width,
                              const int vocab_size, cudaStream_t stream) {

  int N = batch_size * beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  broadcast_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, N);
  // print_first_k(log_probs, 100, stream);
}

template <typename T>
__global__
void topK_kernel(const T* log_probs, int* ids, const int batch_size, const int N, const int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val, max_val;
  __shared__ float s_max_val;
  for(int ite = 0; ite < batch_size; ++ite) {
    bool choosed = false;
    val = (tid < N ) ? (float)log_probs[ite * N + tid] : -1e20f;

    for(int kids = 0; kids < K; ++kids) {
      max_val = blockReduceMax<float>(val);

      if(threadIdx.x == 0)
        s_max_val = max_val;
      __syncthreads();

      if(s_max_val == val && !choosed && tid < N) {
        ids[ite * gridDim.x * K + blockIdx.x * K + kids] = tid + ite * N;
        val = -1e20f;
        choosed = true;
      }
    }
  }
}

template <typename T>
__global__
void topK_kernel_2nd(const T* log_probs, int* ids, const int batch_size, const int N, const int K, const int id_offset) {
  // log_probs, ids, batch_size, beam_width * grid.x, beam_width, beam_width * vocab_size
  int tid = threadIdx.x;
  float val, max_val;
  __shared__ float s_max_val;
  __shared__ int beam_index;
  __shared__ int ids_before_sort[16];

  for(int ite = 0; ite < batch_size; ++ite) {
    bool choosed = false;
    const int id = (tid < N) ? ids[ite * N + tid] : -1;
    val = (tid < N) ? (float)log_probs[id] : -1e20f;

    __syncthreads();

    if(tid == 0) beam_index = 0;
    if(tid < 16) ids_before_sort[tid] = -1;

    __syncthreads();
    while(beam_index < K) {
      int begin_beam_index = beam_index;
      max_val = blockReduceMax<float>(val);
      if(threadIdx.x == 0) {
        s_max_val = max_val;
      }
      __syncthreads();
      if(s_max_val == val && !choosed && id != -1) {
        int id_offset_ = atomicAdd(&beam_index, 1);
        ids_before_sort[id_offset_] = id;
        val = -1e20f;
        choosed = true;
      }
      __syncthreads();

      // simply sort the ids
      if(threadIdx.x == 0 && beam_index - begin_beam_index > 1) {
        for(int i = begin_beam_index; i < beam_index; i++) {
          for(int j = i; j < beam_index; j++) {
            if(ids_before_sort[j] < ids_before_sort[i]) {
              int tmpid = ids_before_sort[j];
              ids_before_sort[j] = ids_before_sort[i];
              ids_before_sort[i] = tmpid;
            }
          }
        }
      }
    }
    __syncthreads();
    if(tid < K) ids[ite * K + tid] = ids_before_sort[tid];
    __syncthreads();
  }
}

void topK(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
          cudaStream_t stream) {
  int N = beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  /* First round topK, for each batch, get grid.x * K values */
  topK_kernel<float><<<grid, block, 0, stream>>>(log_probs, ids, batch_size, N, beam_width);
  /*Second round, for each batch, get the final TopK values out from grid.x * K values. */
  // print_first_k(ids, 32*5, stream);
  topK_kernel_2nd<float><<<1, block, 0, stream>>>(log_probs, ids, batch_size, beam_width * grid.x, beam_width, N);
  // print_first_k(ids, 5, stream);
}

template <typename T>
__global__
void update_kernel(T* log_probs, T* cum_log_probs,
                   int* ids, bool* finished,
                   int* parent_ids, int* sequence_length,
                   int* word_ids, int* output_ids,
                   const int batch_size, const int beam_width,
                   const int vocab_size, const int end_id,
                   int* finished_count) {
  int tid = threadIdx.x;
  sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

  int beam_id = ids[tid];
  beam_id /= vocab_size;
  int word_id = ids[tid];
  word_id %= vocab_size;

  cum_log_probs[tid] = log_probs[ids[tid]];
  sequence_length[tid] = sequence_length[beam_id];
  // 等效finished = tf.gather(finished, beam_indices)和tf.logical_or(finished, tf.equal(output_id, decoding_args.end_id))
  finished[tid] = word_id == end_id ? 1 : 0;
  parent_ids[tid] = beam_id;
  word_ids[tid] = word_id;
  output_ids[tid] = word_id;

  // TODO use reduce sum to compute how many sentence are finished
  // int fi = finished[tid]
  // int total_finish = reduceSum(fi);
}

template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
                                        const int hidden_units, T* from_tensor) {
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  int index = word_ids[blockIdx.x];
  if (index == 0)
    from_tensor[write_pos] = 0.0f;
  else
    from_tensor[write_pos] = embedding_table[index * hidden_units + threadIdx.x];
}

void update(float* log_probs, float* cum_log_probs,
            int* ids, bool* finished,
            int* parent_ids, int* sequence_length,
            int* word_ids, int* output_ids,
            const int batch_size, const int beam_width,
            const int vocab_size, cudaStream_t stream,
            const int end_id, int* finished_count) {

  dim3 grid(1);
  dim3 block(batch_size * beam_width);

  assert(block.x <= 1024);

  update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, ids,
      finished, parent_ids, sequence_length,
      word_ids, output_ids, batch_size,
      beam_width, vocab_size, end_id,
      finished_count);
  // print_first_k(ids, 10, stream);
  // print_first_k(parent_ids, 10, stream);
  // print_first_k(word_ids, 10, stream);
}

template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
                      const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream) {
  dim3 grid(batch_size * beam_width);
  dim3 block(hidden_units);
  assert(hidden_units <= 1024);
  embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}


template <typename T>
__global__ void update_logits_kernel(T* logits, const T* bias, const int end_id, const bool* finished, const int n) {
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

template <typename T>
__global__ void update_logits_kernel_v2(T* logits, const int end_id, const bool* finished, const int n) {
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

void update_logits(float* logits, const float* bias, const int end_id, const bool* finished,
                   const int m, const int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel<float><<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}

void update_logits_v2(float* logits, const int end_id, const bool* finished,
                      const int m, const int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel_v2<float><<<grid, block, 0, stream>>>(logits, end_id, finished, n);
}

template <typename T>
__global__ void init_kernel(bool* finished, int* sequence_length, int* word_ids, T* cum_log_probs, const int sentence_id, const int n, const int beam_width) {
  int tid = threadIdx.x;
  finished[tid] = false;
  sequence_length[tid] = 0;
  word_ids[tid] = sentence_id;
  cum_log_probs[tid] = (T)(tid % beam_width == 0 ? 0.0f: -1e20f);
}

template <typename T>
__global__ void update_KV_cache_kernel(
  T* key_src_cache, T* key_tgt_cache,
  T* value_src_cache, T* value_tgt_cache,
  const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim, const int cache_size, const int step, const int decoder_layers) {
  int layer_id = blockIdx.x / batch_size / beam_width / step;
  int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
  int beam_id = (blockIdx.x % (beam_width * step)) / step;
  int step_id = blockIdx.x % step;

  int hidden_id = step_id * batch_size * beam_width * hidden_dim +
                  beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

  int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim +
                      batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

  T* key_src_ptr = key_src_cache + layer_id * cache_size;
  T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
  T* value_src_ptr = value_src_cache + layer_id * cache_size;
  T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;


  for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
    key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
    value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
  }

}
template <typename T>
void update_KV_cache(T** key_cache, T** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
                     const int step, const int cache_size, const int decoder_layers, cudaStream_t stream) {
  dim3 grid(decoder_layers * batch_size * beam_width * step);
  dim3 block(min(1024, hidden_dim));

  int src_id = step & 0x1;
  int tgt_id = 1 - src_id;

  update_KV_cache_kernel<<<grid, block, 0, stream>>>(
    key_cache[src_id], key_cache[tgt_id],
    value_cache[src_id], value_cache[tgt_id],
    beam_ids, batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
}

__global__ void init_top_kernel(int* topk_ids_buf) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  topk_ids_buf[idx] = 0;
}

void init(bool* finished, int* sequence_length, int* word_ids, float* cum_log_probs, int* topk_ids_buf, const int sentence_id,
          const int batch_size, const int beam_width, const int vocabSize, cudaStream_t stream) {
  dim3 grid(1);
  dim3 block(min(1024, batch_size * beam_width));
  assert(batch_size * beam_width <= 1024);
  init_kernel<float><<<grid, block, 0, stream>>>(finished, sequence_length, word_ids, cum_log_probs, sentence_id, batch_size * beam_width, beam_width);

  int topk_ids_buf_size = batch_size * beam_width * (ceil)((beam_width * vocabSize * 1.0) / 1024.0);
  int thread_num = min(1024, topk_ids_buf_size);
  dim3 top_grid((topk_ids_buf_size - 1) / thread_num + 1);
  dim3 top_block(thread_num);
  init_top_kernel<<<top_grid, top_block, 0, stream>>>(topk_ids_buf);
}

template<typename T>
__global__
void sine_position_encoder_kernel(T* output, int step, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;

  // input = input * hidden_dim**0.5
  output[bid * n + tid] = output[bid * n + tid] * (T)sqrtf(float(n));

  float log_timescale_increment = __logf(10000) / (half_n - 1.f);
  float inv_timescales = __expf( (tid % (int)half_n) * -1 * log_timescale_increment );
  float scaled_time = inv_timescales * step;

  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;
}

template<typename T>
void sine_position_encoder(
  T* output,
  int step,
  int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  sine_position_encoder_kernel<T><<<grid, block, 0, stream>>>(output, step, n);
}

template void add_bias_act_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta,
  int m, int n, cudaStream_t stream);

template void add_bias_act_kernelLauncher<half>(
  half* out, const half* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<half>(
  half* out, const half* input, const half* bias, const half* gamma, const half* beta,
  int m, int n, cudaStream_t stream);

template void embedding_lookup(const float* embedding_table, const int* word_ids, float* from_tensor,
                               const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void embedding_lookup(const half* embedding_table, const int* word_ids, half* from_tensor,
                               const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void update_KV_cache(float** key_cache, float** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
                              const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void update_KV_cache(half** key_cache, half** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
                              const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void sine_position_encoder(
  float* output,
  int step,
  int m, int n,
  cudaStream_t stream);

template void sine_position_encoder(
  half* output,
  int step,
  int m, int n,
  cudaStream_t stream);

}//namespace

