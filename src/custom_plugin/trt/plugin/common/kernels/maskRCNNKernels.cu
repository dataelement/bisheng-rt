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

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

#include "maskRCNNKernels.h"
#include "plugin.h"
#include <NvInfer.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <time.h>
#include <chrono>

#define DUBUG_KERNEL 0
#define DUBUG_BATCH 0
#define DEBUG_T 1

#define dMIN(a, b) ((a) < (b) ? (a) : (b))
#define dMAX(a, b) ((a) > (b) ? (a) : (b))
#define dCLAMP(x, xMin, xMax) ((x) > (xMin) ? ((x) < (xMax) ? (x) : (xMax)) : (xMin))

/* *********************************** Debug tools *********************************** */
#define PRINT_FUNC_NAME_() do{\
  std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
} while (0)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

// static const char *_cudaGetErrorEnum(cublasStatus_t error) {
//   switch (error) {
//   case CUBLAS_STATUS_SUCCESS:
//     return "CUBLAS_STATUS_SUCCESS";

//   case CUBLAS_STATUS_NOT_INITIALIZED:
//     return "CUBLAS_STATUS_NOT_INITIALIZED";

//   case CUBLAS_STATUS_ALLOC_FAILED:
//     return "CUBLAS_STATUS_ALLOC_FAILED";

//   case CUBLAS_STATUS_INVALID_VALUE:
//     return "CUBLAS_STATUS_INVALID_VALUE";

//   case CUBLAS_STATUS_ARCH_MISMATCH:
//     return "CUBLAS_STATUS_ARCH_MISMATCH";

//   case CUBLAS_STATUS_MAPPING_ERROR:
//     return "CUBLAS_STATUS_MAPPING_ERROR";

//   case CUBLAS_STATUS_EXECUTION_FAILED:
//     return "CUBLAS_STATUS_EXECUTION_FAILED";

//   case CUBLAS_STATUS_INTERNAL_ERROR:
//     return "CUBLAS_STATUS_INTERNAL_ERROR";

//   case CUBLAS_STATUS_NOT_SUPPORTED:
//     return "CUBLAS_STATUS_NOT_SUPPORTED";

//   case CUBLAS_STATUS_LICENSE_ERROR:
//     return "CUBLAS_STATUS_LICENSE_ERROR";
//   }
//   return "<unknown>";
// }

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + \
                             (_cudaGetErrorEnum(result)) + " " + file +  \
                             ":" + std::to_string(line) + " \n");
    \
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void print_to_file(T* result, const int size, char* file) {
  FILE* fd = fopen(file, "w");
  float* tmp = (float*)malloc(sizeof(float) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for(int i = 0; i < size; ++i)
    fprintf(fd, "%f\n", (float)tmp[i]);
  free(tmp);
  fclose(fd);
}

template <typename T>
void print_to_screen(T* result, const int size) {
  float* tmp = (float*)malloc(sizeof(float) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for(int i = 0; i < size; ++i) {
    printf("%d, %f\n", i, (float)tmp[i]);
  }

  free(tmp);
}

template<typename T>
void check_max_val(const T* result, const int size) {
  T* tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float max_val = -100000;
  for(int i = 0 ; i < size; i++) {
    float val = (float)(tmp[i]);
    if(val > max_val) max_val = val;
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

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
    // printf("%d ", (int(buf[i])));
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

template <typename BoxType>
struct BBoxT {
  BoxType y1, x1, y2, x2;
};

template <typename BoxType>
struct OcrBBoxT {
  BoxType x1, y1, x2, y2;
};


template <typename DType>
__global__ void argMaxReset_kernel(
  int samples, int NClass, const DType* in_scores, const int* maxIdx, DType* out_scores) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int max_idx = samples * NClass;
  if (idx >= max_idx)
    return;

  int sampleIdx = idx / NClass;
  int classIdx = idx % NClass;
  if (classIdx != maxIdx[sampleIdx])
    out_scores[idx] = 0;
  else
    out_scores[idx] = in_scores[idx];
}

template <typename DType>
struct ScanItem {
  DType data;
  int idx;
};

template <typename DType>
struct GreaterItem {
  __host__ __device__ __forceinline__ ScanItem<DType> operator()(
    const ScanItem<DType>& a, const ScanItem<DType>& b) const {
    return (a.data > b.data ? a : b);
  }
};

template <typename DType>
__global__ void resetMemValue_kernel(void* outPtr, int samples, float val) {
  DType* out = static_cast<DType*>(outPtr);
  int loop = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < samples; idx += loop) {
    out[idx] = (DType) val;
  }
}

// blockDim.x : NClass
// GroupDim.x : sample count
// GroupDim.y : batch N
// outScore : DType[ N * sample * 1 ]
// outLabel : int[ N * sample * 1 ]
// outBbox : int[ N * sample * 4 ]
template <typename DType, typename BoxType, int Threads = 32>
__global__ void argMaxGroup_kernel(int samples, int start_class_id, int NClass, const void* inScorePtr, const void* inBboxPtr,
                                   const void* validSampleCountPtr, void* outScorePtr, void* outLabelPtr, void* outBboxPtr) {
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const BoxType* inBbox = static_cast<const BoxType*>(inBboxPtr);
  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  DType* outScore = static_cast<DType*>(outScorePtr);
  BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);
  BoxType* outBbox = static_cast<BoxType*>(outBboxPtr);

  const int N = blockIdx.y;
  const int validSamples = validSampleCount[N];

  typedef ScanItem<DType> ScanItemD;
  typedef cub::BlockReduce<ScanItemD, Threads> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int iSample = blockIdx.x; iSample < validSamples; iSample += gridDim.x) {
    int classOffset = (N * samples + iSample) * NClass; // start from [batch, count, class0]
    // total IPerThread * blockDim
    ScanItemD maxItem = {0.0f, -1};
    for (int i = start_class_id; i < NClass; i += Threads) {
      int curIdx = i + threadIdx.x;
      ScanItemD item = {0.0f, -1};
      if (curIdx < NClass) {
        item.data = inScore[classOffset + curIdx];
        item.idx = curIdx;
      }
      const int validNum = (NClass - i > Threads ? Threads : NClass - i);
      ScanItemD aggregate = BlockReduce(temp_storage).Reduce(item, GreaterItem<DType>(), validNum);
      __syncthreads();
      if (aggregate.data > maxItem.data) {
        maxItem = aggregate;
      }
#if DUBUG_KERNEL
      if (N == DUBUG_BATCH && threadIdx.x == 0 && iSample < 15 /*&& maxItem.idx >= 32*/) {
        printf("argMaxGroup N:%d, iSample:%d, maxItem(score:%.3f, idx:%d)validReduceNum:%d\n", N, iSample,
               (float) maxItem.data, maxItem.idx, validNum);
      }
#endif
    }

    const int dstOffset = N * samples + iSample;
    if (threadIdx.x == 0) {
      outScore[dstOffset] = maxItem.data;
      outLabel[dstOffset] = (BoxType) maxItem.idx;
      outBbox[dstOffset * 4] = inBbox[(classOffset + maxItem.idx) * 4];
      outBbox[dstOffset * 4 + 1] = inBbox[(classOffset + maxItem.idx) * 4 + 1];
      outBbox[dstOffset * 4 + 2] = inBbox[(classOffset + maxItem.idx) * 4 + 2];
      outBbox[dstOffset * 4 + 3] = inBbox[(classOffset + maxItem.idx) * 4 + 3];
    }
  }
}

// blockDim.x : NClass
// GroupDim.x : sample count
// GroupDim.y : batch N
// outScore : DType[ N * sample * 1 ]
// outLabel : int[ N * sample * 1 ]
template <typename DType, typename BoxType, int Threads = 32>
__global__ void argOcrMaxGroup_kernel(int samples, int start_class_id, int NClass, const void* inScorePtr,
                                      const void* validSampleCountPtr, void* outScorePtr, void* outLabelPtr) {
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  DType* outScore = static_cast<DType*>(outScorePtr);
  BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);

  const int N = blockIdx.y;
  const int validSamples = validSampleCount[N];

  typedef ScanItem<DType> ScanItemD;
  typedef cub::BlockReduce<ScanItemD, Threads> BlockReduce;

  for (int iSample = blockIdx.x; iSample < validSamples; iSample += gridDim.x) {
    const int dstOffset = N * samples + iSample;
    int classOffset = (N * samples + iSample) * NClass; // start from [batch, count, class0]
    // total IPerThread * blockDim
    for (int i = start_class_id; i < NClass; i += Threads) {
      int curIdx = i + threadIdx.x;
      // 取出第一个类别的分数，ocr目前只有两类，背景和text
      if (threadIdx.x == 1) {
        outScore[dstOffset] = inScore[classOffset + curIdx];;
        outLabel[dstOffset] = (BoxType) curIdx;
      }
    }
  }
}

struct BlockClassSumPrefix {
  int total;
  // Constructor
  __device__ BlockClassSumPrefix()
    : total(0) {
  }
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ int operator()(int aggregate) {
    int old = total;
    total += aggregate;
    return old;
  }
};

#define LabelShift (DType)(2.5f)
#define MinValidScore (DType)(0.00f)

template <typename DType>
__device__ __forceinline__ DType getKey(DType score, int lable, int NClass) {
  return (lable < 0 ? (DType) 0 : ((DType)(NClass - lable - 1) * LabelShift + score + MinValidScore));
}

template <typename DType>
__device__ __forceinline__ DType getOcrKey(DType score, int lable, int NClass) {
  return (lable < 0 ? (DType) 0 : ((DType) score));
}

template <typename DType, typename BoxType>
__device__ __forceinline__ void getScoreLable(DType key, int NClass, DType& score, BoxType& lable) {
  int i = key / LabelShift;
  score = (key <= MinValidScore ? (DType) 0 : key - (DType) i * LabelShift - MinValidScore);
  score = dCLAMP(score, (DType) 0, (DType) 1.0);
  lable = (BoxType)(key <= MinValidScore ? -1 : (NClass - i - 1));
}

template <typename DType, typename BoxType>
__device__ __forceinline__ void getOcrScoreLable(DType key, int NClass, DType& score, BoxType& lable) {
  float eps = 0.001f;
  int i = 0;
  score = key;
  lable = (BoxType)(abs(key + 100000.0f) <= eps ? -1 : (NClass - i - 1));
  // lable = (NClass - i - 1);
}

// blockDim.x : threads
// gridDim.x : batch N
// validSampleCount INPUT : int [N]
// classStartPos OUTPUT: int [N * (Class + 1)], need memset to zero before this kernel
// outScore OUTPUT : DType [N * samples]
// outLabel OUTPUT : int [N * samples]
// outSampleIdx OUTPUT : int [N * samples]
// outValidSampleCount : int [N]
// IPerThread * Threads >= sample-count
#define MaxClassNum 255
template <typename DType, typename BoxType, int Threads = 256, int IPerThread = 4>
__global__ void sortPerClass_kernel(
  // int N,
  int samples, int NClass, int background, float scoreThreshold, const void* validSampleCountPtr,
  const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, void* classStartPosPtr, void* outScorePtr,
  void* outLabelPtr, void* outSampleIdxPtr, void* outValidSampleCountPtr) {
  typedef cub::BlockExchange<DType, Threads, IPerThread> BlockExchangeKey;
  typedef cub::BlockExchange<int, Threads, IPerThread> BlockExchangeI;
  typedef cub::BlockRadixSort<DType, Threads, IPerThread, int> BlockRadixSort;
  typedef cub::BlockScan<int, Threads> BlockScanClass;
  typedef OcrBBoxT<BoxType> OcrBBox;
  __shared__ union {
    typename BlockExchangeKey::TempStorage storageKey;
    typename BlockExchangeI::TempStorage storageI;
    typename BlockRadixSort::TempStorage storageSort;
    typename BlockScanClass::TempStorage storageScan;
  } temp_storage;
  __shared__ int smemClassCount[MaxClassNum];
  assert(NClass < MaxClassNum);
  assert(IPerThread * Threads >= samples);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
  int* classStartPos = static_cast<int*>(classStartPosPtr);
  DType* outScore = static_cast<DType*>(outScorePtr);
  BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);
  int* outSampleIdx = static_cast<int*>(outSampleIdxPtr);
  int* outValidSampleCount = static_cast<int*>(outValidSampleCountPtr);
  const OcrBBox* inBbox = static_cast<const OcrBBox*>(inBboxPtr);

  for (int s = threadIdx.x; s < NClass + 1; s += blockDim.x) {
    smemClassCount[s] = 0;
  }

  float eps = 0.00001f;
  int N = blockIdx.x;
  int blockOffset = N * samples;
  int validSamples = validSampleCount[N];
  DType key[IPerThread];
  int iSample[IPerThread];
  for (int i = 0; i < IPerThread; ++i) {
    iSample[i] = -1;
    key[i] = -1.0f;
    int curIdx = i * Threads + threadIdx.x;
    if (curIdx < validSamples) {
      int label = (int) (inLabel[blockOffset + curIdx]);
      DType score = inScore[blockOffset + curIdx];
      OcrBBox box = inBbox[blockOffset + curIdx];
      float box_area = (box.x2 - box.x1) * (box.y2 - box.y1);
      // 过滤掉背景，和前景分数小于scoreThreshold的box和area不等于0的box
      if (label != background && label != -1 && score >= (DType) scoreThreshold && abs(box_area) > eps) {
        // 通过LabelShift来区分不同类别的分数，label小的分数越高
        key[i] = getKey(score, label, NClass);
        iSample[i] = curIdx;
      }
    }
  }

  BlockExchangeKey(temp_storage.storageKey).StripedToBlocked(key);
  __syncthreads();
  BlockExchangeI(temp_storage.storageI).StripedToBlocked(iSample);
  __syncthreads();
  // 对不同类别的分数进行排序，同一类别的box分配在一起
  BlockRadixSort(temp_storage.storageSort).SortDescendingBlockedToStriped(key, iSample);
  __syncthreads();

  // store Idx
  cub::StoreDirectStriped<Threads>(threadIdx.x, outSampleIdx + blockOffset, iSample, validSamples);
  BoxType lable[IPerThread];
  DType score[IPerThread];

#pragma unroll
  for (int i = 0; i < IPerThread; ++i) {
    getScoreLable(key[i], NClass, score[i], lable[i]);
  }
  cub::StoreDirectStriped<Threads>(threadIdx.x, outScore + blockOffset, score, validSamples);
  cub::StoreDirectStriped<Threads>(threadIdx.x, outLabel + blockOffset, lable, validSamples);

  // final
  for (int i = 0; i < IPerThread; ++i) {
    if (lable[i] >= (BoxType) 0) {
      atomicAdd(&smemClassCount[(int) lable[i]], 1);
    }
  }
  __syncthreads();

  int classBlockOffset = N * (NClass + 1); // Exclusive-sum, 1st is 0, last is final sum

#if DUBUG_KERNEL
  if (N == DUBUG_BATCH && threadIdx.x == 0) {
    printf("sortPerClass(N:%d) final count of each label, valid samples:%d\n", N, validSamples);
    for (int k = 0; k < NClass; ++k) {
      if (smemClassCount[k] > 0)
        printf("Batch:%d, L:%d, count:%d, \n", N, k, smemClassCount[k]);
    }
  }
  __syncthreads();
#endif

  BlockClassSumPrefix sumPrefix;
  for (int s = 0; s < NClass; s += blockDim.x) {
    // s start from block
    int iClassSamples = 0;
    int iClass = s + threadIdx.x;
    if (iClass < NClass) {
      iClassSamples = smemClassCount[iClass];
    }
    BlockScanClass(temp_storage.storageScan).ExclusiveSum(iClassSamples, iClassSamples, sumPrefix);
    __syncthreads();
    if (iClass < NClass) {
      classStartPos[classBlockOffset + iClass] = iClassSamples;
    }
  }
  if (threadIdx.x == 0) {
    classStartPos[classBlockOffset + NClass] = sumPrefix.total;
    assert(sumPrefix.total <= validSamples); // background data removed.
    outValidSampleCount[N] = sumPrefix.total;
#if DUBUG_KERNEL
    if (N == DUBUG_BATCH)
      printf("After sortPerClass, batch:%d valid samples total:%d\n", N, sumPrefix.total);
#endif
  }
}

// blockDim.x : threads
// gridDim.x : batch N
// validSampleCount INPUT : int [N]
// classStartPos OUTPUT: int [N * (Class + 1)], need memset to zero before this kernel
// outScore OUTPUT : DType [N * samples]
// outLabel OUTPUT : int [N * samples]
// outSampleIdx OUTPUT : int [N * samples]
// outValidSampleCount : int [N]
// IPerThread * Threads >= sample-count
#define MaxClassNum 255
template <typename DType, typename BoxType, int Threads = 256, int IPerThread = 4>
__global__ void sortOcrPerClass_kernel(
  // int N,
  int samples, int NClass, int background, float scoreThreshold, const void* validSampleCountPtr,
  const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, void* classStartPosPtr, void* outScorePtr,
  void* outLabelPtr, void* outSampleIdxPtr, void* outValidSampleCountPtr) {
  typedef cub::BlockExchange<DType, Threads, IPerThread> BlockExchangeKey;
  typedef cub::BlockExchange<int, Threads, IPerThread> BlockExchangeI;
  typedef cub::BlockRadixSort<DType, Threads, IPerThread, int> BlockRadixSort;
  typedef cub::BlockScan<int, Threads> BlockScanClass;
  __shared__ union {
    typename BlockExchangeKey::TempStorage storageKey;
    typename BlockExchangeI::TempStorage storageI;
    typename BlockRadixSort::TempStorage storageSort;
    typename BlockScanClass::TempStorage storageScan;
  } temp_storage;
  __shared__ int smemClassCount[MaxClassNum];
  assert(NClass < MaxClassNum);
  assert(IPerThread * Threads >= samples);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
  int* classStartPos = static_cast<int*>(classStartPosPtr);
  DType* outScore = static_cast<DType*>(outScorePtr);
  BoxType* outLabel = static_cast<BoxType*>(outLabelPtr);
  int* outSampleIdx = static_cast<int*>(outSampleIdxPtr);
  int* outValidSampleCount = static_cast<int*>(outValidSampleCountPtr);

  for (int s = threadIdx.x; s < NClass + 1; s += blockDim.x) {
    smemClassCount[s] = 0;
  }

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int validSamples = validSampleCount[N];
  DType key[IPerThread];
  int iSample[IPerThread];
  for (int i = 0; i < IPerThread; ++i) {
    iSample[i] = -1;
    // key[i] = -1.0f;
    key[i] = -100000.0f;
    int curIdx = i * Threads + threadIdx.x;
    if (curIdx < validSamples) {
      int label = (int) (inLabel[blockOffset + curIdx]);
      DType score = inScore[blockOffset + curIdx];
      if (label != background && label != -1 && score >= (DType) scoreThreshold) {
        key[i] = getOcrKey(score, label, NClass);
        iSample[i] = curIdx;
      }
    }
  }

  BlockExchangeKey(temp_storage.storageKey).StripedToBlocked(key);
  __syncthreads();
  BlockExchangeI(temp_storage.storageI).StripedToBlocked(iSample);
  __syncthreads();
  // 对分数进行排序
  BlockRadixSort(temp_storage.storageSort).SortDescendingBlockedToStriped(key, iSample);
  __syncthreads();

  // store Idx
  cub::StoreDirectStriped<Threads>(threadIdx.x, outSampleIdx + blockOffset, iSample, validSamples);
  BoxType lable[IPerThread];
  DType score[IPerThread];

#pragma unroll
  for (int i = 0; i < IPerThread; ++i) {
    getOcrScoreLable(key[i], NClass, score[i], lable[i]);
  }
  cub::StoreDirectStriped<Threads>(threadIdx.x, outScore + blockOffset, score, validSamples);
  cub::StoreDirectStriped<Threads>(threadIdx.x, outLabel + blockOffset, lable, validSamples);

  // final
  for (int i = 0; i < IPerThread; ++i) {
    if (lable[i] >= (BoxType) 0) {
      atomicAdd(&smemClassCount[(int) lable[i]], 1);
    }
  }
  __syncthreads();

  int classBlockOffset = N * (NClass + 1); // Exclusive-sum, 1st is 0, last is final sum

#if DUBUG_KERNEL
  if (N == DUBUG_BATCH && threadIdx.x == 0) {
    printf("sortPerClass(N:%d) final count of each label, valid samples:%d\n", N, validSamples);
    for (int k = 0; k < NClass; ++k) {
      if (smemClassCount[k] > 0)
        printf("Batch:%d, L:%d, count:%d, \n", N, k, smemClassCount[k]);
    }
  }
  __syncthreads();
#endif

  BlockClassSumPrefix sumPrefix;
  for (int s = 0; s < NClass; s += blockDim.x) {
    // s start from block
    int iClassSamples = 0;
    int iClass = s + threadIdx.x;
    if (iClass < NClass) {
      iClassSamples = smemClassCount[iClass];
    }
    BlockScanClass(temp_storage.storageScan).ExclusiveSum(iClassSamples, iClassSamples, sumPrefix);
    __syncthreads();
    if (iClass < NClass) {
      classStartPos[classBlockOffset + iClass] = iClassSamples;
    }
  }
  if (threadIdx.x == 0) {
    classStartPos[classBlockOffset + NClass] = sumPrefix.total;
    assert(sumPrefix.total <= validSamples); // background data removed.
    outValidSampleCount[N] = sumPrefix.total;
#if DUBUG_KERNEL
    if (N == DUBUG_BATCH)
      printf("After sortPerClass, batch:%d valid samples total:%d\n", N, sumPrefix.total);
#endif
  }
}

template <typename DType>
__device__ __forceinline__ OcrBBoxT<DType> readOcrBbox(const OcrBBoxT<DType>* inBbox, int idx) {
  OcrBBoxT<DType> ret = ((OcrBBoxT<DType>*) (inBbox))[idx];
  return ret;
}

template <typename DType>
__device__ __forceinline__ DType boxOcrIoU(const OcrBBoxT<DType>& a, const OcrBBoxT<DType>& b) {
  OcrBBoxT<DType> overlap = {
    dMAX(a.x1, b.x1), dMAX(a.y1, b.y1), dMIN(a.x2, b.x2), dMIN(a.y2, b.y2),
  };
  DType oW = overlap.x2 - overlap.x1;
  DType oH = overlap.y2 - overlap.y1;
  if (oW < (DType) 0 || oH < (DType) 0)
    return (DType) 0;
  DType oA = oW * oH;
  return (oA / ((a.y2 - a.y1) * (a.x2 - a.x1) + (b.y2 - b.y1) * (b.x2 - b.x1) - oA));
}

// PerClassNMS
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outFlagSamples OUT: int [N * samples]
template <typename DType, typename BoxType, int Threads = 256, int ItemsPerThreads = 4>
__global__ void OcrPerClassNMS_kernel(
  // int N,
  int samples, int NClass, const float nmsThreshold, const void* validSampleCountPtr,
  // const void *inScorePtr,
  const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* classStartsPtr,
  void* outFlagSamplesPtr) {
  typedef OcrBBoxT<BoxType> BBox;
  __shared__ struct {
    BBox refBox[MaxClassNum];
    int endIdx[MaxClassNum];
    int refIdx[MaxClassNum + 1];
    bool markSamples[Threads * ItemsPerThreads];
    int done;
  } smemClasses;
  assert(NClass + 1 < MaxClassNum);
  assert(samples <= Threads * ItemsPerThreads);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  // const DType *inScore = static_cast<const DType *>(inScorePtr);
  const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
  const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
  const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
  const int* classStarts = static_cast<const int*>(classStartsPtr);
  int* outFlagSamples = static_cast<int*>(outFlagSamplesPtr);

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int validSamples = validSampleCount[N];

  if (threadIdx.x == 0) {
    smemClasses.done = 0;
  }

  BBox curBox[ItemsPerThreads];
  int label[ItemsPerThreads];
#pragma unroll
  for (int ite = 0; ite * blockDim.x < validSamples; ++ite) {
    int curIdx = ite * blockDim.x + threadIdx.x;
    if (curIdx < validSamples) {
      label[ite] = (int) inLabel[blockOffset + curIdx];
      curBox[ite] = readOcrBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + curIdx]);
    } else {
      label[ite] = -1;
    }
    smemClasses.markSamples[curIdx] = (label[ite] < 0 ? false : true);
  }

  int classBlockOffset = N * (NClass + 1);
  for (int i = threadIdx.x; i < NClass + 1; i += blockDim.x) {
    int refIdx = classStarts[classBlockOffset + i];
    smemClasses.refIdx[i] = refIdx;
    if (refIdx < validSamples) {
      smemClasses.refBox[i] = readOcrBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NClass; i += blockDim.x) {
    int endIdx = smemClasses.refIdx[i + 1];
    smemClasses.endIdx[i] = endIdx;
    if (endIdx == smemClasses.refIdx[i]) {
      atomicAdd(&smemClasses.done, 1);
    }
  }
  __syncthreads();

#if DUBUG_KERNEL
  // print info
  if (N == DUBUG_BATCH && threadIdx.x == 0) {
    printf("batch:%d, before starting NMS, done count:%d\n", N, smemClasses.done);
    printf("batch:%d, Total num:%d, startPos:\n", N, validSamples);
    for (int k = 0; k < NClass; ++k) {
      if (smemClasses.refIdx[k] != smemClasses.endIdx[k]) {
        printf("Batch:%d, label:%d [%d : %d], check ref-label:%d\n", N, k, smemClasses.refIdx[k],
               smemClasses.endIdx[k], (int) inLabel[blockOffset + smemClasses.refIdx[k]]);
      }
    }
    printf("\n");
  }
  __syncthreads();
#endif

  // class done to check stop point
  while (smemClasses.done < NClass) {

    for (int ite = 0; ite * blockDim.x < validSamples; ++ite) {
      int curIdx = ite * blockDim.x + threadIdx.x;
      int refIdx = -1;
      int endIdx = -1;
      if (curIdx < validSamples && smemClasses.markSamples[curIdx]) {
        if (label[ite] >= 0) {
          refIdx = smemClasses.refIdx[label[ite]];
          endIdx = smemClasses.endIdx[label[ite]];
          if (curIdx > refIdx && curIdx < endIdx) {
            BBox refBox = smemClasses.refBox[label[ite]];
            if (boxOcrIoU(refBox, curBox[ite]) > (DType) nmsThreshold) {
              smemClasses.markSamples[curIdx] = false;
            }
          }
        }
      }
    }
    __syncthreads();

    // push refIdx/refBox forward to next mark
    // only the refIdx thread to push itself. other threads idle
    for (int i = threadIdx.x; i < NClass; i += blockDim.x) {
      int refIdx = smemClasses.refIdx[i];
      int endIdx = smemClasses.endIdx[i];
      if (refIdx < endIdx) {
        do {
          ++refIdx;
        } while (refIdx < endIdx && smemClasses.markSamples[refIdx] == false);
        smemClasses.refIdx[i] = refIdx;
        if (refIdx < endIdx) {
          smemClasses.refBox[i] = readOcrBbox(inBbox, blockOffset + inBboxRefIdx[blockOffset + refIdx]);
        } else {
          atomicAdd(&smemClasses.done, 1);
        }
      }
    }
    __syncthreads();
  }

  // no need to write all data out
  for (int segment = 0; segment < validSamples; segment += blockDim.x) {
    int curIdx = segment + threadIdx.x;
    if (curIdx < validSamples) {
      outFlagSamples[blockOffset + curIdx] = (smemClasses.markSamples[curIdx] ? 1 : 0);
    }
  }
}

// PerClassNMS
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outFlagSamples OUT: int [N * samples]
template <typename DType, typename BoxType, int Threads = 256, int ItemsPerThreads = 4>
__global__ void OcrPerClassNMS_kernel_v2(
  // int N,
  int samples, int NClass, const float nmsThreshold, const void* validSampleCountPtr,
  // const void *inScorePtr,
  const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* classStartsPtr,
  void* outFlagSamplesPtr) {
  typedef OcrBBoxT<BoxType> BBox;
  __shared__ struct {
    BBox refBox[MaxClassNum];
    int endIdx[MaxClassNum];
    int refIdx[MaxClassNum + 1];
    bool markSamples[Threads * ItemsPerThreads];
    int done;
  } smemClasses;
  assert(NClass + 1 < MaxClassNum);
  assert(samples <= Threads * ItemsPerThreads);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  // const DType *inScore = static_cast<const DType *>(inScorePtr);
  const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr);
  const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
  const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
  const int* classStarts = static_cast<const int*>(classStartsPtr);
  int* outFlagSamples = static_cast<int*>(outFlagSamplesPtr);

  int validSamples = validSampleCount[0];
  if (threadIdx.x == 0) {
    smemClasses.done = 0;
  }

  BBox curBox;
  int label;
#pragma unroll

  int curIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (curIdx < validSamples) {
    label = (int) inLabel[curIdx];
    curBox = readOcrBbox(inBbox, inBboxRefIdx[curIdx]);
  } else {
    label = -1;
  }
  smemClasses.markSamples[curIdx] = (label < 0 ? false : true);

  for (int i = threadIdx.x; i < NClass + 1; i += blockDim.x) {
    int refIdx = classStarts[i];
    smemClasses.refIdx[i] = refIdx;
    if (refIdx < validSamples) {
      smemClasses.refBox[i] = readOcrBbox(inBbox, inBboxRefIdx[refIdx]);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NClass; i += blockDim.x) {
    int endIdx = smemClasses.refIdx[i + 1];
    smemClasses.endIdx[i] = endIdx;
    if (endIdx == smemClasses.refIdx[i]) {
      atomicAdd(&smemClasses.done, 1);
    }
  }
  __syncthreads();

#if DUBUG_KERNEL
  // print info
  if (N == DUBUG_BATCH && threadIdx.x == 0) {
    printf("batch:%d, before starting NMS, done count:%d\n", N, smemClasses.done);
    printf("batch:%d, Total num:%d, startPos:\n", N, validSamples);
    for (int k = 0; k < NClass; ++k) {
      if (smemClasses.refIdx[k] != smemClasses.endIdx[k]) {
        printf("Batch:%d, label:%d [%d : %d], check ref-label:%d\n", N, k, smemClasses.refIdx[k],
               smemClasses.endIdx[k], (int) inLabel[blockOffset + smemClasses.refIdx[k]]);
      }
    }
    printf("\n");
  }
  __syncthreads();
#endif

  // class done to check stop point
  while (smemClasses.done < NClass) {
    int curIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int refIdx = -1;
    int endIdx = -1;
    if (curIdx < validSamples && smemClasses.markSamples[curIdx]) {
      if (label >= 0) {
        refIdx = smemClasses.refIdx[label];
        endIdx = smemClasses.endIdx[label];
        if (curIdx > refIdx && curIdx < endIdx) {
          BBox refBox = smemClasses.refBox[label];
          if (boxOcrIoU(refBox, curBox) > (DType) nmsThreshold) {
            smemClasses.markSamples[curIdx] = false;
          }
        }
      }
    }

    __syncthreads();

    // push refIdx/refBox forward to next mark
    // only the refIdx thread to push itself. other threads idle
    for (int i = threadIdx.x; i < NClass; i += blockDim.x) {
      int refIdx = smemClasses.refIdx[i];
      int endIdx = smemClasses.endIdx[i];
      if (refIdx < endIdx) {
        do {
          ++refIdx;
        } while (refIdx < endIdx && smemClasses.markSamples[refIdx] == false);
        smemClasses.refIdx[i] = refIdx;
        if (refIdx < endIdx) {
          smemClasses.refBox[i] = readOcrBbox(inBbox, inBboxRefIdx[refIdx]);
        } else {
          atomicAdd(&smemClasses.done, 1);
        }
      }
    }
    __syncthreads();
  }

  // no need to write all data out
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIdx < validSamples) {
    outFlagSamples[outIdx] = (smemClasses.markSamples[outIdx] ? 1 : 0);
  }
}

// // TopKGather
// // gridDim.x : batch-N
// // blockDim.x : Threads
// // ItemsPerThreads : = divUp(samples, Threads)
// // outDetectionCount : int [N], must be set 0 before kernel
// template <typename DType, typename BoxType, int Threads = 256, int MaxItemsPerThreads = 24>
// __global__ void OcrTopKGatherProposal_kernel(int samples, int keepTopK, const void* validSampleCountPtr,
//     const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr,
//     const void* inFlagSamplesPtr, void* outBboxPtr) {
//   typedef OcrBBoxT<BoxType> BBox;
//   typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
//   typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
//   typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
//   typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
//   typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
//   typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
//   typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
//   typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
//   typedef cub::BlockRadixSort<DType, Threads, 9, int> BlockRadixSort9;
//   typedef cub::BlockRadixSort<DType, Threads, 10, int> BlockRadixSort10;
//   typedef cub::BlockRadixSort<DType, Threads, 11, int> BlockRadixSort11;
//   typedef cub::BlockRadixSort<DType, Threads, 12, int> BlockRadixSort12;
//   typedef cub::BlockRadixSort<DType, Threads, 13, int> BlockRadixSort13;
//   typedef cub::BlockRadixSort<DType, Threads, 14, int> BlockRadixSort14;
//   typedef cub::BlockRadixSort<DType, Threads, 15, int> BlockRadixSort15;
//   typedef cub::BlockRadixSort<DType, Threads, 16, int> BlockRadixSort16;
//   typedef cub::BlockRadixSort<DType, Threads, 17, int> BlockRadixSort17;
//   typedef cub::BlockRadixSort<DType, Threads, 18, int> BlockRadixSort18;
//   typedef cub::BlockRadixSort<DType, Threads, 19, int> BlockRadixSort19;
//   typedef cub::BlockRadixSort<DType, Threads, 20, int> BlockRadixSort20;
//   typedef cub::BlockRadixSort<DType, Threads, 21, int> BlockRadixSort21;
//   typedef cub::BlockRadixSort<DType, Threads, 22, int> BlockRadixSort22;
//   typedef cub::BlockRadixSort<DType, Threads, 23, int> BlockRadixSort23;
//   typedef cub::BlockRadixSort<DType, Threads, 24, int> BlockRadixSort24;
//   __shared__ union {
//     typename BlockRadixSort24::TempStorage sort24;
//     typename BlockRadixSort23::TempStorage sort23;
//     typename BlockRadixSort22::TempStorage sort22;
//     typename BlockRadixSort21::TempStorage sort21;
//     typename BlockRadixSort20::TempStorage sort20;
//     typename BlockRadixSort19::TempStorage sort19;
//     typename BlockRadixSort18::TempStorage sort18;
//     typename BlockRadixSort17::TempStorage sort17;
//     typename BlockRadixSort16::TempStorage sort16;
//     typename BlockRadixSort15::TempStorage sort15;
//     typename BlockRadixSort14::TempStorage sort14;
//     typename BlockRadixSort13::TempStorage sort13;
//     typename BlockRadixSort12::TempStorage sort12;
//     typename BlockRadixSort11::TempStorage sort11;
//     typename BlockRadixSort10::TempStorage sort10;
//     typename BlockRadixSort9::TempStorage sort9;
//     typename BlockRadixSort8::TempStorage sort8;
//     typename BlockRadixSort7::TempStorage sort7;
//     typename BlockRadixSort6::TempStorage sort6;
//     typename BlockRadixSort5::TempStorage sort5;
//     typename BlockRadixSort4::TempStorage sort4;
//     typename BlockRadixSort3::TempStorage sort3;
//     typename BlockRadixSort2::TempStorage sort2;
//     typename BlockRadixSort1::TempStorage sort1;
//   } temp_storage;

//   assert(MaxItemsPerThreads * Threads >= samples);

//   const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
//   const DType* inScore = static_cast<const DType*>(inScorePtr);
//   const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
//   const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
//   const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
//   BBox* outBbox = static_cast<BBox*>(outBboxPtr);

//   int N = blockIdx.x;
//   int blockOffset = N * samples;
//   int validSamples = validSampleCount[N];
//   int finalTopK = dMIN(keepTopK, validSamples);

//   int idx[MaxItemsPerThreads];
//   DType score[MaxItemsPerThreads];
//   int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

//   for (int ite = 0; ite < totalItems; ++ite) {
//     int curIdx = ite * blockDim.x + threadIdx.x;
//     if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx]) {
//       idx[ite] = curIdx;
//       score[ite] = inScore[blockOffset + curIdx];
//     } else {
//       idx[ite] = -1;
//       score[ite] = -1 * INFINITY;
//     }
//   }

//   switch (totalItems) {
//   case 0:
//     break;
//   case 1:
//     BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
//     break;
//   case 2:
//     BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
//     break;
//   case 3:
//     BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
//     break;
//   case 4:
//     BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
//     break;
//   case 5:
//     BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
//     break;
//   case 6:
//     BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
//     break;
//   case 7:
//     BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
//     break;
//   case 8:
//     BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
//   case 9:
//     BlockRadixSort9(temp_storage.sort9).SortDescendingBlockedToStriped((DType(&)[9]) score, (int(&)[9]) idx);
//     break;
//   case 10:
//     BlockRadixSort10(temp_storage.sort10).SortDescendingBlockedToStriped((DType(&)[10]) score, (int(&)[10]) idx);
//     break;
//   case 11:
//     BlockRadixSort11(temp_storage.sort11).SortDescendingBlockedToStriped((DType(&)[11]) score, (int(&)[11]) idx);
//     break;
//   case 12:
//     BlockRadixSort12(temp_storage.sort12).SortDescendingBlockedToStriped((DType(&)[12]) score, (int(&)[12]) idx);
//     break;
//   case 13:
//     BlockRadixSort13(temp_storage.sort13).SortDescendingBlockedToStriped((DType(&)[13]) score, (int(&)[13]) idx);
//     break;
//   case 14:
//     BlockRadixSort14(temp_storage.sort14).SortDescendingBlockedToStriped((DType(&)[14]) score, (int(&)[14]) idx);
//     break;
//   case 15:
//     BlockRadixSort15(temp_storage.sort15).SortDescendingBlockedToStriped((DType(&)[15]) score, (int(&)[15]) idx);
//     break;
//   case 16:
//     BlockRadixSort16(temp_storage.sort16).SortDescendingBlockedToStriped((DType(&)[16]) score, (int(&)[16]) idx);
//   case 17:
//     BlockRadixSort17(temp_storage.sort17).SortDescendingBlockedToStriped((DType(&)[17]) score, (int(&)[17]) idx);
//     break;
//   case 18:
//     BlockRadixSort18(temp_storage.sort18).SortDescendingBlockedToStriped((DType(&)[18]) score, (int(&)[18]) idx);
//     break;
//   case 19:
//     BlockRadixSort19(temp_storage.sort19).SortDescendingBlockedToStriped((DType(&)[19]) score, (int(&)[19]) idx);
//     break;
//   case 20:
//     BlockRadixSort20(temp_storage.sort20).SortDescendingBlockedToStriped((DType(&)[20]) score, (int(&)[20]) idx);
//     break;
//   case 21:
//     BlockRadixSort21(temp_storage.sort21).SortDescendingBlockedToStriped((DType(&)[21]) score, (int(&)[21]) idx);
//     break;
//   case 22:
//     BlockRadixSort22(temp_storage.sort22).SortDescendingBlockedToStriped((DType(&)[22]) score, (int(&)[22]) idx);
//     break;
//   case 23:
//     BlockRadixSort23(temp_storage.sort23).SortDescendingBlockedToStriped((DType(&)[23]) score, (int(&)[23]) idx);
//     break;
//   case 24:
//     BlockRadixSort24(temp_storage.sort24).SortDescendingBlockedToStriped((DType(&)[24]) score, (int(&)[24]) idx);
//     break;
//   default:
//     assert(false);
//   }
//   __syncthreads();

//   int outBlockOffset = N * keepTopK;
//   int topkItems = (keepTopK + (Threads - 1)) / Threads;
//   for (int i = 0; i < topkItems; ++i) {
//     int curI = i * blockDim.x + threadIdx.x;
//     if (curI < keepTopK) {
//       BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
//       if (curI < finalTopK && idx[i] >= 0) {
//         oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
//       }
//       ((BBox*) outBbox)[outBlockOffset + curI] = oB;
//     }
//   }
// }

// template <typename DType, typename BoxType, int Threads = 256, int MaxItemsPerThreads = 24>
// __global__ void OcrTopKGather_kernel(int samples, int keepTopK, const void* validSampleCountPtr, const void* inScorePtr,
//                                      const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr,
//                                      const void* inCosPtr, const void* inSinPtr, void* outDetectionPtr) {
//   typedef OcrBBoxT<BoxType> BBox;
//   typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
//   typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
//   typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
//   typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
//   typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
//   typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
//   typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
//   typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
//   typedef cub::BlockRadixSort<DType, Threads, 9, int> BlockRadixSort9;
//   typedef cub::BlockRadixSort<DType, Threads, 10, int> BlockRadixSort10;
//   typedef cub::BlockRadixSort<DType, Threads, 11, int> BlockRadixSort11;
//   typedef cub::BlockRadixSort<DType, Threads, 12, int> BlockRadixSort12;
//   typedef cub::BlockRadixSort<DType, Threads, 13, int> BlockRadixSort13;
//   typedef cub::BlockRadixSort<DType, Threads, 14, int> BlockRadixSort14;
//   typedef cub::BlockRadixSort<DType, Threads, 15, int> BlockRadixSort15;
//   typedef cub::BlockRadixSort<DType, Threads, 16, int> BlockRadixSort16;
//   typedef cub::BlockRadixSort<DType, Threads, 17, int> BlockRadixSort17;
//   typedef cub::BlockRadixSort<DType, Threads, 18, int> BlockRadixSort18;
//   typedef cub::BlockRadixSort<DType, Threads, 19, int> BlockRadixSort19;
//   typedef cub::BlockRadixSort<DType, Threads, 20, int> BlockRadixSort20;
//   typedef cub::BlockRadixSort<DType, Threads, 21, int> BlockRadixSort21;
//   typedef cub::BlockRadixSort<DType, Threads, 22, int> BlockRadixSort22;
//   typedef cub::BlockRadixSort<DType, Threads, 23, int> BlockRadixSort23;
//   typedef cub::BlockRadixSort<DType, Threads, 24, int> BlockRadixSort24;
//   __shared__ union {
//     typename BlockRadixSort24::TempStorage sort24;
//     typename BlockRadixSort23::TempStorage sort23;
//     typename BlockRadixSort22::TempStorage sort22;
//     typename BlockRadixSort21::TempStorage sort21;
//     typename BlockRadixSort20::TempStorage sort20;
//     typename BlockRadixSort19::TempStorage sort19;
//     typename BlockRadixSort18::TempStorage sort18;
//     typename BlockRadixSort17::TempStorage sort17;
//     typename BlockRadixSort16::TempStorage sort16;
//     typename BlockRadixSort15::TempStorage sort15;
//     typename BlockRadixSort14::TempStorage sort14;
//     typename BlockRadixSort13::TempStorage sort13;
//     typename BlockRadixSort12::TempStorage sort12;
//     typename BlockRadixSort11::TempStorage sort11;
//     typename BlockRadixSort10::TempStorage sort10;
//     typename BlockRadixSort9::TempStorage sort9;
//     typename BlockRadixSort8::TempStorage sort8;
//     typename BlockRadixSort7::TempStorage sort7;
//     typename BlockRadixSort6::TempStorage sort6;
//     typename BlockRadixSort5::TempStorage sort5;
//     typename BlockRadixSort4::TempStorage sort4;
//     typename BlockRadixSort3::TempStorage sort3;
//     typename BlockRadixSort2::TempStorage sort2;
//     typename BlockRadixSort1::TempStorage sort1;
//   } temp_storage;
//   assert(MaxItemsPerThreads * Threads >= samples);

//   const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
//   const DType* inScore = static_cast<const DType*>(inScorePtr);
//   const DType* inCos = static_cast<const DType*>(inCosPtr);
//   const DType* inSin = static_cast<const DType*>(inSinPtr);
//   const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr); // InLabel keeps INT32
//   const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
//   const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
//   const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
//   DType* outDetections = static_cast<DType*>(outDetectionPtr);

//   int N = blockIdx.x;
//   int blockOffset = N * samples;
//   int validSamples = validSampleCount[N];
//   int finalTopK = dMIN(keepTopK, validSamples);

//   int idx[MaxItemsPerThreads];
//   DType score[MaxItemsPerThreads];
//   int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

//   for (int ite = 0; ite < totalItems; ++ite) {
//     int curIdx = ite * blockDim.x + threadIdx.x;
//     if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx]) {
//       idx[ite] = curIdx;
//       score[ite] = inScore[blockOffset + curIdx];
//     } else {
//       idx[ite] = -1;
//       score[ite] = 0.0f;
//     }
//   }

//   switch (totalItems) {
//   case 0:
//     break;
//   case 1:
//     BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
//     break;
//   case 2:
//     BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
//     break;
//   case 3:
//     BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
//     break;
//   case 4:
//     BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
//     break;
//   case 5:
//     BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
//     break;
//   case 6:
//     BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
//     break;
//   case 7:
//     BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
//     break;
//   case 8:
//     BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
//   case 9:
//     BlockRadixSort9(temp_storage.sort9).SortDescendingBlockedToStriped((DType(&)[9]) score, (int(&)[9]) idx);
//     break;
//   case 10:
//     BlockRadixSort10(temp_storage.sort10).SortDescendingBlockedToStriped((DType(&)[10]) score, (int(&)[10]) idx);
//     break;
//   case 11:
//     BlockRadixSort11(temp_storage.sort11).SortDescendingBlockedToStriped((DType(&)[11]) score, (int(&)[11]) idx);
//     break;
//   case 12:
//     BlockRadixSort12(temp_storage.sort12).SortDescendingBlockedToStriped((DType(&)[12]) score, (int(&)[12]) idx);
//     break;
//   case 13:
//     BlockRadixSort13(temp_storage.sort13).SortDescendingBlockedToStriped((DType(&)[13]) score, (int(&)[13]) idx);
//     break;
//   case 14:
//     BlockRadixSort14(temp_storage.sort14).SortDescendingBlockedToStriped((DType(&)[14]) score, (int(&)[14]) idx);
//     break;
//   case 15:
//     BlockRadixSort15(temp_storage.sort15).SortDescendingBlockedToStriped((DType(&)[15]) score, (int(&)[15]) idx);
//     break;
//   case 16:
//     BlockRadixSort16(temp_storage.sort16).SortDescendingBlockedToStriped((DType(&)[16]) score, (int(&)[16]) idx);
//   case 17:
//     BlockRadixSort17(temp_storage.sort17).SortDescendingBlockedToStriped((DType(&)[17]) score, (int(&)[17]) idx);
//     break;
//   case 18:
//     BlockRadixSort18(temp_storage.sort18).SortDescendingBlockedToStriped((DType(&)[18]) score, (int(&)[18]) idx);
//     break;
//   case 19:
//     BlockRadixSort19(temp_storage.sort19).SortDescendingBlockedToStriped((DType(&)[19]) score, (int(&)[19]) idx);
//     break;
//   case 20:
//     BlockRadixSort20(temp_storage.sort20).SortDescendingBlockedToStriped((DType(&)[20]) score, (int(&)[20]) idx);
//     break;
//   case 21:
//     BlockRadixSort21(temp_storage.sort21).SortDescendingBlockedToStriped((DType(&)[21]) score, (int(&)[21]) idx);
//     break;
//   case 22:
//     BlockRadixSort22(temp_storage.sort22).SortDescendingBlockedToStriped((DType(&)[22]) score, (int(&)[22]) idx);
//     break;
//   case 23:
//     BlockRadixSort23(temp_storage.sort23).SortDescendingBlockedToStriped((DType(&)[23]) score, (int(&)[23]) idx);
//     break;
//   case 24:
//     BlockRadixSort24(temp_storage.sort24).SortDescendingBlockedToStriped((DType(&)[24]) score, (int(&)[24]) idx);
//     break;
//   default:
//     assert(false);
//   }
//   __syncthreads();

//   int outBlockOffset = N * keepTopK;
//   int topkItems = (keepTopK + (Threads - 1)) / Threads;
//   for (int i = 0; i < topkItems; ++i) {
//     int curI = i * blockDim.x + threadIdx.x;
//     if (curI < keepTopK) {
//       BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
//       DType oS = 0.0f;
//       DType oCos = 0.0f;
//       DType oSin = 0.0f;
//       BoxType oL = -1;
//       if (curI < finalTopK && idx[i] >= 0 && score[i] > MinValidScore) {
//         oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
//         oS = score[i];
//         oL = (BoxType) inLabel[blockOffset + idx[i]];
//         oCos = inCos[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
//         oSin = inSin[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
//       }
//       outDetections[(outBlockOffset + curI) * 8] = oB.x1;
//       outDetections[(outBlockOffset + curI) * 8 + 1] = oB.y1;
//       outDetections[(outBlockOffset + curI) * 8 + 2] = oB.x2;
//       outDetections[(outBlockOffset + curI) * 8 + 3] = oB.y2;
//       outDetections[(outBlockOffset + curI) * 8 + 4] = oL;
//       outDetections[(outBlockOffset + curI) * 8 + 5] = oS;
//       outDetections[(outBlockOffset + curI) * 8 + 6] = oCos;
//       outDetections[(outBlockOffset + curI) * 8 + 7] = oSin;
//     }
//   }
// }


// TopKGather
// gridDim.x : batch-N
// blockDim.x : Threads
// ItemsPerThreads : = divUp(samples, Threads)
// outDetectionCount : int [N], must be set 0 before kernel
#define MaxItemsPerThreads 8
template <typename DType, typename BoxType, int Threads = 256>
__global__ void OcrTopKGatherProposal_kernel(int samples, int keepTopK, const void* validSampleCountPtr,
    const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr,
    const void* inFlagSamplesPtr, void* outBboxPtr) {
  typedef OcrBBoxT<BoxType> BBox;
  typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
  typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
  typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
  typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
  typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
  typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
  typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
  typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
  __shared__ union {
    typename BlockRadixSort8::TempStorage sort8;
    typename BlockRadixSort7::TempStorage sort7;
    typename BlockRadixSort6::TempStorage sort6;
    typename BlockRadixSort5::TempStorage sort5;
    typename BlockRadixSort4::TempStorage sort4;
    typename BlockRadixSort3::TempStorage sort3;
    typename BlockRadixSort2::TempStorage sort2;
    typename BlockRadixSort1::TempStorage sort1;
  } temp_storage;
  assert(MaxItemsPerThreads * Threads >= samples);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
  const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
  const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
  BBox* outBbox = static_cast<BBox*>(outBboxPtr);

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int validSamples = validSampleCount[N];
  int finalTopK = dMIN(keepTopK, validSamples);

  int idx[MaxItemsPerThreads];
  DType score[MaxItemsPerThreads];
  int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

  for (int ite = 0; ite < totalItems; ++ite) {
    int curIdx = ite * blockDim.x + threadIdx.x;
    if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx]) {
      idx[ite] = curIdx;
      score[ite] = inScore[blockOffset + curIdx];
    } else {
      idx[ite] = -1;
      score[ite] = -1 * INFINITY;
    }
  }

  switch (totalItems) {
  case 0:
    break;
  case 1:
    BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
    break;
  case 2:
    BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
    break;
  case 3:
    BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
    break;
  case 4:
    BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
    break;
  case 5:
    BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
    break;
  case 6:
    BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
    break;
  case 7:
    BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
    break;
  case 8:
    BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
    break;
  default:
    assert(false);
  }
  __syncthreads();

  int outBlockOffset = N * keepTopK;
  int topkItems = (keepTopK + (Threads - 1)) / Threads;
  for (int i = 0; i < topkItems; ++i) {
    int curI = i * blockDim.x + threadIdx.x;
    if (curI < keepTopK) {
      BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
      if (curI < finalTopK && idx[i] >= 0) {
        oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
      }
      ((BBox*) outBbox)[outBlockOffset + curI] = oB;
    }
  }
}

#define MaxItemsPerThreads 8
template <typename DType, typename BoxType, int Threads = 256>
__global__ void OcrTopKGather_kernel(int samples, int keepTopK, const void* validSampleCountPtr, const void* inScorePtr,
                                     const void* inLabelPtr, const void* inBboxPtr, const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr,
                                     const void* inCosPtr, const void* inSinPtr, void* outDetectionPtr) {
  typedef OcrBBoxT<BoxType> BBox;
  typedef cub::BlockRadixSort<DType, Threads, 1, int> BlockRadixSort1;
  typedef cub::BlockRadixSort<DType, Threads, 2, int> BlockRadixSort2;
  typedef cub::BlockRadixSort<DType, Threads, 3, int> BlockRadixSort3;
  typedef cub::BlockRadixSort<DType, Threads, 4, int> BlockRadixSort4;
  typedef cub::BlockRadixSort<DType, Threads, 5, int> BlockRadixSort5;
  typedef cub::BlockRadixSort<DType, Threads, 6, int> BlockRadixSort6;
  typedef cub::BlockRadixSort<DType, Threads, 7, int> BlockRadixSort7;
  typedef cub::BlockRadixSort<DType, Threads, 8, int> BlockRadixSort8;
  __shared__ union {
    typename BlockRadixSort8::TempStorage sort8;
    typename BlockRadixSort7::TempStorage sort7;
    typename BlockRadixSort6::TempStorage sort6;
    typename BlockRadixSort5::TempStorage sort5;
    typename BlockRadixSort4::TempStorage sort4;
    typename BlockRadixSort3::TempStorage sort3;
    typename BlockRadixSort2::TempStorage sort2;
    typename BlockRadixSort1::TempStorage sort1;
  } temp_storage;
  assert(MaxItemsPerThreads * Threads >= samples);

  const int* validSampleCount = static_cast<const int*>(validSampleCountPtr);
  const DType* inScore = static_cast<const DType*>(inScorePtr);
  const DType* inCos = static_cast<const DType*>(inCosPtr);
  const DType* inSin = static_cast<const DType*>(inSinPtr);
  const BoxType* inLabel = static_cast<const BoxType*>(inLabelPtr); // InLabel keeps INT32
  const BBox* inBbox = static_cast<const BBox*>(inBboxPtr);
  const int* inBboxRefIdx = static_cast<const int*>(inBboxRefIdxPtr);
  const int* inFlagSamples = static_cast<const int*>(inFlagSamplesPtr);
  DType* outDetections = static_cast<DType*>(outDetectionPtr);

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int validSamples = validSampleCount[N];
  int finalTopK = dMIN(keepTopK, validSamples);

  int idx[MaxItemsPerThreads];
  DType score[MaxItemsPerThreads];
  int totalItems = (validSamples + (blockDim.x - 1)) / blockDim.x;

  for (int ite = 0; ite < totalItems; ++ite) {
    int curIdx = ite * blockDim.x + threadIdx.x;
    if (curIdx < validSamples && inFlagSamples[blockOffset + curIdx]) {
      idx[ite] = curIdx;
      score[ite] = inScore[blockOffset + curIdx];
    } else {
      idx[ite] = -1;
      score[ite] = 0.0f;
    }
  }

  switch (totalItems) {
  case 0:
    break;
  case 1:
    BlockRadixSort1(temp_storage.sort1).SortDescendingBlockedToStriped((DType(&)[1]) score, (int(&)[1]) idx);
    break;
  case 2:
    BlockRadixSort2(temp_storage.sort2).SortDescendingBlockedToStriped((DType(&)[2]) score, (int(&)[2]) idx);
    break;
  case 3:
    BlockRadixSort3(temp_storage.sort3).SortDescendingBlockedToStriped((DType(&)[3]) score, (int(&)[3]) idx);
    break;
  case 4:
    BlockRadixSort4(temp_storage.sort4).SortDescendingBlockedToStriped((DType(&)[4]) score, (int(&)[4]) idx);
    break;
  case 5:
    BlockRadixSort5(temp_storage.sort5).SortDescendingBlockedToStriped((DType(&)[5]) score, (int(&)[5]) idx);
    break;
  case 6:
    BlockRadixSort6(temp_storage.sort6).SortDescendingBlockedToStriped((DType(&)[6]) score, (int(&)[6]) idx);
    break;
  case 7:
    BlockRadixSort7(temp_storage.sort7).SortDescendingBlockedToStriped((DType(&)[7]) score, (int(&)[7]) idx);
    break;
  case 8:
    BlockRadixSort8(temp_storage.sort8).SortDescendingBlockedToStriped((DType(&)[8]) score, (int(&)[8]) idx);
    break;
  default:
    assert(false);
  }
  __syncthreads();

  int outBlockOffset = N * keepTopK;
  int topkItems = (keepTopK + (Threads - 1)) / Threads;
  for (int i = 0; i < topkItems; ++i) {
    int curI = i * blockDim.x + threadIdx.x;
    if (curI < keepTopK) {
      BBox oB = {(BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f, (BoxType) 0.0f};
      DType oS = 0.0f;
      DType oCos = 0.0f;
      DType oSin = 0.0f;
      BoxType oL = -1;
      if (curI < finalTopK && idx[i] >= 0 && score[i] > MinValidScore) {
        oB = ((BBox*) inBbox)[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
        oS = score[i];
        oL = (BoxType) inLabel[blockOffset + idx[i]];
        oCos = inCos[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
        oSin = inSin[blockOffset + inBboxRefIdx[blockOffset + idx[i]]];
      }
      outDetections[(outBlockOffset + curI) * 8] = oB.x1;
      outDetections[(outBlockOffset + curI) * 8 + 1] = oB.y1;
      outDetections[(outBlockOffset + curI) * 8 + 2] = oB.x2;
      outDetections[(outBlockOffset + curI) * 8 + 3] = oB.y2;
      outDetections[(outBlockOffset + curI) * 8 + 4] = oL;
      outDetections[(outBlockOffset + curI) * 8 + 5] = oS;
      outDetections[(outBlockOffset + curI) * 8 + 6] = oCos;
      outDetections[(outBlockOffset + curI) * 8 + 7] = oSin;
    }
  }
}

RefineDetectionWorkSpace::RefineDetectionWorkSpace(
  const int batchSize, const int sampleCount, const RefineNMSParameters& param, const nvinfer1::DataType inType)
  : argMaxScoreDims(sampleCount, 1)
  , argMaxBboxDims(sampleCount, 4)
  , argMaxLabelDims(sampleCount, 1)
  , sortClassScoreDims(sampleCount, 1)
  , sortClassLabelDims(sampleCount, 1)
  , sortClassSampleIdxDims(sampleCount + 1, 1)
  , sortClassPosDims(param.numClasses + 1, 1)
  , sortNMSMarkDims(sampleCount, 1) {
  size_t sumSize = 0;

  const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;

  // resource
  // arMaxScore : [N, samples] : m_Type
  argMaxScoreOffset = sumSize;
  sumSize += AlignMem(dimVolume(argMaxScoreDims) * typeSize(type) * batchSize);

  argMaxBboxOffset = sumSize;
  // argMaxBbox : [N, samples, 4] : m_Type
  sumSize += AlignMem(dimVolume(argMaxBboxDims) * typeSize(type) * batchSize);

  argMaxLabelOffset = sumSize;
  // argMaxLabel : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(argMaxLabelDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

  sortClassScoreOffset = sumSize;
  // sortClassScore : [N, samples] : m_Type
  sumSize += AlignMem(dimVolume(sortClassScoreDims) * typeSize(type) * batchSize);

  sortClassLabelOffset = sumSize;
  // sortClassLabel : [N, samples] : m_Type
  sumSize += AlignMem(dimVolume(sortClassLabelDims) * typeSize(type) * batchSize);

  sortClassSampleIdxOffset = sumSize;
  // sortClassSampleIdx : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(sortClassSampleIdxDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

  sortClassValidCountOffset = sumSize;
  // sortClassValidCount : [N, 1] : kINT32
  sumSize += AlignMem(dimVolume(sortClassValidCountDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

  sortClassPosOffset = sumSize;
  // sortClassPos : [N, numClasses+1] : kINT32
  sumSize += AlignMem(dimVolume(sortClassPosDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

  sortNMSMarkOffset = sumSize;
  // sortNMSMark : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(sortNMSMarkDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);

  totalSize = sumSize;
}

OcrProposalWorkSpace::OcrProposalWorkSpace(const int batchSize, const int inputCnt, const int sampleCount,
    const RefineNMSParameters& param, const int numSegment, const nvinfer1::DataType inType)
  : preRefineSortedScoreDims(inputCnt, 1)
  , preRefineBboxDims(inputCnt, 4)
  , preRefineSortedScoreTopDims(sampleCount*numSegment, 1)
  , preRefineBboxTopDims(sampleCount*numSegment, 4)
  , argMaxScoreDims(sampleCount, 1)
  , argMaxBboxDims(sampleCount, 4)
  , argMaxLabelDims(sampleCount, 1)
  , sortClassScoreDims(sampleCount, 1)
  , sortClassLabelDims(sampleCount, 1)
  , sortClassSampleIdxDims(sampleCount, 1)
  , sortClassPosDims(param.numClasses + 1, 1)
  , sortNMSMarkDims(sampleCount, 1) {
  size_t sumSize = 0;

  const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;

  // resource
  // temp storage size for sorting scores
  tempStorageOffset = sumSize;
  // sumSize += (1 << 28) * batchSize;
  sumSize += (1 << 28) * batchSize;
  // printf("sumSize %lu \n", sumSize);
  // preRefineSortedScore: [N, inputcnt, 1]
  preRefineSortedScoreOffset = sumSize;
  sumSize += AlignMem(dimVolume(preRefineSortedScoreDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu, %lu\n", sumSize, AlignMem(dimVolume(preRefineSortedScoreDims) * typeSize(type) * batchSize));

  // preRefineBbox: [N, inputcnt, 4] // sorted bbox
  preRefineBboxOffset = sumSize;
  sumSize += AlignMem(dimVolume(preRefineBboxDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu, %lu\n", sumSize, AlignMem(dimVolume(preRefineBboxDims) * typeSize(type) * batchSize));

  // preRefineBbox: [N, sampleCount*numSegment, 1]
  preRefineSortedScoreTopOffset = sumSize;
  sumSize += AlignMem(dimVolume(preRefineSortedScoreTopDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  // preRefineBbox: [N, sampleCount*numSegment, 4] // sorted bbox
  preRefineBboxTopOffset = sumSize;
  sumSize += AlignMem(dimVolume(preRefineBboxTopDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  // arMaxScore : [N, samples] : m_Type
  argMaxScoreOffset = sumSize;
  sumSize += AlignMem(dimVolume(argMaxScoreDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  argMaxBboxOffset = sumSize;
  // argMaxBbox : [N, samples, 4] : m_Type
  sumSize += AlignMem(dimVolume(argMaxBboxDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  argMaxLabelOffset = sumSize;
  // argMaxLabel : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(argMaxLabelDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortClassScoreOffset = sumSize;
  // sortClassScore : [N, samples] : m_Type
  sumSize += AlignMem(dimVolume(sortClassScoreDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortClassLabelOffset = sumSize;
  // sortClassLabel : [N, samples] : m_Type
  sumSize += AlignMem(dimVolume(sortClassLabelDims) * typeSize(type) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortClassSampleIdxOffset = sumSize;
  // sortClassSampleIdx : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(sortClassSampleIdxDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortClassValidCountOffset = sumSize;
  // sortClassValidCount : [N, 1] : kINT32
  sumSize += AlignMem(dimVolume(sortClassValidCountDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortClassPosOffset = sumSize;
  // sortClassPos : [N, numClasses+1] : kINT32
  sumSize += AlignMem(dimVolume(sortClassPosDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  sortNMSMarkOffset = sumSize;
  // sortNMSMark : [N, samples] : kINT32
  sumSize += AlignMem(dimVolume(sortNMSMarkDims) * typeSize(nvinfer1::DataType::kINT32) * batchSize);
  // printf("sumSize %lu \n", sumSize);

  totalSize = sumSize;
}

OcrWorkROIAlignSpace::OcrWorkROIAlignSpace(const int batchSize, const int mFeatureLength, const xy_t* mFeatureSpatialSize, bool mPadBorder) {
  if (mPadBorder) {
    size_t sumSize = 0;
    const nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;

    p2PadOffset = sumSize;
    sumSize += typeSize(type) * batchSize * mFeatureLength * (mFeatureSpatialSize[0].y + 2) * (mFeatureSpatialSize[0].x + 2);

    p3PadOffset = sumSize;
    sumSize += typeSize(type) * batchSize * mFeatureLength * (mFeatureSpatialSize[1].y + 2) * (mFeatureSpatialSize[1].x + 2);

    p4PadOffset = sumSize;
    sumSize += typeSize(type) * batchSize * mFeatureLength * (mFeatureSpatialSize[2].y + 2) * (mFeatureSpatialSize[2].x + 2);

    p5PadOffset = sumSize;
    sumSize += typeSize(type) * batchSize * mFeatureLength * (mFeatureSpatialSize[3].y + 2) * (mFeatureSpatialSize[3].x + 2);

    totalSize = sumSize;
  }
};

template <int Threads>
cudaError_t argMaxGroup(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
                        const void* inScore, const void* inBbox, const void* validSamples, void* outScore, void* outLabel, void* outBbox) {
  int maxGridX = dMIN(samples, 512 / N);
  dim3 gridDim = {(unsigned int) nAlignDown(maxGridX, 32), (unsigned int) N, 1};
  dim3 threads = {Threads, 1, 1};
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    argMaxGroup_kernel<float, float, Threads><<<gridDim, threads, 0, stream>>>(
      samples, 0, NClass, inScore, inBbox, validSamples, outScore, outLabel, outBbox);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  return cudaGetLastError();
}

template <int Threads>
cudaError_t argOcrMaxGroup(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
                           const void* inScore, const void* validSamples, void* outScore, void* outLabel) {
  int maxGridX = dMIN(samples, 512 / N);
  dim3 gridDim = {(unsigned int) nAlignDown(maxGridX, 32), (unsigned int) N, 1};
  dim3 threads = {Threads, 1, 1};
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    argOcrMaxGroup_kernel<float, float, Threads><<<gridDim, threads, 0, stream>>>(
      samples, 0, NClass, inScore, validSamples, outScore, outLabel);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  return cudaGetLastError();
}

template <int Threads, int ItermPerThreads>
cudaError_t sortPerClass(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass, int background,
                         float scoreThreshold, const void* inSampleValidCount, const void* inScorePtr, const void* inLabelPtr,
                         const void* inBboxPtr, void* outclassStartPosPtr, void* outScorePtr, void* outLabelPtr, void* outSampleIdxPtr,
                         void* outValidSampleCountPtr) {
  int blocks = N;
  int threads = Threads;

  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    sortPerClass_kernel<float, float, Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples, NClass,
        background, scoreThreshold, inSampleValidCount, inScorePtr, inLabelPtr, inBboxPtr, outclassStartPosPtr,
        outScorePtr, outLabelPtr, outSampleIdxPtr, outValidSampleCountPtr);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  // print_first_k((int *)inSampleValidCount, 1, stream);
  // print_first_k((float *)inScorePtr, 100, stream);
  // print_first_k((float *)inLabelPtr, 100, stream);
  // print_first_k((float *)inBboxPtr, 100, stream);
  // print_first_k((int *)outclassStartPosPtr, 3, stream);
  // print_first_k((int *)outValidSampleCountPtr, 1, stream);
  // print_first_k((float *)outScorePtr, 100, stream);
  // print_first_k((float *)outLabelPtr, 100, stream);
  // print_first_k((int *)outSampleIdxPtr, 100, stream);

  return cudaGetLastError();
};

template <int Threads, int ItermPerThreads>
cudaError_t sortOcrPerClass(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass, int background,
                            float scoreThreshold, const void* inSampleValidCount, const void* inScorePtr, const void* inLabelPtr,
                            const void* inBboxPtr, void* outclassStartPosPtr, void* outScorePtr, void* outLabelPtr, void* outSampleIdxPtr,
                            void* outValidSampleCountPtr) {
  int blocks = N;
  int threads = Threads;

  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    sortOcrPerClass_kernel<float, float, Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples, NClass,
        background, scoreThreshold, inSampleValidCount, inScorePtr, inLabelPtr, inBboxPtr, outclassStartPosPtr,
        outScorePtr, outLabelPtr, outSampleIdxPtr, outValidSampleCountPtr);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }
  // print_first_k((int *)inSampleValidCount, 1, stream);
  // print_first_k((float *)inScorePtr, 100, stream);
  // print_first_k((float *)inLabelPtr, 100, stream);
  // print_first_k((float *)inBboxPtr, 100, stream);
  // print_first_k((int *)outclassStartPosPtr, 2, stream);
  // print_first_k((int *)outValidSampleCountPtr, 1, stream);
  // print_first_k((float *)outScorePtr, 100, stream);
  // print_first_k((float *)outLabelPtr, 100, stream);
  // print_first_k((int *)outSampleIdxPtr, 100, stream);

  return cudaGetLastError();
};

template <int Threads, int ItermPerThreads = 4>
cudaError_t OcrPerClassNMS(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
                           const float nmsThreshold, const void* validSampleCount,
                           // const void *inScore,
                           const void* inLabel, const void* inBbox, const void* inBboxRefIdx, const void* classStarts, void* outFlagSamples) {
  int blocks = N;
  int threads = Threads;

  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    OcrPerClassNMS_kernel<float, float, Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples, NClass, nmsThreshold,
        validSampleCount, inLabel, inBbox, inBboxRefIdx, classStarts, outFlagSamples);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  return cudaGetLastError();
}

template <int Threads, int ItermPerThreads = 4>
cudaError_t OcrPerClassNMS_v2(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int NClass,
                              const float nmsThreshold, const void* validSampleCount,
                              // const void *inScore,
                              const void* inLabel, const void* inBbox, const void* inBboxRefIdx, const void* classStarts, void* outFlagSamples) {
  int threads = Threads;
  int blocks = DivUp(N * samples, threads);

  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    OcrPerClassNMS_kernel_v2<float, float, Threads, ItermPerThreads><<<blocks, threads, 0, stream>>>(samples,
        NClass, nmsThreshold, validSampleCount, inLabel, inBbox, inBboxRefIdx, classStarts, outFlagSamples);
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  return cudaGetLastError();
}

// template <int Threads, int MaxItemsPerThreads = 24>
// cudaError_t OcrKeepTopKGather(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int keepTopK,
//                               const void* validSampleCountPtr, const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr,
//                               const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr, void* outDetections, int proposal,
//                               const void* inCos=NULL, const void* inSin=NULL) {
//   int blocks = N;
//   int threads = Threads;

//   switch (dtype) {
//   case nvinfer1::DataType::kFLOAT:
//     if (proposal) {
//       OcrTopKGatherProposal_kernel<float, float, Threads, MaxItemsPerThreads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
//           validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
//           outDetections);
//     } else {
//       OcrTopKGather_kernel<float, float, Threads, MaxItemsPerThreads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
//           validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
//           inCos, inSin, outDetections);
//     }
//     break;
//   case nvinfer1::DataType::kHALF:
//     break;
//   default:
//     assert(false);
//   }

//   return cudaGetLastError();
// }

template <int Threads>
cudaError_t OcrKeepTopKGather(cudaStream_t stream, int N, nvinfer1::DataType dtype, int samples, int keepTopK,
                              const void* validSampleCountPtr, const void* inScorePtr, const void* inLabelPtr, const void* inBboxPtr,
                              const void* inBboxRefIdxPtr, const void* inFlagSamplesPtr, void* outDetections, int proposal,
                              const void* inCos=NULL, const void* inSin=NULL) {
  int blocks = N;
  int threads = Threads;

  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    if (proposal) {
      OcrTopKGatherProposal_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
          validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
          outDetections);
    } else {
      OcrTopKGather_kernel<float, float, Threads><<<blocks, threads, 0, stream>>>(samples, keepTopK,
          validSampleCountPtr, inScorePtr, inLabelPtr, inBboxPtr, inBboxRefIdxPtr, inFlagSamplesPtr,
          inCos, inSin, outDetections);
    }
    break;
  case nvinfer1::DataType::kHALF:
    break;
  default:
    assert(false);
  }

  return cudaGetLastError();
}

cudaError_t OcrRefineBatchClassNMS(cudaStream_t stream, int N, int samples, nvinfer1::DataType dtype, int max_size,
                                   const RefineNMSParameters& param, const RefineDetectionWorkSpace& refineOffset, void* workspace,
                                   const void* inScores, const void* inDelta, const void* inCountValid, const void* inROI,
                                   const void* inCos, const void* inSin, void* outDetections) {
  int NClass = param.numClasses;
  int8_t* wsPtr = static_cast<int8_t*>(workspace);
  void* argMaxScorePtr = wsPtr + refineOffset.argMaxScoreOffset;
  void* argMaxLabelPtr = wsPtr + refineOffset.argMaxLabelOffset;
  void* argMaxBBoxPtr = wsPtr + refineOffset.argMaxBboxOffset;

  void* sortClassScorePtr = wsPtr + refineOffset.sortClassScoreOffset;
  void* sortClassLabelPtr = wsPtr + refineOffset.sortClassLabelOffset;
  void* sortClassSampleIdxPtr = wsPtr + refineOffset.sortClassSampleIdxOffset;
  void* sortClassValidCountPtr = wsPtr + refineOffset.sortClassValidCountOffset;
  void* sortClassPosPtr = wsPtr + refineOffset.sortClassPosOffset;
  void* sortNMSMarkPtr = wsPtr + refineOffset.sortNMSMarkOffset;

  cudaError_t status = cudaSuccess;
  CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

  // print_first_k((float *)inScores, 100, stream);
  // print_first_k((float *)inDelta, 100, stream);
  // print_first_k((float *)inROI, 100, stream);

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  if (NClass > 1) {
    // multiple classes
    status = argOcrMaxGroup<32>(stream, N, dtype, samples, NClass, inScores, inCountValid, argMaxScorePtr,
                                argMaxLabelPtr); // argMaxBBoxPtr means delta of bboxes
    argMaxBBoxPtr = const_cast<void*>(inDelta);
    assert(status == cudaSuccess);
    CUASSERT(status);
    // print_first_k((float *)argMaxScorePtr, 100, stream);
    // print_first_k((float *)argMaxLabelPtr, 100, stream);
  } else {
    // Only one class
    argMaxScorePtr = const_cast<void*>(inScores);
    argMaxBBoxPtr = const_cast<void*>(inDelta);
    int threads = 512;
    int blocks = (N * samples + threads - 1) / threads;
    blocks = dMIN(blocks, 8);
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: {
      resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      break;
    }
    default:
      assert(false);
    }
  }
  // gpu_timer.Stop();
  // printf("argOcrMaxGroup: %f ", (float(gpu_timer.ElapsedMillis())));

  // gpu_timer.Start();
  status = OcrApplyDelta2Bboxes(stream, N, samples, max_size, 2, inROI, argMaxBBoxPtr, argMaxBBoxPtr, 1);
  assert(status == cudaSuccess);
  // gpu_timer.Stop();
  // printf("OcrApplyDelta2Bboxes: %f ", (float(gpu_timer.ElapsedMillis())));

  // gpu_timer.Start();
  if (samples <= 1024) {
    status = sortPerClass<256, 4>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                  inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                  sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
  } else if (samples <= 2048) {
    status = sortPerClass<256, 8>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                  inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                  sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
  } else if (samples <= 4096) {
    status = sortPerClass<256, 16>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                   inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                   sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
  } else {
    assert(false && "unsupported sortPerClass");
    return cudaErrorLaunchFailure;
  }
  assert(status == cudaSuccess);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("sortPerClass: %f ", (float(gpu_timer.ElapsedMillis())));

  // gpu_timer.Start();
  status = OcrPerClassNMS<256>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
                               // sortClassScorePtr,
                               sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
  assert(status == cudaSuccess);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("OcrPerClassNMS: %f ", (float(gpu_timer.ElapsedMillis())));

  // gpu_timer.Start();
  status = OcrKeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
                                  sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outDetections, 0, inCos, inSin);
  assert(status == cudaSuccess);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("OcrKeepTopKGather: %f ", (float(gpu_timer.ElapsedMillis())));

  // print_first_k((float *)outDetections, 300*8, stream);
  return status;
}

cudaError_t OcrDecodeBox(cudaStream_t stream, int N, int samples, int max_size, int cascade_stage, nvinfer1::DataType dtype,
                         const void* inDelta, const void* inROI, void* outDetections) {
  cudaError_t status = cudaSuccess;

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  status = OcrApplyDelta2Bboxes(stream, N, samples, max_size, cascade_stage, inROI, inDelta, outDetections, 1);
  assert(status == cudaSuccess);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("OcrDecodeBox: %f ", (float(gpu_timer.ElapsedMillis())));

  return status;
}

struct BF_SCORE {
  float bg, fg;
};
// in_scores : [N, samples, 2]
// output_score : [N, samples, 1]
__global__ void extract_fg_kernel(int samples, const void* in_scores, void* output_score) {
  const BF_SCORE* in = static_cast<const BF_SCORE*>(in_scores);
  float* out = static_cast<float*>(output_score);

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;
  for (int i = 0; i < totalItems; i++) {
    int cur_id = i * blockDim.x + threadIdx.x;
    out[blockOffset + cur_id] = in[blockOffset + cur_id].fg;
  }
}

__global__ void set_offset_kernel(int stride, int size, int* output) {
  // One block, because batch size shouldn't be too large.
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    output[i] = i * stride;
  }
}

__global__ void ocr_resample_kernel(int orig_size, int sample_size, const void* orig_score_ptr, const void* orig_bbox_ptr,
                                    void* sampled_score_ptr, void* sampled_bbox_ptr) {
  const float* in_score = static_cast<const float*>(orig_score_ptr);
  const OcrBBoxT<float>* in_bbox = static_cast<const OcrBBoxT<float>*>(orig_bbox_ptr);
  float* out_score = static_cast<float*>(sampled_score_ptr);
  OcrBBoxT<float>* out_bbox = static_cast<OcrBBoxT<float>*>(sampled_bbox_ptr);

  int N = blockIdx.x;
  int blockOffset_in = N * orig_size;
  int blockOffset_out = N * sample_size;
  int realSampleCnt = dMIN(sample_size, orig_size);
  int totalItems = (realSampleCnt + (blockDim.x - 1)) / blockDim.x;

  for (int i = 0; i < totalItems; i++) {
    int cur_id = i * blockDim.x + threadIdx.x;
    if (cur_id < realSampleCnt) {
      out_score[blockOffset_out + cur_id] = in_score[blockOffset_in + cur_id];
      out_bbox[blockOffset_out + cur_id] = in_bbox[blockOffset_in + cur_id];
    }
  }
}

__global__ void ocr_resort_sample_kernel(int total_num, int num_segments, int sample_size,
    void* orig_score_ptr, void* orig_bbox_ptr) {
  float* in_score = static_cast<float*>(orig_score_ptr);
  OcrBBoxT<float>* in_bbox = static_cast<OcrBBoxT<float>*>(orig_bbox_ptr);

  int segment_index = blockIdx.x;
  int segment_num = total_num / num_segments;
  int segment_offest = segment_index * segment_num;
  int resample_offest = segment_index * sample_size;
  int totalItems = (sample_size + (blockDim.x - 1)) / blockDim.x;

  for (int i = 0; i < totalItems; i++) {
    int cur_id = i * blockDim.x + threadIdx.x;
    if (cur_id < sample_size) {
      in_score[resample_offest + cur_id] = in_score[segment_offest + cur_id];
      in_bbox[resample_offest + cur_id] = in_bbox[segment_offest + cur_id];
    }
  }
}

cudaError_t OcrProposalRefineBatchClassNMS(cudaStream_t stream, int N, int inputCnt, int samples, int mSegments, int max_size,
    nvinfer1::DataType dtype,
    const RefineNMSParameters& param, const OcrProposalWorkSpace& proposalOffset, void* workspace,
    const void* inScores, //[N, inputcnt, 1]
    const void* inDelta,  //[N, inputcnt, 4]
    const void* inCountValid,
    const void* inAnchors, //[N, inputcnt, 4]
    void* outProposals) {
  int8_t* wsPtr = static_cast<int8_t*>(workspace);
  void* tempStoragePtr = wsPtr + proposalOffset.tempStorageOffset;
  void* preRefineSortedScorePtr = wsPtr + proposalOffset.preRefineSortedScoreOffset;
  void* preRefineBboxPtr = wsPtr + proposalOffset.preRefineBboxOffset;
  void* preRefineSortedScoreTopPtr = wsPtr + proposalOffset.preRefineSortedScoreTopOffset;
  void* preRefineBboxTopPtr = wsPtr + proposalOffset.preRefineBboxTopOffset;

  void* argMaxScorePtr = wsPtr + proposalOffset.argMaxScoreOffset;
  void* argMaxLabelPtr = wsPtr + proposalOffset.argMaxLabelOffset;
  void* argMaxBBoxPtr = wsPtr + proposalOffset.argMaxBboxOffset;

  void* sortClassScorePtr = wsPtr + proposalOffset.sortClassScoreOffset;
  void* sortClassLabelPtr = wsPtr + proposalOffset.sortClassLabelOffset;
  void* sortClassSampleIdxPtr = wsPtr + proposalOffset.sortClassSampleIdxOffset;
  void* sortClassValidCountPtr = wsPtr + proposalOffset.sortClassValidCountOffset;
  void* sortClassPosPtr = wsPtr + proposalOffset.sortClassPosOffset;
  void* sortNMSMarkPtr = wsPtr + proposalOffset.sortNMSMarkOffset;

  cudaError_t status = cudaSuccess;
  CUASSERT(cudaMemsetAsync(sortClassValidCountPtr, 0, N * sizeof(int), stream));

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  // Here, inDelta are converted to normalize coordinates based on anchors
  status = OcrApplyDelta2Bboxes_v2(stream, N, inputCnt, max_size, 0, inAnchors, inDelta, const_cast<void*>(inDelta), 0);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("OcrApplyDelta2Bboxes: %f ", (float(gpu_timer.ElapsedMillis())));

  // 先分段进行排序，减少总数，降低排序时间
  int* offsets = static_cast<int*>(tempStoragePtr);
  int num_segments = mSegments;
  set_offset_kernel<<<1, BLOCK_MAX_THREADS, 0, stream>>>(N*inputCnt/num_segments, num_segments + 1, offsets);
  assert(cudaGetLastError() == cudaSuccess);
  tempStoragePtr = static_cast<void*>(static_cast<int*>(tempStoragePtr) + (num_segments + 1));

  // gpu_timer.Start();
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, (float*) inScores,
      (float*) preRefineSortedScorePtr, (OcrBBoxT<float>*) inDelta, (OcrBBoxT<float>*) preRefineBboxPtr, N * inputCnt,
      num_segments, offsets, offsets + 1, 0, 8 * sizeof(float), stream);

  assert((1 << 28) * (size_t)N > temp_storage_bytes);

  cub::DeviceSegmentedRadixSort::SortPairsDescending(tempStoragePtr, temp_storage_bytes, (float*) inScores,
      (float*) preRefineSortedScorePtr, (OcrBBoxT<float>*) inDelta, (OcrBBoxT<float>*) preRefineBboxPtr, N * inputCnt,
      num_segments, offsets, offsets + 1, 0, 8 * sizeof(float), stream);
  // gpu_timer.Stop();
  // printf("segment SortPairsDescending: %f ", (gpu_timer.ElapsedMillis()));

  // num_segments * samples重新进行排序
  // gpu_timer.Start();
  assert((num_segments * samples) <= (N * inputCnt / num_segments));
  // printf("num_segments %d, samples %d, N %d, inputcnt %d, num_segments * samples=%d, \
  //         N * inputCnt / num_segments=%d, res=%d", num_segments, samples, N, inputCnt, num_segments * samples, N * inputCnt / num_segments, (num_segments * samples) <= (N * inputCnt / num_segments));
  ocr_resort_sample_kernel<<<num_segments, dMIN(samples, BLOCK_MAX_THREADS), 0, stream>>>(N * inputCnt, num_segments, samples,
      preRefineSortedScorePtr, preRefineBboxPtr);
  set_offset_kernel<<<1, BLOCK_MAX_THREADS, 0, stream>>>(num_segments * samples, 1 + 1, offsets);
  assert(cudaGetLastError() == cudaSuccess);
  temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, (float*) preRefineSortedScorePtr,
      (float*) preRefineSortedScoreTopPtr, (OcrBBoxT<float>*) preRefineBboxPtr, (OcrBBoxT<float>*) preRefineBboxTopPtr,
      num_segments * samples, 1, offsets, offsets + 1, 0, 8 * sizeof(float), stream);
  assert((1 << 28) * (size_t)N > temp_storage_bytes);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(tempStoragePtr, temp_storage_bytes, (float*) preRefineSortedScorePtr,
      (float*) preRefineSortedScoreTopPtr, (OcrBBoxT<float>*) preRefineBboxPtr, (OcrBBoxT<float>*) preRefineBboxTopPtr,
      num_segments * samples, 1, offsets, offsets + 1, 0, 8 * sizeof(float), stream);
  // gpu_timer.Stop();
  // printf("SortPairsDescending: %f ", (gpu_timer.ElapsedMillis()));

  // gpu_timer.Start();
  int NClass = param.numClasses;
  assert(NClass == 1);
  if (NClass == 1) {
    // Only one class
    ocr_resample_kernel<<<N, dMIN(samples, BLOCK_MAX_THREADS), 0, stream>>>(
      num_segments * samples, samples, preRefineSortedScoreTopPtr, preRefineBboxTopPtr, argMaxScorePtr, argMaxBBoxPtr);

    int threads = BLOCK_MAX_THREADS;
    int blocks = (N * samples + threads - 1) / threads;
    blocks = dMIN(blocks, 8);
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: {
      resetMemValue_kernel<float><<<blocks, threads, 0, stream>>>(argMaxLabelPtr, N * samples, 0);
      break;
    }
    case nvinfer1::DataType::kHALF: {
      break;
    }
    default:
      assert(false);
    }
  }
  // gpu_timer.Stop();
  // printf("ocr_resample_kernel: %f ", (gpu_timer.ElapsedMillis()));

  // gpu_timer.Start();
  if (samples <= 1024) {
    const int ItermPerThreads = 1024 / BLOCK_MAX_THREADS;
    status = sortOcrPerClass<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                      inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                      sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    CUASSERT(status);

    status = OcrPerClassNMS_v2<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
                                     // sortClassScorePtr,
                                     sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    CUASSERT(status);
  } else if (samples <= 2048) {
    const int ItermPerThreads = 2048 / BLOCK_MAX_THREADS;
    status = sortOcrPerClass<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                      inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                      sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    CUASSERT(status);

    status = OcrPerClassNMS_v2<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
                                     // sortClassScorePtr,
                                     sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    CUASSERT(status);
  } else if (samples <= 4096) {
    const int ItermPerThreads = 4096 / BLOCK_MAX_THREADS;
    status = sortOcrPerClass<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                      inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                      sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    CUASSERT(status);

    status = OcrPerClassNMS_v2<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
                                     // sortClassScorePtr,
                                     sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    CUASSERT(status);
  } else if (samples <= 6144) {
    const int ItermPerThreads = 6144 / BLOCK_MAX_THREADS;
    status = sortOcrPerClass<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.backgroundLabelId, param.scoreThreshold,
                                      inCountValid, argMaxScorePtr, argMaxLabelPtr, argMaxBBoxPtr, sortClassPosPtr, sortClassScorePtr,
                                      sortClassLabelPtr, sortClassSampleIdxPtr, sortClassValidCountPtr);
    CUASSERT(status);

    status = OcrPerClassNMS_v2<BLOCK_MAX_THREADS, ItermPerThreads>(stream, N, dtype, samples, NClass, param.iouThreshold, sortClassValidCountPtr,
                                     // sortClassScorePtr,
                                     sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortClassPosPtr, sortNMSMarkPtr);
    CUASSERT(status);
  } else {
    assert(false && "unsupported sortPerClass");
    return cudaErrorLaunchFailure;
  }
  // print_first_k((int *)sortNMSMarkPtr, 100, stream);
  // gpu_timer.Stop();
  // printf("nms: %f ", (gpu_timer.ElapsedMillis()));

  // gpu_timer.Start();
  // // nvidia边缘盒子支持的线程数比较少，超过256会出性能问题
  // status = OcrKeepTopKGather<256>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
  //                                 sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outProposals, 1);

  // A100、3080卡系列线程设置成256，会出现内存越界，怀疑是BlockRadixSort分配了太多
  status = OcrKeepTopKGather<768>(stream, N, dtype, samples, param.keepTopK, sortClassValidCountPtr, sortClassScorePtr,
                                  sortClassLabelPtr, argMaxBBoxPtr, sortClassSampleIdxPtr, sortNMSMarkPtr, outProposals, 1);
  CUASSERT(status);
  // gpu_timer.Stop();
  // printf("OcrKeepTopKGather: %f ", (gpu_timer.ElapsedMillis()));
  // printf("outProposals:\n");
  // print_first_k((float *)outProposals, 100, stream);

  return status;
}

struct BBOX {
  float y1, x1, y2, x2;
};

struct DELTA {
  float dy, dx, logdh, logdw;
};

struct OCRBBOX {
  float x1, y1, x2, y2;
};

struct OCRDELTA {
  float dx, dy, logdw, logdh;
};

__global__ void ocr_apply_delta_kernel(int samples, int cascade_stage, const void* anchors, const void* delta,
                                       void* outputBbox, bool weight_Flag, int max_size0) {

  const OCRBBOX* anchors_in = static_cast<const OCRBBOX*>(anchors);
  const OCRDELTA* delta_in = static_cast<const OCRDELTA*>(delta);
  OCRBBOX* bbox_out = static_cast<OCRBBOX*>(outputBbox);

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;

  for (int i = 0; i < totalItems; i++) {
    int cur_id = i * blockDim.x + threadIdx.x;

    OCRBBOX cur_anchor_xyxy = anchors_in[blockOffset + cur_id];
    // convert xyxy -> cxcywh
    // cx, cy, w, h
    OCRBBOX cur_anchor_cxywh;

    cur_anchor_cxywh.x1 = (cur_anchor_xyxy.x1 + cur_anchor_xyxy.x2) / 2;
    cur_anchor_cxywh.y1 = (cur_anchor_xyxy.y1 + cur_anchor_xyxy.y2) / 2;
    cur_anchor_cxywh.x2 = (cur_anchor_xyxy.x2 - cur_anchor_xyxy.x1);
    cur_anchor_cxywh.y2 = (cur_anchor_xyxy.y2 - cur_anchor_xyxy.y1);

    OCRDELTA cur_delta = delta_in[blockOffset + cur_id];

    if (weight_Flag) {
      if (cascade_stage == 0) {
        // multiply std_dev
        cur_delta.dx *= 0.1;
        cur_delta.dy *= 0.1;
        cur_delta.logdw *= 0.2;
        cur_delta.logdh *= 0.2;
      } else if (cascade_stage == 1) {
        // multiply std_dev
        cur_delta.dx *= 0.05;
        cur_delta.dy *= 0.05;
        cur_delta.logdw *= 0.1;
        cur_delta.logdh *= 0.1;
      } else if (cascade_stage == 2) {
        // multiply std_dev
        cur_delta.dx /= 30.0f;
        cur_delta.dy /= 30.0f;
        cur_delta.logdw /= 15.0f;
        cur_delta.logdh /= 15.0f;
      }
    }

    // apply delta
    float max_size = (float)max_size0;
    float clip = log(max_size / 16.f);
    cur_anchor_cxywh.x1 += cur_delta.dx * cur_anchor_cxywh.x2;
    cur_anchor_cxywh.y1 += cur_delta.dy * cur_anchor_cxywh.y2;
    cur_anchor_cxywh.x2 *= expf(dMIN(cur_delta.logdw, clip));
    cur_anchor_cxywh.y2 *= expf(dMIN(cur_delta.logdh, clip));

    cur_anchor_xyxy.x1 = cur_anchor_cxywh.x1 - 0.5 * cur_anchor_cxywh.x2;
    cur_anchor_xyxy.y1 = cur_anchor_cxywh.y1 - 0.5 * cur_anchor_cxywh.y2;
    cur_anchor_xyxy.x2 = cur_anchor_xyxy.x1 + cur_anchor_cxywh.x2;
    cur_anchor_xyxy.y2 = cur_anchor_xyxy.y1 + cur_anchor_cxywh.y2;

    // clip bbox: a more precision clip method based on real window could be implemented
    cur_anchor_xyxy.x1 = dMAX(dMIN(cur_anchor_xyxy.x1, max_size), 0.0);
    cur_anchor_xyxy.y1 = dMAX(dMIN(cur_anchor_xyxy.y1, max_size), 0.0);
    cur_anchor_xyxy.x2 = dMAX(dMIN(cur_anchor_xyxy.x2, max_size), 0.0);
    cur_anchor_xyxy.y2 = dMAX(dMIN(cur_anchor_xyxy.y2, max_size), 0.0);

    bbox_out[blockOffset + cur_id].x1 = cur_anchor_xyxy.x1;
    bbox_out[blockOffset + cur_id].y1 = cur_anchor_xyxy.y1;
    bbox_out[blockOffset + cur_id].x2 = cur_anchor_xyxy.x2;
    bbox_out[blockOffset + cur_id].y2 = cur_anchor_xyxy.y2;
  }
}

__global__ void ocr_apply_delta_kernel_v2(int samples, int cascade_stage, const void* anchors, const void* delta,
    void* outputBbox, bool weight_Flag, int max_size0) {

  const OCRBBOX* anchors_in = static_cast<const OCRBBOX*>(anchors);
  const OCRDELTA* delta_in = static_cast<const OCRDELTA*>(delta);
  OCRBBOX* bbox_out = static_cast<OCRBBOX*>(outputBbox);

  int num_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int cur_id = num_idx; cur_id < samples; cur_id += blockDim.x * gridDim.x) {

    OCRBBOX cur_anchor_xyxy = anchors_in[cur_id];
    // convert xyxy -> cxcywh
    // cx, cy, w, h
    OCRBBOX cur_anchor_cxywh;

    cur_anchor_cxywh.x1 = (cur_anchor_xyxy.x1 + cur_anchor_xyxy.x2) / 2;
    cur_anchor_cxywh.y1 = (cur_anchor_xyxy.y1 + cur_anchor_xyxy.y2) / 2;
    cur_anchor_cxywh.x2 = (cur_anchor_xyxy.x2 - cur_anchor_xyxy.x1);
    cur_anchor_cxywh.y2 = (cur_anchor_xyxy.y2 - cur_anchor_xyxy.y1);

    OCRDELTA cur_delta = delta_in[cur_id];

    if (weight_Flag) {
      if (cascade_stage == 0) {
        // multiply std_dev
        cur_delta.dx *= 0.1;
        cur_delta.dy *= 0.1;
        cur_delta.logdw *= 0.2;
        cur_delta.logdh *= 0.2;
      } else if (cascade_stage == 1) {
        // multiply std_dev
        cur_delta.dx *= 0.05;
        cur_delta.dy *= 0.05;
        cur_delta.logdw *= 0.1;
        cur_delta.logdh *= 0.1;
      } else if (cascade_stage == 2) {
        // multiply std_dev
        cur_delta.dx /= 30.0f;
        cur_delta.dy /= 30.0f;
        cur_delta.logdw /= 15.0f;
        cur_delta.logdh /= 15.0f;
      }
    }

    // apply delta
    // float max_size = 1600.0f;
    float max_size = (float)max_size0;
    float clip = log(max_size / 16.f);
    cur_anchor_cxywh.x1 += cur_delta.dx * cur_anchor_cxywh.x2;
    cur_anchor_cxywh.y1 += cur_delta.dy * cur_anchor_cxywh.y2;
    cur_anchor_cxywh.x2 *= expf(dMIN(cur_delta.logdw, clip));
    cur_anchor_cxywh.y2 *= expf(dMIN(cur_delta.logdh, clip));

    cur_anchor_xyxy.x1 = cur_anchor_cxywh.x1 - 0.5 * cur_anchor_cxywh.x2;
    cur_anchor_xyxy.y1 = cur_anchor_cxywh.y1 - 0.5 * cur_anchor_cxywh.y2;
    cur_anchor_xyxy.x2 = cur_anchor_xyxy.x1 + cur_anchor_cxywh.x2;
    cur_anchor_xyxy.y2 = cur_anchor_xyxy.y1 + cur_anchor_cxywh.y2;

    // clip bbox: a more precision clip method based on real window could be implemented
    cur_anchor_xyxy.x1 = dMAX(dMIN(cur_anchor_xyxy.x1, max_size), 0.0);
    cur_anchor_xyxy.y1 = dMAX(dMIN(cur_anchor_xyxy.y1, max_size), 0.0);
    cur_anchor_xyxy.x2 = dMAX(dMIN(cur_anchor_xyxy.x2, max_size), 0.0);
    cur_anchor_xyxy.y2 = dMAX(dMIN(cur_anchor_xyxy.y2, max_size), 0.0);

    bbox_out[cur_id].x1 = cur_anchor_xyxy.x1;
    bbox_out[cur_id].y1 = cur_anchor_xyxy.y1;
    bbox_out[cur_id].x2 = cur_anchor_xyxy.x2;
    bbox_out[cur_id].y2 = cur_anchor_xyxy.y2;
  }
}

cudaError_t OcrApplyDelta2Bboxes(cudaStream_t stream, int N,
                                 int samples,         // number of anchors per image
                                 int max_size,
                                 int cascade_stage,
                                 const void* anchors, // [N, anchors, (x1, y1, x2, y2)]
                                 const void* delta,   //[N, anchors, (dx, dy, log(dw), log(dh)])
                                 void* outputBbox,     //[N, anchors, (x1, y1, x2, y2)]
                                 bool weight_Flag
                                ) {

  int blocks = N;
  int threads = dMIN(samples, BLOCK_MAX_THREADS);

  // delta multiply bbox_std
  // apply delta steps:
  //  cy = anchor_cy + dy*height
  //  cx = anchor_cx + dx*weight
  //  h = exp(dh)*anchor_h
  //  w = exp(dw)*anchor_w
  // clip the bbox
  // printf("cascade_stage %d, max_size %d, delta \n", cascade_stage, max_size, delta);
  // print_first_k((float *)delta, 100, stream);
  // printf("cascade_stage %d, anchors \n", cascade_stage, anchors);
  // print_first_k((float *)anchors, 100, stream);
  ocr_apply_delta_kernel<<<blocks, threads, 0, stream>>>(samples, cascade_stage, anchors, delta, outputBbox, weight_Flag, max_size);
  // printf("cascade_stage %d, outputBbox \n", cascade_stage, outputBbox);
  // print_first_k((float *)outputBbox, 100, stream);

  return cudaGetLastError();
}

cudaError_t OcrApplyDelta2Bboxes_v2(cudaStream_t stream, int N,
                                    int samples,         // number of anchors per image
                                    int max_size,
                                    int cascade_stage,
                                    const void* anchors, // [N, anchors, (x1, y1, x2, y2)]
                                    const void* delta,   //[N, anchors, (dx, dy, log(dw), log(dh)])
                                    void* outputBbox,     //[N, anchors, (x1, y1, x2, y2)]
                                    bool weight_Flag
                                   ) {
  // todo: 不能超过硬件所支持的最大线程数, 需要确定设备每个block最大的线程数和每个grid最大的block数
  int threads = dMIN(N * samples, BLOCK_MAX_THREADS);
  int blocks = DivUp(N * samples, threads);

  ocr_apply_delta_kernel_v2<<<blocks, threads, 0, stream>>>(N * samples, cascade_stage, anchors, delta,
      outputBbox, weight_Flag, max_size);

  return cudaGetLastError();
}

template <typename Tfeat>
__device__ inline Tfeat interpolateOcrBilinear(const Tfeat* src, xy_t srcDims, float y, float x, bool pad_border) {
  if (y < 0 || y > srcDims.y - 1) {
    return 0;
  }
  if (x < 0 || x > srcDims.x - 1) {
    return 0;
  }

  const int y0 = floorf(y);
  const int y1 = ceilf(y);
  const float yAlpha = y - y0;
  const int x0 = floorf(x);
  const int x1 = ceilf(x);
  const float xAlpha = x - x0;

  const float src00 = src[(y0) *srcDims.x + (x0)];
  const float src01 = src[(y0) *srcDims.x + (x1)];
  const float src10 = src[(y1) *srcDims.x + (x0)];
  const float src11 = src[(y1) *srcDims.x + (x1)];

  const float src0 = src00 * (1 - xAlpha) + src01 * xAlpha;
  const float src1 = src10 * (1 - xAlpha) + src11 * xAlpha;

  return src0 * (1 - yAlpha) + src1 * yAlpha;

}

template <typename Trois, typename Tfeat>
__global__ void roiOcrAlign_kernel(int featureCount, int roiCount,

                                   float threshold, const Trois* rois,

                                   const Tfeat* P2, const xy_t P2dims, const Tfeat* P3, const xy_t P3dims, const Tfeat* P4, const xy_t P4dims,
                                   const Tfeat* P5, const xy_t P5dims,

                                   Tfeat* pooled, const xy_t poolDims, bool pad_border) {
  const int batch = blockIdx.x;
  const int feature = blockIdx.y;

  // blockDim.x个box并行计算，超出部分顺序执行
  for (int roiIdx = threadIdx.x; roiIdx < roiCount; roiIdx += blockDim.x) {
    const Trois* roi = rois + 4 * (batch * roiCount + roiIdx);

    float x1 = roi[0];
    float y1 = roi[1];
    float x2 = roi[2];
    float y2 = roi[3];

    // if (!(0 <= y1 && y1 <= 1600 && 0 <= x1 && x1 <= 1600 && 0 <= y2 && y2 <= 1600 && 0 <= x2 && x2 <= 1600 && y1 < y2
    //         && x1 < x2))
    // {
    //     continue;
    // }
    // else
    // {
    // }

    const float hw = (y2 - y1) * (x2 - x1);

    const Tfeat* src = P2;
    xy_t srcDims = P2dims;
    int iP = 2;
    float feature_stride = 4.0f;

    // 原始代码有bug，会不断对threshold进行赋值操作
    if (hw >= threshold) {
      src = P3;
      srcDims = P3dims;
      ++iP;
      feature_stride = 8.0f;
    }

    if (hw >= threshold * 4) {
      src = P4;
      srcDims = P4dims;
      ++iP;
      feature_stride = 16.0f;
    }

    if (hw >= threshold * 4 * 4) {
      src = P5;
      srcDims = P5dims;
      ++iP;
      feature_stride = 32.0f;
    }

    src += srcDims.x * srcDims.y * (batch * featureCount + feature);

    Tfeat* dst
      = pooled + poolDims.x * poolDims.y * (batch * roiCount * featureCount + roiIdx * featureCount + feature);

    // if (roiIdx == 274 && feature == 0)
    // {
    //     printf("%f ", (float(x1)));
    //     printf("%f ", (float(y1)));
    //     printf("%f ", (float(x2)));
    //     printf("%f ", (float(y2)));
    //     printf("%f ", (float(hw)));
    //     printf("%f ", (float(threshold)));
    //     printf("%f ", (float(feature_stride)));
    //     printf("\n");
    // }

    // 对x1, y1, x2, y2进行归一化
    y1 = y1 / feature_stride;
    x1 = x1 / feature_stride;
    y2 = y2 / feature_stride;
    x2 = x2 / feature_stride;
    if (pad_border) {
      y1 = y1 + 1;
      x1 = x1 + 1;
      y2 = y2 + 1;
      x2 = x2 + 1;
    }
    const float spacing_w = (x2 - x1) / poolDims.x;
    const float spacing_h = (y2 - y1) / poolDims.y;
    const float xStart = (x1 + spacing_w / 2 - 0.5);
    const float yStart = (y1 + spacing_h / 2 - 0.5);
    const float xEnd = xStart + spacing_w * (poolDims.x - 1);
    const float yEnd = yStart + spacing_h * (poolDims.y - 1);

    // if (roiIdx == 274 && feature == 0)
    // {
    //     printf("%f ", (float(xStart)));
    //     printf("%f ", (float(yStart)));
    //     printf("%f ", (float(xEnd)));
    //     printf("%f ", (float(yEnd)));
    //     printf("\n");
    // }

    for (int yy = 0; yy < poolDims.y; ++yy) {
      const float ySample = min(yStart + spacing_h * yy, yEnd);

      for (int xx = 0; xx < poolDims.x; ++xx) {
        const float xSample = min(xStart + spacing_w * xx, xEnd);

        float result = interpolateOcrBilinear(src, srcDims, ySample, xSample, pad_border);
        *dst = result;
        dst++;

        // if (roiIdx == 274 && feature == 0)
        // {
        //     printf("%f ", (float(result)));
        // }
      }
    }
  }
}

template <typename Trois, typename Tfeat>
__global__ void roiOcrAlign_kernel_v2(int virtual_thread_count, int depth,

                                      float threshold, const Trois* rois,

                                      const Tfeat* P2, const xy_t P2dims, const Tfeat* P3, const xy_t P3dims, const Tfeat* P4, const xy_t P4dims,
                                      const Tfeat* P5, const xy_t P5dims,

                                      Tfeat* pooled, const xy_t poolDims, bool pad_border) {
  int num_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int out_idx = num_idx; out_idx < virtual_thread_count; out_idx += blockDim.x * gridDim.x) {
    // out_idx = w + crop_width * (h + crop_height * (d + depth * b))
    int idx = out_idx;
    const int x = idx % poolDims.x;
    idx /= poolDims.x;
    const int y = idx % poolDims.y;
    idx /= poolDims.y;
    const int d = idx % depth;
    const int b = idx / depth;

    float x1 = rois[b * 4];
    float y1 = rois[b * 4 + 1];
    float x2 = rois[b * 4 + 2];
    float y2 = rois[b * 4 + 3];

    const float hw = (y2 - y1) * (x2 - x1);

    const Tfeat* src = P2;
    xy_t srcDims = P2dims;
    int iP = 2;
    float feature_stride = 4.0f;

    // 原始代码有bug，会不断对threshold进行赋值操作
    if (hw >= threshold) {
      src = P3;
      srcDims = P3dims;
      ++iP;
      feature_stride = 8.0f;
    }

    if (hw >= threshold * 4) {
      src = P4;
      srcDims = P4dims;
      ++iP;
      feature_stride = 16.0f;
    }

    if (hw >= threshold * 4 * 4) {
      src = P5;
      srcDims = P5dims;
      ++iP;
      feature_stride = 32.0f;
    }

    // 对x1, y1, x2, y2进行归一化
    y1 = y1 / feature_stride;
    x1 = x1 / feature_stride;
    y2 = y2 / feature_stride;
    x2 = x2 / feature_stride;
    if (pad_border) {
      y1 += 1;
      x1 += 1;
      y2 += 1;
      x2 += 1;
    }
    const float spacing_w = (x2 - x1) / poolDims.x;
    const float spacing_h = (y2 - y1) / poolDims.y;
    const float xStart = (x1 + spacing_w / 2 - 0.5);
    const float yStart = (y1 + spacing_h / 2 - 0.5);
    const float xEnd = xStart + spacing_w * (poolDims.x - 1);
    const float yEnd = yStart + spacing_h * (poolDims.y - 1);

    const float in_y = min(yStart + spacing_h * y, yEnd);
    const float in_x = min(xStart + spacing_w * x, xEnd);

    if (in_y < 0 || in_y > srcDims.y - 1) {
      pooled[out_idx] = 0;
      continue;
    }
    if (in_x < 0 || in_x > srcDims.x - 1) {
      pooled[out_idx] = 0;
      continue;
    }

    const int b_in = 0;
    const int top_y_index = floorf(in_y);
    const int bottom_y_index = ceilf(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = floorf(in_x);
    const int right_x_index = ceilf(in_x);
    const float x_lerp = in_x - left_x_index;

    const float top_left(static_cast<float>(
                           src[((b_in * depth + d) * srcDims.y + top_y_index) * srcDims.x + left_x_index]));
    const float top_right(static_cast<float>(
                            src[((b_in * depth + d) * srcDims.y + top_y_index) * srcDims.x + right_x_index]));
    const float bottom_left(static_cast<float>(
                              src[((b_in * depth + d) * srcDims.y + bottom_y_index) * srcDims.x + left_x_index]));
    const float bottom_right(static_cast<float>(
                               src[((b_in * depth + d) * srcDims.y + bottom_y_index) * srcDims.x + right_x_index]));
    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    pooled[out_idx] = top + (bottom - top) * y_lerp;
  }
}

cudaError_t roiOcrAlign(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

                        const void* rois, const void* const layers[], const xy_t* layerDims,

                        void* pooled, const xy_t poolDims, bool pad_border) {
  const dim3 blocks(batchSize, featureCount);
  const int threads(256);

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  roiOcrAlign_kernel<<<blocks, threads, 0, stream>>>(featureCount, roiCount, firstThreshold,
      static_cast<const float*>(rois),

      static_cast<const float*>(layers[0]), layerDims[0], static_cast<const float*>(layers[1]), layerDims[1],
      static_cast<const float*>(layers[2]), layerDims[2], static_cast<const float*>(layers[3]), layerDims[3],

      static_cast<float*>(pooled), poolDims, pad_border);
  // gpu_timer.Stop();
  // printf("roiOcrAlign: %f ", (float(gpu_timer.ElapsedMillis())));
  return cudaGetLastError();
}

__global__ void featurePad_kernel(
  const float* P2, const xy_t P2dims, const float* P3, const xy_t P3dims, const float* P4, const xy_t P4dims,
  const float* P5, const xy_t P5dims, float* p2Pad, float* p3Pad, float* p4Pad, float* p5Pad) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int batch = blockIdx.z;

  // 等效于image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
  if (x < P2dims.x && y < P2dims.y) {
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(y+1)*(P2dims.x+2)+(x+1)] = P2[batch*P2dims.x*P2dims.y+y*P2dims.x+x];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(x+1)] = P2[batch*P2dims.x*P2dims.y+x];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(P2dims.y+1)*(P2dims.x+2)+(x+1)] = P2[batch*P2dims.x*P2dims.y+
        (P2dims.y-1)*P2dims.x+x];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(y+1)*(P2dims.x+2)] = P2[batch*P2dims.x*P2dims.y+y*P2dims.x];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(y+1)*(P2dims.x+2)+(P2dims.x+1)] = P2[batch*P2dims.x*P2dims.y+
        y*P2dims.x+
        (P2dims.x-1)];

    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)] = P2[batch*P2dims.x*P2dims.y];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(P2dims.x+1)] = P2[batch*P2dims.x*P2dims.y+(P2dims.x-1)];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(P2dims.y+1)*(P2dims.x+2)] = P2[batch*P2dims.x*P2dims.y+(P2dims.y-1)*P2dims.x];
    p2Pad[batch*(P2dims.x+2)*(P2dims.y+2)+(P2dims.y+1)*(P2dims.x+2)+(P2dims.x+1)] = P2[batch*P2dims.x*P2dims.y+
        (P2dims.y-1)*P2dims.x+
        (P2dims.x-1)];
  }

  if (x < P3dims.x && y < P3dims.y) {
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(y+1)*(P3dims.x+2)+(x+1)] = P3[batch*P3dims.x*P3dims.y+y*P3dims.x+x];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(x+1)] = P3[batch*P3dims.x*P3dims.y+x];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(P3dims.y+1)*(P3dims.x+2)+(x+1)] = P3[batch*P3dims.x*P3dims.y+
        (P3dims.y-1)*P3dims.x+x];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(y+1)*(P3dims.x+2)] = P3[batch*P3dims.x*P3dims.y+y*P3dims.x];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(y+1)*(P3dims.x+2)+(P3dims.x+1)] = P3[batch*P3dims.x*P3dims.y+
        y*P3dims.x+(P3dims.x-1)];

    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)] = P3[batch*P3dims.x*P3dims.y];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(P3dims.x+1)] = P3[batch*P3dims.x*P3dims.y+(P3dims.x-1)];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(P3dims.y+1)*(P3dims.x+2)] = P3[batch*P3dims.x*P3dims.y+(P3dims.y-1)*P3dims.x];
    p3Pad[batch*(P3dims.x+2)*(P3dims.y+2)+(P3dims.y+1)*(P3dims.x+2)+(P3dims.x+1)] = P3[batch*P3dims.x*P3dims.y+
        (P3dims.y-1)*P3dims.x+
        (P3dims.x-1)];
  }

  if (x < P4dims.x && y < P4dims.y) {
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(y+1)*(P4dims.x+2)+(x+1)] = P4[batch*P4dims.x*P4dims.y+y*P4dims.x+x];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(x+1)] = P4[batch*P4dims.x*P4dims.y+x];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(P4dims.y+1)*(P4dims.x+2)+(x+1)] = P4[batch*P4dims.x*P4dims.y+
        (P4dims.y-1)*P4dims.x+x];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(y+1)*(P4dims.x+2)] = P4[batch*P4dims.x*P4dims.y+y*P4dims.x];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(y+1)*(P4dims.x+2)+(P4dims.x+1)] = P4[batch*P4dims.x*P4dims.y+
        y*P4dims.x+(P4dims.x-1)];

    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)] = P4[batch*P4dims.x*P4dims.y];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(P4dims.x+1)] = P4[batch*P4dims.x*P4dims.y+(P4dims.x-1)];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(P4dims.y+1)*(P4dims.x+2)] = P4[batch*P4dims.x*P4dims.y+(P4dims.y-1)*P4dims.x];
    p4Pad[batch*(P4dims.x+2)*(P4dims.y+2)+(P4dims.y+1)*(P4dims.x+2)+(P4dims.x+1)] = P4[batch*P4dims.x*P4dims.y+
        (P4dims.y-1)*P4dims.x+
        (P4dims.x-1)];
  }

  if (x < P5dims.x && y < P5dims.y) {
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(y+1)*(P5dims.x+2)+(x+1)] = P5[batch*P5dims.x*P5dims.y+y*P5dims.x+x];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(x+1)] = P5[batch*P5dims.x*P5dims.y+x];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(P5dims.y+1)*(P5dims.x+2)+(x+1)] = P5[batch*P5dims.x*P5dims.y+
        (P5dims.y-1)*P5dims.x+x];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(y+1)*(P5dims.x+2)] = P5[batch*P5dims.x*P5dims.y+y*P5dims.x];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(y+1)*(P5dims.x+2)+(P5dims.x+1)] = P5[batch*P5dims.x*P5dims.y+
        y*P5dims.x+(P5dims.x-1)];

    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)] = P5[batch*P5dims.x*P5dims.y];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(P5dims.x+1)] = P5[batch*P5dims.x*P5dims.y+(P5dims.x-1)];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(P5dims.y+1)*(P5dims.x+2)] = P5[batch*P5dims.x*P5dims.y+(P5dims.y-1)*P5dims.x];
    p5Pad[batch*(P5dims.x+2)*(P5dims.y+2)+(P5dims.y+1)*(P5dims.x+2)+(P5dims.x+1)] = P5[batch*P5dims.x*P5dims.y+
        (P5dims.y-1)*P5dims.x+
        (P5dims.x-1)];
  }
}

cudaError_t roiOcrAlign_v2(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

                           const void* rois, const void* const layers[], const xy_t* layerDims, const OcrWorkROIAlignSpace& roialginOffset,

                           void* workspace, void* pooled, const xy_t poolDims, bool pad_border) {
  int8_t* wsPtr = static_cast<int8_t*>(workspace);
  void* p2PadPtr = wsPtr + roialginOffset.p2PadOffset;
  void* p3PadPtr = wsPtr + roialginOffset.p3PadOffset;
  void* p4PadPtr = wsPtr + roialginOffset.p4PadOffset;
  void* p5PadPtr = wsPtr + roialginOffset.p5PadOffset;

  // GpuTimer gpu_timer;
  // gpu_timer.Start();
  if (pad_border) {
    dim3 block(32, 32);
    dim3 grid((layerDims[0].x - 1) / block.x + 1, (layerDims[0].y - 1) / block.y + 1, batchSize * featureCount);

    featurePad_kernel<<<grid, block, 0, stream>>>(
      static_cast<const float*>(layers[0]), layerDims[0], static_cast<const float*>(layers[1]), layerDims[1],
      static_cast<const float*>(layers[2]), layerDims[2], static_cast<const float*>(layers[3]), layerDims[3],
      static_cast<float*>(p2PadPtr), static_cast<float*>(p3PadPtr),
      static_cast<float*>(p4PadPtr), static_cast<float*>(p5PadPtr));
  }
  // gpu_timer.Stop();
  // printf("featurePad: %f ", (float(gpu_timer.ElapsedMillis())));

  // todo: 不能超过硬件所支持的最大线程数, 需要确定设备每个block最大的线程数和每个grid最大的block数
  const int thread_per_block = BLOCK_MAX_THREADS;
  const int virtual_thread_count = batchSize * roiCount * featureCount * poolDims.x * poolDims.y;
  const int block_count = DivUp(virtual_thread_count, thread_per_block);
  const dim3 blocks(block_count);
  const int threads(thread_per_block);

  // gpu_timer.Start();
  if (pad_border) {
    xy_t p2Dims(layerDims[0].y + 2, layerDims[0].x + 2);
    xy_t p3Dims(layerDims[1].y + 2, layerDims[1].x + 2);
    xy_t p4Dims(layerDims[2].y + 2, layerDims[2].x + 2);
    xy_t p5Dims(layerDims[3].y + 2, layerDims[3].x + 2);
    roiOcrAlign_kernel_v2<<<blocks, threads, 0, stream>>>(virtual_thread_count, featureCount, firstThreshold,
        static_cast<const float*>(rois),
        static_cast<const float*>(p2PadPtr), p2Dims, static_cast<const float*>(p3PadPtr), p3Dims,
        static_cast<const float*>(p4PadPtr), p4Dims, static_cast<const float*>(p5PadPtr), p5Dims,
        static_cast<float*>(pooled), poolDims, pad_border);
  } else {
    roiOcrAlign_kernel_v2<<<blocks, threads, 0, stream>>>(virtual_thread_count, featureCount, firstThreshold,
        static_cast<const float*>(rois),
        static_cast<const float*>(layers[0]), layerDims[0], static_cast<const float*>(layers[1]), layerDims[1],
        static_cast<const float*>(layers[2]), layerDims[2], static_cast<const float*>(layers[3]), layerDims[3],
        static_cast<float*>(pooled), poolDims, pad_border);
  }
  // gpu_timer.Stop();
  // printf("roiOcrAlign: %f ", (float(gpu_timer.ElapsedMillis())));
  return cudaGetLastError();
}

__global__ void resize_nearest_kernel_2d(int nbatch, float scale, int2 osize, float const* idata, int istride,
    int ibatchstride, float* odata, int ostride, int obatchstride) {

  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  for (int batch = z0; batch < nbatch; batch += gridDim.z) {
    for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
      for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
        int ix = int(ox / scale);
        int iy = int(oy / scale);
        odata[batch * obatchstride + oy * ostride + ox] = idata[batch * ibatchstride + iy * istride + ix];
      }
    }
  }
}

void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
                   int istride, int ibatchstride, float* odata, int ostride, int obatchstride) {

  resize_nearest_kernel_2d<<<grid, block, 0, stream>>>(
    nbatch, scale, osize, idata, istride, ibatchstride, odata, ostride, obatchstride);
}

struct BOX {
  float y1, x1, y2, x2;
};

struct DETECTION {
  float y1, x1, y2, x2, class_id, score;
};

struct OCRDETECTION {
  float y1, x1, y2, x2, class_id, score, cos, sin;
};

__global__ void ocr_specialslice_kernel(int samples, const void* idata, void* odata) {

  int N = blockIdx.x;
  int blockOffset = N * samples;
  int totalItems = (samples + (blockDim.x - 1)) / blockDim.x;
  const OCRDETECTION* in_detections = static_cast<const OCRDETECTION*>(idata);
  BOX* out_bboxes = static_cast<BOX*>(odata);

  for (int i = 0; i < totalItems; i++) {
    int cur_id = i * blockDim.x + threadIdx.x;

    out_bboxes[blockOffset + cur_id].y1 = in_detections[blockOffset + cur_id].y1;
    out_bboxes[blockOffset + cur_id].x1 = in_detections[blockOffset + cur_id].x1;
    out_bboxes[blockOffset + cur_id].y2 = in_detections[blockOffset + cur_id].y2;
    out_bboxes[blockOffset + cur_id].x2 = in_detections[blockOffset + cur_id].x2;
  }
}

void specialOcrSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata) {
  int blocks = batch_size;
  int threads = dMIN(boxes_cnt, 2048);

  ocr_specialslice_kernel<<<blocks, threads, 0, stream>>>(boxes_cnt, idata, odata);
}

__global__ void resize_bilinear_kernel_2d(int nbatch, float scale, int2 osize, int2 isize, float const* idata, int istride,
    int ibatchstride, float* odata, int ostride, int obatchstride) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  for (int batch = z0; batch < nbatch; batch += gridDim.z) {
    for (int oy = y; oy < osize.y; oy += blockDim.y * gridDim.y) {
      for (int ox = x; ox < osize.x; ox += blockDim.x * gridDim.x) {
        float ix = ox / scale;
        float iy = oy / scale;

        if ((iy < 0 || iy > isize.y - 1) || ((ix < 0 || ix > isize.x - 1))) {
          odata[batch * obatchstride + oy * ostride + ox] = 0;
        } else {
          const int y0 = static_cast<int>(iy);
          const float yAlpha = iy - static_cast<float>(y0);
          const int x0 = static_cast<int>(ix);
          const float xAlpha = ix - static_cast<float>(x0);

          assert(x0 < isize.x);
          assert(y0 < isize.y);

          const int y1 = (yAlpha == 0) ? y0 : y0 + 1;
          const int x1 = (xAlpha == 0) ? x0 : x0 + 1;

          assert(x1 < isize.x);
          assert(y1 < isize.y);

          const float src00 = idata[batch * ibatchstride + y0 * istride + x0];
          const float src01 = idata[batch * ibatchstride + y0 * istride + x1];
          const float src10 = idata[batch * ibatchstride + y1 * istride + x0];
          const float src11 = idata[batch * ibatchstride + y1 * istride + x1];

          const float src0 = src00 * (1 - xAlpha) + src01 * xAlpha;
          const float src1 = src10 * (1 - xAlpha) + src11 * xAlpha;

          odata[batch * obatchstride + oy * ostride + ox] = src0 * (1 - yAlpha) + src1 * yAlpha;
        }
      }
    }
  }
}

void resizeBilinear(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, int2 isize, float const* idata,
                    int istride, int ibatchstride, float* odata, int ostride, int obatchstride) {

  resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>(
    nbatch, scale, osize, isize, idata, istride, ibatchstride, odata, ostride, obatchstride);
}
