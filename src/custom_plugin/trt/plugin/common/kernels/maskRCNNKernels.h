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

#ifndef TRT_MASKRCNN_UTILS_H
#define TRT_MASKRCNN_UTILS_H

#include "NvInfer.h"
#include "plugin.h"
#include <sys/time.h>

using namespace nvinfer1;

const int BLOCK_MAX_THREADS = 1024;

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

inline int DivUp(int a, int b) {
  return (a + b - 1) / b;
}

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline size_t nAlignUp(size_t x, size_t align) {
  size_t mask = align - 1;
  assert((align & mask) == 0); // power of 2
  return (x + mask) & (~mask);
}

inline size_t nAlignDown(size_t x, size_t align) {
  size_t mask = align - 1;
  assert((align & mask) == 0); // power of 2
  return (x) & (~mask);
}

inline size_t dimVolume(const nvinfer1::Dims& dims) {
  size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i)
    volume *= dims.d[i];

  return volume;
}

inline size_t typeSize(const nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kHALF:
    return sizeof(uint16_t);
  case nvinfer1::DataType::kINT8:
    return sizeof(uint8_t);
  case nvinfer1::DataType::kINT32:
    return sizeof(uint32_t);
  default:
    return 0;
  }
}

#define AlignMem(x) nAlignUp(x, 256)

template <typename Dtype>
struct CudaBind {
  size_t mSize;
  void* mPtr;

  CudaBind(size_t size) {
    mSize = size;
    CUASSERT(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
  }

  ~CudaBind() {
    if (mPtr != nullptr) {
      CUASSERT(cudaFree(mPtr));
      mPtr = nullptr;
    }
  }
};

struct xy_t {
  int y;
  int x;

  xy_t()
    : y(0)
    , x(0) {
  }
  xy_t(int y_, int x_)
    : y(y_)
    , x(x_) {
  }
};

struct RefineNMSParameters {
  int backgroundLabelId, numClasses, keepTopK;
  float scoreThreshold, iouThreshold;
};

struct RefineDetectionWorkSpace {
  RefineDetectionWorkSpace(
    const int batchSize, const int sampleCount, const RefineNMSParameters& param, const nvinfer1::DataType type);

  RefineDetectionWorkSpace() = default;

  nvinfer1::DimsHW argMaxScoreDims;
  nvinfer1::DimsHW argMaxBboxDims;
  nvinfer1::DimsHW argMaxLabelDims;
  nvinfer1::DimsHW sortClassScoreDims;
  nvinfer1::DimsHW sortClassLabelDims;
  nvinfer1::DimsHW sortClassSampleIdxDims;
  nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}};
  nvinfer1::DimsHW sortClassPosDims;
  nvinfer1::DimsHW sortNMSMarkDims;

  size_t argMaxScoreOffset = 0;
  size_t argMaxBboxOffset = 0;
  size_t argMaxLabelOffset = 0;
  size_t sortClassScoreOffset = 0;
  size_t sortClassLabelOffset = 0;
  size_t sortClassSampleIdxOffset = 0;
  size_t sortClassValidCountOffset = 0;
  size_t sortClassPosOffset = 0;
  size_t sortNMSMarkOffset = 0;
  size_t totalSize = 0;
};

struct OcrProposalWorkSpace {
  OcrProposalWorkSpace(const int batchSize, const int inputCnt, const int sampleCount, const RefineNMSParameters& param,
                       const int numSegment, const nvinfer1::DataType type);

  OcrProposalWorkSpace() = default;

  nvinfer1::DimsHW preRefineSortedScoreDims;
  nvinfer1::DimsHW preRefineBboxDims;
  nvinfer1::DimsHW preRefineSortedScoreTopDims;
  nvinfer1::DimsHW preRefineBboxTopDims;
  nvinfer1::DimsHW argMaxScoreDims;
  nvinfer1::DimsHW argMaxBboxDims;
  nvinfer1::DimsHW argMaxLabelDims;
  nvinfer1::DimsHW sortClassScoreDims;
  nvinfer1::DimsHW sortClassLabelDims;
  nvinfer1::DimsHW sortClassSampleIdxDims;
  nvinfer1::Dims sortClassValidCountDims = {1, {1, 0}};
  nvinfer1::DimsHW sortClassPosDims;
  nvinfer1::DimsHW sortNMSMarkDims;

  size_t tempStorageOffset = 0;
  size_t preRefineSortedScoreOffset = 0;
  size_t preRefineBboxOffset = 0;
  size_t preRefineSortedScoreTopOffset = 0;
  size_t preRefineBboxTopOffset = 0;
  size_t argMaxScoreOffset = 0;
  size_t argMaxBboxOffset = 0;
  size_t argMaxLabelOffset = 0;
  size_t sortClassScoreOffset = 0;
  size_t sortClassLabelOffset = 0;
  size_t sortClassSampleIdxOffset = 0;
  size_t sortClassValidCountOffset = 0;
  size_t sortClassPosOffset = 0;
  size_t sortNMSMarkOffset = 0;
  size_t totalSize = 0;
};

struct OcrWorkROIAlignSpace {
  OcrWorkROIAlignSpace(const int batchSize, const int mFeatureLength, const xy_t* mFeatureSpatialSize, bool mPadBorder);

  OcrWorkROIAlignSpace() = default;

  size_t p2PadOffset = 0;
  size_t p3PadOffset = 0;
  size_t p4PadOffset = 0;
  size_t p5PadOffset = 0;
  size_t totalSize = 0;
};

cudaError_t OcrRefineBatchClassNMS(cudaStream_t stream, int N, int samples, nvinfer1::DataType dtype, int max_size,
                                   const RefineNMSParameters& param, const RefineDetectionWorkSpace& refineOffset, void* workspace,
                                   const void* inScores, const void* inDelta, const void* inCountValid, const void* inROI,
                                   const void* inCos, const void* inSin, void* outDetections);

cudaError_t OcrProposalRefineBatchClassNMS(cudaStream_t stream, int N,
    int inputCnt, // candidate anchors
    int samples,  // preNMS_topK
    int max_size,
    int mSegments, // num_segments
    nvinfer1::DataType dtype, const RefineNMSParameters& param, const OcrProposalWorkSpace& proposalOffset,
    void* workspace, const void* inScores, const void* inDelta, const void* inCountValid, const void* inAnchors,
    void* outProposals);

cudaError_t OcrDecodeBox(cudaStream_t stream, int N, int samples, int max_size, int cascade_stage, nvinfer1::DataType dtype,
                         const void* inDelta, const void* inROI, void* outDetections);

cudaError_t OcrApplyDelta2Bboxes(cudaStream_t stream, int N,
                                 int samples,         // number of anchors per image
                                 int max_size,
                                 int cascade_stage,
                                 const void* anchors, // [N, anchors, (x1, y1, x2, y2)]
                                 const void* delta,   //[N, anchors, (dx, dy, log(dw), log(dh)]
                                 void* outputBbox, bool weight_Flag);

cudaError_t OcrApplyDelta2Bboxes_v2(cudaStream_t stream, int N,
                                    int samples,         // number of anchors per image
                                    int max_size,
                                    int cascade_stage,
                                    const void* anchors, // [N, anchors, (x1, y1, x2, y2)]
                                    const void* delta,   //[N, anchors, (dx, dy, log(dw), log(dh)]
                                    void* outputBbox, bool weight_Flag);

// PYRAMID ROIALIGN
cudaError_t roiOcrAlign(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

                        const void* rois, const void* const layers[], const xy_t* layerDims,

                        void* pooled, const xy_t poolDims, bool pad_border);

cudaError_t roiOcrAlign_v2(cudaStream_t stream, int batchSize, int featureCount, int roiCount, float firstThreshold,

                           const void* rois, const void* const layers[], const xy_t* layerDims, const OcrWorkROIAlignSpace& roialginOffset,

                           void* workspace, void* pooled, const xy_t poolDims, bool pad_border);

// RESIZE NEAREST
void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
                   int istride, int ibatchstride, float* odata, int ostride, int obatchstride);
// SPECIAL SLICE
void specialSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata);
void specialOcrSlice(cudaStream_t stream, int batch_size, int boxes_cnt, const void* idata, void* odata);

void resizeBilinear(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, int2 isize, float const* idata,
                    int istride, int ibatchstride, float* odata, int ostride, int obatchstride);


#endif // TRT_MASKRCNN_UTILS_H
