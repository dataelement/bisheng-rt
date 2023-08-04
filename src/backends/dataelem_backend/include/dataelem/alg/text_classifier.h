#ifndef DATAELEM_ALG_CLASSIFIER_H_
#define DATAELEM_ALG_CLASSIFIER_H_

// #pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "ext/ppocr/preprocess_op.h"
#include "ext/ppocr/utility.h"

namespace dataelem { namespace alg {

class TextClassifier : public Algorithmer {
 public:
  TextClassifier() = default;
  ~TextClassifier() = default;

  TRITONSERVER_Error* init(JValue& model_config, TRITONSERVER_Server* server);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  virtual TRITONSERVER_Error* PreprocessStep(
      AlgRunContext* context, const OCTensorList& inputs,
      OCTensorList& outputs);

  virtual TRITONSERVER_Error* GraphExecuateStep(
      AlgRunContext* context, const OCTensorList& inputs,
      OCTensorList& outputs);

  virtual TRITONSERVER_Error* PostprocessStep(
      AlgRunContext* context, OCTensorList& inputs);

 private:
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  int cls_batch_num_ = 1;
  std::vector<int> cls_image_shape_ = {3, 48, 192};

  // pre-process
  PaddleOCR::ClsResizeImg resize_op_;
  PaddleOCR::Normalize normalize_op_;
  PaddleOCR::PermuteBatch permute_op_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_CLASSIFIER_H_