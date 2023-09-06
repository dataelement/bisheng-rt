#ifndef BLS_BACKEND_ALG_ALGORITHMS_PPDET_H_
#define BLS_BACKEND_ALG_ALGORITHMS_PPDET_H_

// #pragma once
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "dataelem/framework/alg.h"
#include "ext/ppocr/postprocess_op.h"
#include "ext/ppocr/preprocess_op.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace dataelem { namespace alg {

class PPDetDBPrep : public Algorithmer {
 public:
  PPDetDBPrep() = default;
  ~PPDetDBPrep() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  int max_side_len_ = 960;
  bool fixed_shape_ = false;
  std::vector<int> min_side_lens_;
  int version_ = 1;
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
};

class PPDetDBPost : public Algorithmer {
 public:
  PPDetDBPost() = default;
  ~PPDetDBPost() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

  void EnlageBbox(
      std::vector<std::vector<std::vector<int>>>& bbox, int delta_w,
      int delta_h);

 protected:
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.6;
  double det_db_unclip_ratio_ = 1.5;
  std::string det_db_score_mode_ = "fast";
  bool use_dilation_ = false;
  float delta_w_ = 0;
  float delta_h_ = 0;
  int version_ = 1;

  PaddleOCR::PostProcessor post_processor_;
};

}}  // namespace dataelem::alg

#endif  // BLS_BACKEND_ALG_ALGORITHMS_DBNET_H_
