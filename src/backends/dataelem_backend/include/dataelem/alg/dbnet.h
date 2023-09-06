#ifndef DATAELEM_ALG_DBNET_H_
#define DATAELEM_ALG_DBNET_H_

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

#include "ext/ppocr/postprocess_op.h"
#include "ext/ppocr/preprocess_op.h"

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class DBNet : public Algorithmer {
 public:
  DBNet() = default;
  ~DBNet() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

 protected:
  // int max_side_len_ = 960;
  // double det_db_thresh_ = 0.3;
  // double det_db_box_thresh_ = 0.5;
  // double det_db_unclip_ratio_ = 2.0;
  // std::string det_db_score_mode_ = "slow";

  int max_side_len_ = 736;
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.6;
  double det_db_unclip_ratio_ = 1.5;
  std::string det_db_score_mode_ = "slow";

  bool use_dilation_ = false;
  bool visualize_ = true;

  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  bool is_scale_ = true;

  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
};

// class DBNetPrep : public DBNet {
//  public:
//   TRITONSERVER_Error* Execute(AlgRunContext* context);

//  private:
//   // pre-process
//   PaddleOCR::ResizeImgType0 resize_op_;
//   PaddleOCR::Normalize normalize_op_;
//   PaddleOCR::Permute permute_op_;
// };

class DBNetPost : public DBNet {
 public:
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  // post-process
  PaddleOCR::PostProcessor post_processor_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_DBNET_H_
