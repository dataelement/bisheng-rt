#ifndef BLS_BACKEND_ALG_ALGORITHMS_CROPCONCAT_H_
#define BLS_BACKEND_ALG_ALGORITHMS_CROPCONCAT_H_

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
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace dataelem { namespace alg {

class CropConcat : public Algorithmer {
 public:
  CropConcat() = default;
  ~CropConcat() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  int max_w_ = 1200;
  int img_h_ = 48;
  float hw_thrd_ = 1.5;
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool rot_bbox_ = true;
  StepConfig graph_io_names_ = {
      {"ORI_IMG", "DET_BBOX", "DET_BBOX_SCORE"},
      {"PROCESSED_IMG", "PROCESSED_IMG_WIDTH", "DET_BBOX_NEW",
       "DET_BBOX_SCORE_NEW", "PROCESSED_ROT"}};
};

}}  // namespace dataelem::alg

#endif  // BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_
