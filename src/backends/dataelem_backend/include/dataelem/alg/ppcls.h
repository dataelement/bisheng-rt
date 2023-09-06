#ifndef BLS_BACKEND_ALG_ALGORITHMS_PPCLS_H_
#define BLS_BACKEND_ALG_ALGORITHMS_PPCLS_H_

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

class PPClsAngle : public Algorithmer {
 public:
  PPClsAngle() = default;
  ~PPClsAngle() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  int max_w_ = 192;
  int min_w_ = 192;
  int img_h_ = 48;
  int batch_size_ = 32;
  std::vector<int> batch_sizes_;
  bool fixed_batch_ = false;
  int version_ = 1;
  ;

  StringList sub_graph_names_ = {"cls_angle_graph"};
  StepConfig sub_graph_io_names_ = {{"x"}, {"save_infer_model/scale_0.tmp_1"}};
};

}}  // namespace dataelem::alg

#endif  // BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_
