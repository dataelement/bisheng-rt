#ifndef DATAELEM_ALG_GENERAL_PREP_H_
#define DATAELEM_ALG_GENERAL_PREP_H_

// #pragma once
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class GeneralPrepOp : public Algorithmer {
 public:
  GeneralPrepOp() = default;
  ~GeneralPrepOp() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  bool use_imdec_ = true;
  bool use_b64dec_ = false;
  bool use_resize_ = false;
  bool use_padding_ = false;
  bool use_b64out_ = false;

  int resize_param_ = 1000;
  int padding_param_ = 0;
  int imdec_param_ = 1;  // 1 color, 0 gray
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_GENERAL_PREP_H_