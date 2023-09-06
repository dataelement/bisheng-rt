#ifndef BLS_BACKEND_ALG_ALGORITHMS_IMGDECODE_H_
#define BLS_BACKEND_ALG_ALGORITHMS_IMGDECODE_H_

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

class ImgDecode : public Algorithmer {
 public:
  ImgDecode() = default;
  ~ImgDecode() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  float max_side_len_ = 960.0f;
  ;
  StepConfig graph_io_names_ = {{"BIN_IMG"}, {"ORI_IMG", "SHAPE_LIST"}};
};

}}  // namespace dataelem::alg

#endif  // BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_
