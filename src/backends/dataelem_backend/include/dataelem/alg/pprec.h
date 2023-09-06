#ifndef BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_
#define BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_

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

class PPRec : public Algorithmer {
 public:
  PPRec() = default;
  ~PPRec() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);
  std::vector<std::string> ReadDict(const std::string& path);

 protected:
  int max_w_ = 1200;
  int min_w_ = 320;
  int img_h_ = 48;
  int batch_size_ = 32;
  std::vector<int> batch_sizes_;
  std::vector<int> w_sizes_;
  float thresh_ = 0.9;
  bool output_matrix_ = false;
  bool fixed_batch_ = false;
  std::vector<std::string> label_list_;
  int process_type_ = 0;
  int version_ = 1;
  ;

  StringList sub_graph_names_ = {"rec_ch_graph"};
  StepConfig sub_graph_io_names_ = {{"x"}, {"feat_ind", "feat_prob"}};
};

class PPRecCh : public PPRec {
 public:
  PPRecCh() = default;
  ~PPRecCh() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

 protected:
  StringList sub_graph_names_ = {"rec_ch_graph"};
};

class PPRecLatin : public PPRec {
 public:
  PPRecLatin() = default;
  ~PPRecLatin() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

 protected:
  StringList sub_graph_names_ = {"rec_latin_graph"};
};

}}  // namespace dataelem::alg

#endif  // BLS_BACKEND_ALG_ALGORITHMS_PPREC_H_
