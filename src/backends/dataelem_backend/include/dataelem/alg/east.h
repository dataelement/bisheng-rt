#ifndef DATAELEM_EAST_H_
#define DATAELEM_ALG_EAST_H_

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class EastV4 : public Algorithmer {
 public:
  EastV4() {}
  ~EastV4() {}

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  float _nms_threshold;
  float _score_threshold;
  bool _use_text_direction;
  std::vector<int> _scale_list;

  StepConfig graph_io_names_ = {
      {"input_images"},
      {"final/score", "final/geometry", "final/cos_map", "final/sin_map"}};
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_EAST_H_
