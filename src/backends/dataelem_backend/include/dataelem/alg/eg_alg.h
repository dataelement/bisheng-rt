#ifndef DATAELEM_ALG_EG_ALG_H_
#define DATAELEM_ALG_EG_ALG_H_

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {


class EgAlg : public Algorithmer {
 public:
  EgAlg() = default;
  ~EgAlg() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state)
  {
    Algorithmer::init(model_state);
    graph_names_ = {"addsub_python", "addsub_tf"};
    alg_name_ = "EgAlg";
    return nullptr;
  };

  virtual TRITONSERVER_Error* Execute(AlgRunContext* context);
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_EG_ALG_H_
