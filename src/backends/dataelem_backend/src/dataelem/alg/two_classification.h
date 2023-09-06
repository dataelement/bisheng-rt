#ifndef DATAELEM_ALG_TWO_CLASSIFICATION_H_
#define DATAELEM_ALG_TWO_CLASSIFICATION_H_

#include "dataelem/common/apidata.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class TwoClassification : public Algorithmer {
 public:
  TwoClassification() = default;
  ~TwoClassification() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  TRITONSERVER_Error* GraphStep(
      AlgRunContext* context, const MatList& inputs, MatList& outputs);

  TRITONSERVER_Error* PreprocessStep(
      const APIData& params, const MatList& inputs, MatList& outputs);

  TRITONSERVER_Error* PostprocessStep(
      const APIData& params, const MatList& inputs, MatList& outputs);

 private:
  int _max_size = 224;
  int _batch_size = 32;
  StepConfig graph_io_names_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_TWO_CLASSIFICATION_H_
