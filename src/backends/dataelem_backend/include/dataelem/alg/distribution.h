#ifndef DATAELEM_DISTRIBUTION_ALG_H_
#define DATAELEM_DISTRIBUTION_ALG_H_

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {


class Distribution : public Algorithmer {
 public:
  Distribution() = default;
  ~Distribution() = default;

  void set_instance_id(int instance_id)
  {
    instance_id_ = instance_id;
    //std::cout<<"distribution, instance id:"<<instance_id_<<std::endl;
  }

 protected:
  std::string base_graph_name_ = "rec_ch_graph";
  int instance_id_ = 0;
  StepConfig graph_io_names_ = {{"x"}, {"feat_ind", "feat_prob"}};

  virtual TRITONSERVER_Error* init(triton::backend::BackendModel* model_state); 

  virtual TRITONSERVER_Error* Execute(AlgRunContext* context);
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_REQ_SPLIT_ALG_H_
