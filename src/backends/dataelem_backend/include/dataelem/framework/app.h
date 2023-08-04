#ifndef DATAELEM_FRAMEWORK_APP_H_
#define DATAELEM_FRAMEWORK_APP_H_

#include "dataelem/common/apidata.h"
#include "dataelem/framework/alg_utils.h"
#include "dataelem/framework/iop.h"

#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace dataelem { namespace alg {

class Application : public IApp {
 public:
  Application() = default;
  virtual ~Application() = default;
  virtual TRITONSERVER_Error* init(triton::backend::BackendModel* model_state)
  {
    graph_executor_ =
        std::make_unique<GraphExecutor>(model_state->TritonServer());
    server_ = model_state->TritonServer();
    return nullptr;
  };
  virtual std::string Name() { return app_name_; }
  virtual TRITONSERVER_Error* Execute(
      AlgRunContext* context, std::string* resp) = 0;

 protected:
  virtual TRITONSERVER_Error* GraphExecuate(
      const std::string& graph_name, AlgRunContext* context,
      const StringList& input_names, const StringList& output_names,
      const OCTensorList& inputs,
      std::future<TRITONSERVER_InferenceResponse*>* future);

 protected:
  std::string app_name_;
  std::unique_ptr<GraphExecutor> graph_executor_;
  TRITONSERVER_Server* server_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_APP_H_
