#include "dataelem/framework/app.h"
#include "dataelem/framework/alg_utils.h"

namespace dataelem { namespace alg {

TRITONSERVER_Error*
Application::GraphExecuate(
    const std::string& graph_name, AlgRunContext* context,
    const StringList& input_names, const StringList& output_names,
    const OCTensorList& inputs,
    std::future<TRITONSERVER_InferenceResponse*>* future)
{
  RETURN_ERROR_IF_FALSE(
      input_names.size() == inputs.size(), TRITONSERVER_ERROR_INTERNAL,
      (std::string("input tensors is wrong in graph:") + graph_name));

  bool ready = false;
  TRITONSERVER_ServerModelIsReady(server_, graph_name.c_str(), -1, &ready);
  if (!ready) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("model " + graph_name + " is not ready").c_str());
  }

  TRITONSERVER_Error* err = nullptr;
  TRITONSERVER_InferenceRequest* graph_request = nullptr;
  err = CreateServerRequestWithTensors(
      context->GetBackendRequestInfo(), graph_executor_->GetServer(),
      graph_name.c_str(), &inputs, input_names, output_names, &graph_request);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request);
    return err;
  }

  err = graph_executor_->AsyncExecute(graph_request, context, future);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request);
    return err;
  }

  return nullptr;
}

}}  // namespace dataelem::alg