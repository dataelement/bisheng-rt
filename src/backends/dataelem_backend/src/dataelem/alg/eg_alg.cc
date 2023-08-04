#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

#include "dataelem/alg/eg_alg.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(EgAlg);

TRITONSERVER_Error*
EgAlg::Execute(AlgRunContext* context)
{
  OCTensorList inputs, outputs;
  StringList input_names = {"INPUT0", "INPUT1"};
  StringList output_names = {"OUTPUT0", "OUTPUT1"};

  TRITONSERVER_Error* err = nullptr;
  context->GetTensor(input_names, inputs);
  TRITONSERVER_InferenceRequest* graph_request_1 = nullptr;
  err = CreateServerRequestWithTensors(
      context->GetBackendRequestInfo(), graph_executor_->GetServer(),
      graph_names_[0].c_str(), &inputs, input_names, output_names,
      &graph_request_1);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request_1);
    return err;
  }

  TRITONSERVER_InferenceRequest* graph_request_2 = nullptr;
  err = CreateServerRequestWithTensors(
      context->GetBackendRequestInfo(), graph_executor_->GetServer(),
      graph_names_[1].c_str(), &inputs, input_names, output_names,
      &graph_request_2);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request_2);
    return err;
  }

  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(2);
  graph_executor_->AsyncExecute(graph_request_1, context, &futures[0]);
  graph_executor_->AsyncExecute(graph_request_2, context, &futures[1]);

  auto* graph_response_1 = futures[0].get();
  ParseTensorsFromServerResponse(graph_response_1, {"OUTPUT0"}, &outputs);
  GraphInferResponseDelete(graph_response_1);

  auto* graph_response_2 = futures[1].get();
  ParseTensorsFromServerResponse(graph_response_2, {"OUTPUT1"}, &outputs);
  GraphInferResponseDelete(graph_response_2);

  context->SetTensor(output_names, std::move(outputs));
  return nullptr;
}


}}  // namespace dataelem::alg
