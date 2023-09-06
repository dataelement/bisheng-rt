#include "dataelem/alg/distribution.h"

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(Distribution);

TRITONSERVER_Error* 
Distribution::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  graph_names_ = {"Distribution"};
  alg_name_ = "Distribution";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "base_graph_name", &base_graph_name_);
    
  std::string input_names_str;
  std::vector<std::string> input_names_vec;
  SafeParseParameter(params, "graph_input_name", &input_names_str, "");
  ParseArrayFromString(input_names_str, input_names_vec);
  if(input_names_vec.size() > 0 && input_names_vec[0].size() > 0){
    graph_io_names_.input_names = input_names_vec;
  }

  std::string output_names_str;
  std::vector<std::string> output_names_vec;
  SafeParseParameter(params, "graph_output_name", &output_names_str, "");
  ParseArrayFromString(output_names_str, output_names_vec);
  if(output_names_vec.size() > 0 && output_names_vec[0].size() > 0){
    graph_io_names_.output_names = output_names_vec;
  }

    return nullptr;
  };

TRITONSERVER_Error*
Distribution::Execute(AlgRunContext* context)
{
  OCTensorList inputs, outputs;

  context->GetTensor(graph_io_names_.input_names, inputs);
  TRITONSERVER_InferenceRequest* graph_request = nullptr;
  CreateServerRequestWithTensors(
      context->GetBackendRequestInfo(), graph_executor_->GetServer(),
      (base_graph_name_ + "__" + std::to_string(instance_id_)).c_str(), &inputs,
      graph_io_names_.input_names, graph_io_names_.output_names, &graph_request);

  std::future<TRITONSERVER_InferenceResponse*> future;
  graph_executor_->AsyncExecute(graph_request, context, &future);

  auto* graph_response = future.get();
  ParseTensorsFromServerResponse(graph_response, graph_io_names_.output_names, &outputs);


  context->SetTensor(graph_io_names_.output_names, std::move(outputs));
  GraphInferResponseDelete(graph_response);
  return nullptr;
}


}}  // namespace dataelem::alg
