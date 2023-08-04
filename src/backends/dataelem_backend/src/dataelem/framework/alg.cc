#include "absl/strings/escaping.h"

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_utils.h"

#include "dataelem/framework/alg.h"


namespace dataelem { namespace alg {

TRITONSERVER_Error*
Algorithmer::GraphExecuateStep(
    AlgRunContext* context, const StringList& input_names,
    const StringList& output_names, const OCTensorList& inputs,
    OCTensorList& outputs)
{
  RETURN_ERROR_IF_FALSE(
      input_names.size() == inputs.size(), TRITONSERVER_ERROR_INTERNAL,
      (std::string("graph input tensors wrong in Alg:") + alg_name_));

  TRITONSERVER_Error* err = nullptr;
  TRITONSERVER_InferenceRequest* graph_request = nullptr;
  err = CreateServerRequestWithTensors(
      context->GetBackendRequestInfo(), graph_executor_->GetServer(),
      graph_names_[0].c_str(), &inputs, input_names, output_names,
      &graph_request);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request);
    return err;
  }

  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(1);
  err = graph_executor_->AsyncExecute(graph_request, context, &futures[0]);
  if (err != nullptr) {
    TRITONSERVER_InferenceRequestDelete(graph_request);
    return err;
  }

  auto* graph_response = futures[0].get();
  err = ParseTensorsFromServerResponse(graph_response, output_names, &outputs);
  GraphInferResponseDelete(graph_response);
  RETURN_IF_ERROR(err);

  return nullptr;
}

TRITONSERVER_Error*
Algorithmer::DecodeStep(AlgRunContext* context)
{
  StringList& input_names = enc_dec_io_names_.input_names;
  StringList& output_names = enc_dec_io_names_.output_names;
  if (input_names.size() == 0) {
    return nullptr;
  }

  RETURN_ERROR_IF_FALSE(
      input_names.size() == output_names.size(), TRITONSERVER_ERROR_INTERNAL,
      (std::string("wrong config in enc/dec ios, alg:") + alg_name_));

  OCTensor* tensor = nullptr;
  for (size_t i = 0; i < input_names.size(); i++) {
    RETURN_ERROR_IF_FALSE(
        context->GetTensor(input_names[i], &tensor),
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to get decode input tensors in Alg:") +
         alg_name_));

    std::string image_raw_bytes;
    if (absl::Base64Unescape(tensor->GetString(0), &image_raw_bytes)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("base64 decode failed in alg:" + alg_name_).c_str());
    }

    cv::Mat src = imdecodeOp(image_raw_bytes, -1);
    RETURN_ERROR_IF_TRUE(
        src.empty(), TRITONSERVER_ERROR_INTERNAL,
        "imdecode failed in alg:" + alg_name_);
    context->SetTensor(input_names[i], std::move(src));
  }

  return nullptr;
}

TRITONSERVER_Error*
Algorithmer::UpdateIONames(JValue& config)
{
  StringList input_names, output_names;
  {
    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(config.MemberAsArray("input", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

      // Notice: optional inputs put in optional_inputs_, get it manually
      if (io.Find("optional")) {
        bool is_optional = false;
        RETURN_IF_ERROR(io.MemberAsBool("optional", &is_optional));
        if (is_optional) {
          optional_inputs_.emplace_back(io_name);
          continue;
        }
      }

      input_names.emplace_back(io_name);
    }
  }
  {
    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(config.MemberAsArray("output", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      output_names.emplace_back(io_name);
    }
  }
  io_names_ = StepConfig(input_names, output_names);

  JValue params;
  if (!config.Find("parameters", &params)) {
    return nullptr;
  }

  std::string enc_input_names_str;
  SafeParseParameter(params, "enc_input_names", &enc_input_names_str, "");
  StringList enc_input_names;
  ParseArrayFromString(enc_input_names_str, enc_input_names);

  std::string dec_input_names_str;
  SafeParseParameter(params, "dec_input_names", &dec_input_names_str, "");
  StringList dec_input_names;
  ParseArrayFromString(dec_input_names_str, dec_input_names);

  enc_dec_io_names_ = StepConfig(enc_input_names, dec_input_names);
  return nullptr;
}

TRITONSERVER_Error*
Algorithmer::GraphExecuate(
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