#include "dataelem/apps/bls_app.h"

#include "dataelem/common/apidata.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/app_factory.h"

namespace dataelem { namespace alg {

REGISTER_APP_CLASS(BLSApp);

TRITONSERVER_Error*
BLSApp::Execute(AlgRunContext* context, std::string* resp)
{
  AppRequestInfo info;
  auto timer = triton::common::Timer();

  // deserialize from the input json.
  OCTensor* input = nullptr;
  context->GetTensor("INPUT", &input);
  auto data_ptr = const_cast<char*>(input->data_ptr());
  rapidjson::Document d;
  d.ParseInsitu(data_ptr);
  if (d.HasParseError()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "json parsing error on string");
  }

  APIData ad_data(d);
  cv::Mat input0, input1;
  get_mat_from_ad(ad_data, "INPUT0", input0);
  get_mat_from_ad(ad_data, "INPUT1", input1);

  int n = input0.rows;
  OCTensorList inputs = {input0, input1};
  inputs[0].set_shape({n});
  inputs[1].set_shape({n});

  OCTensorList outputs;
  StringList input_names = {"INPUT0", "INPUT1"};
  StringList output_names = {"OUTPUT0", "OUTPUT1"};

  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(2);
  GraphExecuate(
      graph_names_[0], context, input_names, output_names, inputs, &futures[0]);
  GraphExecuate(
      graph_names_[1], context, input_names, output_names, inputs, &futures[1]);

  auto* resp1 = futures[0].get();
  ParseTensorsFromServerResponse(resp1, {"OUTPUT0"}, &outputs);
  GraphInferResponseDelete(resp1);

  auto* resp2 = futures[1].get();
  ParseTensorsFromServerResponse(resp2, {"OUTPUT1"}, &outputs);
  GraphInferResponseDelete(resp2);

  rapidjson::StringBuffer buffer;
  info.request_id = 1;
  info.elapse = timer.toc();
  WriteOKResponse(&buffer, outputs, output_names, &info);
  *resp = buffer.GetString();

  return nullptr;
}


}}  // namespace dataelem::alg
