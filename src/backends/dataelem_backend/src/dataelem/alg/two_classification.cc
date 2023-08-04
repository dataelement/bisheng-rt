#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

#include "dataelem/alg/two_classification.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(TwoClassification);

TRITONSERVER_Error*
TwoClassification::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "twoclassification";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);
  graph_io_names_ = StepConfig({"images"}, {"labels"});

  return nullptr;
}

TRITONSERVER_Error*
TwoClassification::Execute(AlgRunContext* context)
{
  TRITONSERVER_Error* err = nullptr;
  APIData params;
  OCTensorList tensor_inputs;
  context->GetTensor(io_names_.input_names, tensor_inputs);
  cv::Mat image = tensor_inputs[0].GetImage();
  cv::Mat bboxes = tensor_inputs[1].GetMat();

  MatList inputs;
  OCTensor* patchs_tensor;
  std::vector<absl::string_view> patchs_data;
  if (context->GetTensor(optional_inputs_[0], &patchs_tensor)) {
    patchs_tensor->GetStrings(patchs_data);
    inputs.resize(patchs_data.size());
    for (size_t i = 0; i < patchs_data.size(); i++) {
      err = DecodeImgFromB64(patchs_data[i], inputs[i], cv::IMREAD_COLOR);
      RETURN_IF_ERROR(err);
    }
    params.add("mode", 0);
  } else {
    inputs.emplace_back(image);
    inputs.emplace_back(bboxes);
    params.add("mode", 1);
  }

  MatList prep_outs, graph_outs, post_outs;
  MatList outputs;
  RETURN_IF_ERROR(PreprocessStep(params, inputs, prep_outs));
  RETURN_IF_ERROR(GraphStep(context, prep_outs, graph_outs));
  RETURN_IF_ERROR(PostprocessStep(params, graph_outs, outputs));

  int64_t n = outputs[0].rows;
  OCTensorList tensors = {std::move(outputs[0])};
  tensors[0].set_shape({n});
  context->SetTensor(io_names_.output_names, std::move(tensors));

  return nullptr;
}

TRITONSERVER_Error*
TwoClassification::PreprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  int mode = 1;
  get_ad_value(params, "mode", mode);
  MatList rois;
  const MatList* inputs_ptr = nullptr;
  if (mode == 0) {
    inputs_ptr = &inputs;
  } else {
    getRRectRoisWithPaddingOp5(inputs[0], inputs[1], rois);
    inputs_ptr = &rois;
  }

  std::vector<cv::Mat> inputsList;
  for (unsigned int i = 0; i < inputs_ptr->size(); i++) {
    cv::Mat resize, rgb, dst;
    const cv::Mat& src = inputs_ptr->at(i);
    resizeOp(src, resize, _max_size, _max_size);
    rgb = BGR2RGB(resize);
    rgb.convertTo(dst, CV_32F);
    inputsList.emplace_back(dst);
  }
  cv::Mat out;
  mergeBatchMatOp(inputsList, out);
  outputs.emplace_back(out);

  return nullptr;
}

TRITONSERVER_Error*
TwoClassification::GraphStep(
    AlgRunContext* context, const MatList& inputs, MatList& outputs)
{
  TRITONSERVER_Error* err = nullptr;
  // send request
  const cv::Mat& rois = inputs[0];
  int n = rois.size[0];
  int step0 = rois.step[0];
  int batchs = (int)std::ceil(n * 1.0 / _batch_size);
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(batchs);
  for (int k = 0; k < batchs; k++) {
    int s = k * _batch_size;
    int e = k == (_batch_size - 1) ? n : (k + 1) * _batch_size;
    int sn = e - s;
    std::vector<int> shape = {sn, _max_size, _max_size, 3};
    cv::Mat sub_rois(shape, CV_32F, rois.data + s * step0 * 4);
    OCTensorList input_tensors = {std::move(sub_rois)};
    err = GraphExecuate(
        graph_names_[0], context, graph_io_names_.input_names,
        graph_io_names_.output_names, input_tensors, &futures[k]);
    RETURN_IF_ERROR(err);
  }

  // parse result
  for (int k = 0; k < n; k++) {
    auto* resp = futures[k].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, graph_io_names_.output_names, &outputs));
    outputs.emplace_back(outputs[0].GetMat());
    GraphInferResponseDelete(resp);
  }
  return nullptr;
}

TRITONSERVER_Error*
TwoClassification::PostprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  std::vector<int> result;
  for (size_t i = 0; i < inputs.size(); i++) {
    // mat shape: nx2
    cv::Mat_<float> mat(inputs[i]);
    for (int j = 0; j < mat.rows; j++) {
      result.emplace_back(int(mat(j, 1) > mat(j, 0)));
    }
  }
  int n = result.size();
  outputs.emplace_back(cv::Mat(result).reshape(1, n).clone());
  return nullptr;
}

}}  // namespace dataelem::alg