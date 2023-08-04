#include <array>

#include "dataelem/alg/text_classifier.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(TextClassifier);

TRITONSERVER_Error*
TextClassifier::Execute(AlgRunContext* context)
{
  OCTensorList images;
  RETURN_IF_ERROR(DecodeStep(context, images));

  OCTensorList prep_outs, graph_outs;
  RETURN_IF_ERROR(PreprocessStep(context, images, prep_outs));
  RETURN_IF_ERROR(GraphExecuateStep(context, {prep_outs[0]}, graph_outs));

  graph_outs.emplace_back(images[0]);
  graph_outs.emplace_back(prep_outs[1]);
  RETURN_IF_ERROR(PostprocessStep(context, graph_outs));
  return nullptr;
}

TRITONSERVER_Error*
TextClassifier::init(JValue& model_config, TRITONSERVER_Server* server)
{
  Algorithmer::init(model_config, server);
  alg_name_ = "TextClassifier";

  JValue params;
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

  SafeParseParameter(params, "is_scale", &is_scale_);
  SafeParseParameter(params, "use_tensorrt", &use_tensorrt_);
  SafeParseParameter(params, "precision", &precision_);
  SafeParseParameter(params, "cls_batch_num", &cls_batch_num_);

  // Configs for each step
  decode_step_config_ = StepConfig({}, {});
  prep_step_config_ = StepConfig({"src"}, {"x", "ratio_hw"});
  infer_step_config_ = StepConfig({"x"}, {"save_infer_model/scale_0.tmp_1"});
  post_step_config_ = StepConfig(
      {"sigmoid_0.tmp_0", "src", "ratio_hw"}, {"bboxes", "bbox_scores"});

  return nullptr;
}

void
Normalize(
    cv::Mat* im, const std::vector<float>& mean,
    const std::vector<float>& scale, const bool is_scale)
{
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(*im, bgr_channels);
  for (auto i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(
        bgr_channels[i], CV_32FC1, 1.0 * scale[i], (0.0 - mean[i]) * scale[i]);
  }
  cv::merge(bgr_channels, *im);
}

// Preprocess for the TextClassifier.
//
// Params:
//  context - An AlgRunContext pointer, for the request level shared resource
//  inputs - input tensors
//           Case 1.  one tensor in context, 1) patchs [-1, 48, 192, 3]
//                    usually for internal request
//           Case 2.  one tensor in context, [b64_image+]
//           Case 3.  two tensors in context, 1) b64_image 2) bboxes, [n, 4, 2]
//  outputs - output tensors, [dst,scale]
// Returns:
TRITONSERVER_Error*
TextClassifier::PreprocessStep(
    AlgRunContext* context, const OCTensorList& inputs, OCTensorList& outputs)
{
  // Case 1
  OCTensor patchs;
  if (context->GetTensor("patchs", patchs)) {
    const cv::Mat& imgs = patchs.m();
    normalize_op_.Run(&imgs, mean_, scale_, is_scale_);


    return nullptr;
  };

  OCTensor *b64_image, bboxes;
  bool has_b64_image = context->GetTensor("b64_image", &b64_image);
  bool has_bboxes = context->GetTensor("bboxes", &bboxes);
  // Case 2.
  if (has_b64_image && !has_bboxes) {
    // todo
    void(0);
  };

  // Case 2.
  if (has_b64_image && has_bboxes) {
    // todo
    void(0);
  };
  return nullptr;
}

TRITONSERVER_Error*
TextClassifier::PostprocessStep(AlgRunContext* context, OCTensorList& inputs)
{
  return nullptr;
}

}}  // namespace dataelem::alg