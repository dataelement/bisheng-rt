#include "dataelem/alg/imgdecode.h"

#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(ImgDecode);

TRITONSERVER_Error*
ImgDecode::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "img_decode";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "max_side_len", &max_side_len_);

  return nullptr;
}

TRITONSERVER_Error*
ImgDecode::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  auto img_bin = inputs[0].m();
  cv::Mat ori_img = cv::imdecode(img_bin, CV_LOAD_IMAGE_COLOR);

  std::vector<int> ori_imgs_shape = {1, ori_img.size[0], ori_img.size[1], 3};
  cv::Mat ori_imgs = cv::Mat(ori_imgs_shape, CV_8U, ori_img.data).clone();

  float h = float(ori_img.size[0]);
  float w = float(ori_img.size[1]);
  float r = max_side_len_ / std::max(h, w);

  float resize_h = std::floor(h * r);
  float resize_w = std::floor(w * r);
  resize_h = std::max(std::nearbyint(resize_h / 32.0) * 32.0, 32.0);
  resize_w = std::max(std::nearbyint(resize_w / 32.0) * 32.0, 32.0);

  float ratio_h = resize_h / h;
  float ratio_w = resize_w / w;

  std::vector<float> shape_vec = {h, w, ratio_h, ratio_w};
  std::vector<int> shape = {1, 4};
  cv::Mat shape_list = cv::Mat(shape, CV_32FC1, shape_vec.data()).clone();
  context->SetTensor(
      io_names_.output_names, {std::move(ori_imgs), std::move(shape_list)});

  return nullptr;
}

}}  // namespace dataelem::alg
