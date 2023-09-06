#include <array>

#include "absl/strings/escaping.h"
#include "nlohmann/json.hpp"

#include "dataelem/alg/general_prep.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(GeneralPrepOp);

TRITONSERVER_Error*
GeneralPrepOp::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "general_prep_op";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  auto status = model_config.Find("parameters", &params);
  if (!status) {
    return nullptr;
  }

  std::string ops_config;
  SafeParseParameter(params, "ops", &ops_config);

  if (ops_config.length() > 0) {
    auto ops = nlohmann::json::parse(ops_config.data());
    if (ops.contains("b64dec")) {
      use_b64dec_ = ops["b64dec"].get<bool>();
    }
    if (ops.contains("imdec")) {
      use_imdec_ = ops["imdec"].get<bool>();
    }
    if (ops.contains("imdec_param")) {
      imdec_param_ = ops["imdec_param"].get<int>();
    }

    if (ops.contains("resize")) {
      use_resize_ = ops["resize"].get<bool>();
    }
    if (ops.contains("resize_param")) {
      resize_param_ = ops["resize_param"].get<int>();
    }

    if (ops.contains("padding")) {
      use_padding_ = ops["padding"].get<bool>();
    }
    if (ops.contains("padding_param")) {
      padding_param_ = ops["padding_param"].get<int>();
    }

    if (ops.contains("b64out")) {
      use_b64out_ = ops["b64out"].get<bool>();
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
GeneralPrepOp::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  if (use_b64dec_ && !use_imdec_) {
    // b64dec as out
    std::string image_raw_bytes;
    auto* tensor = &inputs[0];
    if (!absl::Base64Unescape(tensor->GetString(0), &image_raw_bytes)) {
      TRITONJSON_STATUSRETURN("base64 decode failed in alg:" + alg_name_);
    }
    int n = image_raw_bytes.length();
    auto bytes_mat =
        cv::Mat(1, n, CV_8U, const_cast<char*>(image_raw_bytes.data())).clone();
    context->SetTensor(io_names_.output_names, {std::move(bytes_mat)});
    return nullptr;
  }

  cv::Mat img;
  if (use_b64dec_ && use_imdec_) {
    std::string image_raw_bytes;
    auto* tensor = &inputs[0];
    if (!absl::Base64Unescape(tensor->GetString(0), &image_raw_bytes)) {
      TRITONJSON_STATUSRETURN("base64 decode failed in alg:" + alg_name_);
    }

    int n = image_raw_bytes.length();
    auto bytes_mat =
        cv::Mat(n, 1, CV_8U, const_cast<char*>(image_raw_bytes.data()));
    try {
      img = cv::imdecode(bytes_mat, imdec_param_);
    }
    catch (cv::Exception& e) {
      TRITONJSON_STATUSRETURN(e.err);
    }
  } else if (use_imdec_) {
    img = cv::imdecode(inputs[0].m(), imdec_param_);
  } else {
    // input is already a image mat
    img = inputs[0].m();
  }

  RETURN_ERROR_IF_TRUE(
      img.rows < 0 || img.cols < 0 || img.rows > 50000 || img.cols > 50000,
      TRITONSERVER_ERROR_INTERNAL,
      (std::string("image shape is abnormal [") + alg_name_ + "]"));

  cv::Mat resize;
  int w = img.cols, h = img.rows;
  int new_w = w, new_h = h;
  if (use_resize_) {
    int max_edge = std::max(w, h);
    float r = 1.0 * resize_param_ / max_edge;
    new_w = max_edge == w ? resize_param_ : int(r * w);
    new_h = max_edge == h ? resize_param_ : int(r * h);
    cv::resize(img, resize, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);
  } else {
    resize = img;
  }

  cv::Mat result;
  if (use_padding_) {
    int pad_r = resize_param_ - new_w;
    int pad_b = resize_param_ - new_h;
    cv::copyMakeBorder(
        resize, result, 0, pad_b, 0, pad_r, cv::BORDER_CONSTANT,
        padding_param_);
  } else {
    result = resize;
  }

  if (use_b64out_) {
    int len = result.cols * result.rows * result.channels() * 1;
    absl::string_view tmp(reinterpret_cast<const char*>(result.data), len);
    std::string b64_out = absl::Base64Escape(tmp);
    auto output0 = OCTensor(b64_out, true);
    output0.set_base64();
    context->SetTensor(io_names_.output_names[0], std::move(output0));
  } else {
    context->SetTensor(io_names_.output_names, {std::move(result)});
  }

  return nullptr;
}

}}  // namespace dataelem::alg
