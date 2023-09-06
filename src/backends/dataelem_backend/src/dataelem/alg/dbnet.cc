#include <array>

#include "nlohmann/json.hpp"

#include "dataelem/alg/dbnet.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

// REGISTER_ALG_CLASS(DBNetPrep);
REGISTER_ALG_CLASS(DBNetPost);

TRITONSERVER_Error*
DBNet::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "dbnet";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

  SafeParseParameter(params, "max_side_len", &max_side_len_);
  SafeParseParameter(params, "det_db_thresh", &det_db_thresh_);
  SafeParseParameter(params, "det_db_box_thresh", &det_db_box_thresh_);
  SafeParseParameter(params, "det_db_unclip_ratio", &det_db_unclip_ratio_);
  SafeParseParameter(params, "det_db_score_mode", &det_db_score_mode_);
  SafeParseParameter(params, "use_dilation", &use_dilation_);
  SafeParseParameter(params, "det_db_score_mode", &det_db_score_mode_);
  SafeParseParameter(params, "use_tensorrt", &use_tensorrt_);

  return nullptr;
}

// TRITONSERVER_Error*
// DBNetPrep::Execute(AlgRunContext* context)
// {
//   const cv::Mat& img = inputs[0].m();
//   float ratio_h{};
//   float ratio_w{};

//   cv::Mat resize_img;
//   resize_op_.Run(
//       img, resize_img, max_side_len_, ratio_h, ratio_w, use_tensorrt_);
//   normalize_op_.Run(&resize_img, mean_, scale_, is_scale_);

//   // std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols,
//   0.0f); std::vector<int> dst_shape = {1, 3, resize_img.rows,
//   resize_img.cols}; cv::Mat dst = cv::Mat(dst_shape, CV_32FC1);
//   permute_op_.Run(&resize_img, reinterpret_cast<float*>(dst.data));
//   cv::Mat ratio_hw = (cv::Mat_<float>(1, 2) << ratio_h, ratio_w);

//   outputs.emplace_back(std::move(dst));
//   outputs.emplace_back(std::move(ratio_hw));
//   return nullptr;
// }

TRITONSERVER_Error*
DBNetPost::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  const cv::Mat& out_feamap = inputs[0].m();
  const cv::Mat& shape_list = inputs[1].m();  // (1, 1, 4)

  int ori_h = (int)shape_list.at<float>(0, 0, 0);
  int ori_w = (int)shape_list.at<float>(0, 0, 1);
  float ratio_h = shape_list.at<float>(0, 0, 2);
  float ratio_w = shape_list.at<float>(0, 0, 3);

  auto output_shape = inputs[0].shape();
  int n2 = (int)output_shape[2];
  int n3 = (int)output_shape[3];
  int n = n2 * n3;

  float* out_data = reinterpret_cast<float*>(out_feamap.data);

  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char*)cbuf.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float*)pred.data());

  const double threshold = det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (use_dilation_) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  std::vector<std::vector<std::vector<int>>> boxes;

  // boxes: (n, 4, 2)
  std::vector<float> scores;
  boxes = post_processor_.BoxesFromBitmap(
      pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
      det_db_score_mode_, scores, ori_w, ori_h);

  std::vector<float> filter_scores;
  boxes = post_processor_.FilterTagDetRes(
      boxes, ratio_h, ratio_w, ori_h, ori_w, scores, filter_scores);

  // int bb_cnt = boxes.size();
  // std::vector<int> shape = {bb_cnt, 4, 2};
  // cv::Mat output0 = cv::Mat(shape, CV_32S, boxes.data()).clone();
  // cv::Mat output1 = cv::Mat(filter_scores).clone();
  // context->SetTensor(
  //     post_step_config_.output_names, {std::move(output0),
  //     std::move(output1)});

  nlohmann::json tmp_json(boxes);
  std::string tmp_str = tmp_json.dump();
  auto output0 = OCTensor(tmp_str, true);
  context->SetTensor(io_names_.output_names[0], std::move(output0));

  return nullptr;
}

}}  // namespace dataelem::alg
