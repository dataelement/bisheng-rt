#include "dataelem/alg/mrcnn_v5.h"

#include <array>

#include "absl/strings/escaping.h"
#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(MaskRCNNV5Prep);
REGISTER_ALG_CLASS(MaskRCNNV5Post);

TRITONSERVER_Error*
MaskRCNNV5Base::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "mrcnn_v5";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "nms_threshold", &_nms_threshold, 0.2f);
  SafeParseParameter(params, "version", &_version, "v5");

  std::string scale_list_str;
  SafeParseParameter(params, "scale_list", &scale_list_str, "");
  if (!scale_list_str.empty()) {
    ParseArrayFromString(scale_list_str, _scale_list);
  } else {
    _scale_list.assign({200, 400, 600, 800, 1000, 1200, 1400, 1600});
  }

  _use_text_direction = false;
  if (_version.compare("v5") == 0 || _version.compare("trt_v5") == 0) {
    _use_text_direction = true;
  }

  _padding = false;
  if (_version.compare("trt_v5") == 0) {
    _padding = true;
  }

  return nullptr;
}

TRITONSERVER_Error*
MaskRCNNV5Prep::init(triton::backend::BackendModel* model_state)
{
  MaskRCNNV5Base::init(model_state);

  // create input memory buffer
  size_t total_byte_size = 2560 * 2560 * 3 * 4;
  triton::backend::BackendMemory* input_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
      total_byte_size, &input_memory));

  input_buffer_.reset(input_memory);

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "base64dec", &_base64dec, true);

  return nullptr;
}

TRITONSERVER_Error*
MaskRCNNV5Prep::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  // parse input0
  cv::Mat src;
  if (_base64dec) {
    std::string image_raw_bytes;
    auto* tensor1 = &inputs[0];
    if (!absl::Base64Unescape(tensor1->GetString(0), &image_raw_bytes)) {
      TRITONJSON_STATUSRETURN("base64 decode failed in alg:" + alg_name_);
    }
    int n = image_raw_bytes.length();
    auto bytes_mat =
        cv::Mat(n, 1, CV_8U, const_cast<char*>(image_raw_bytes.data()));
    try {
      src = cv::imdecode(bytes_mat, 1);
    }
    catch (cv::Exception& e) {
      TRITONJSON_STATUSRETURN(e.err);
    }
  } else {
    auto img = inputs[0].m();
    src = cv::Mat(img.size[0], img.size[1], CV_8UC3, img.data);
  }

  // parse params
  nlohmann::json params;
  try {
    OCTensor* params_tensor;
    if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
      auto content1 = params_tensor->GetString(0);
      params = nlohmann::json::parse(
          content1.data(), content1.data() + content1.length());
    }
  }
  catch (nlohmann::json::parse_error& e) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }

  char* output_buffer = input_buffer_->MemoryPtr();
  int longer_edge_size;
  if (params.contains("longer_edge_size")) {
    longer_edge_size = params["longer_edge_size"].get<int>();
    if (longer_edge_size <= 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "longer_edge_size must be greater than zero");
    }
  } else {
    longer_edge_size = calc_prop_scale(src, _scale_list);
  }

  // Run preprocess for the input
  // longer_edge_size must be less than convert_dst size
  longer_edge_size = std::min(longer_edge_size, 2560);
  int orig_h = src.rows, orig_w = src.cols;
  float scale = longer_edge_size * 1.0 / std::max(orig_h, orig_w);
  int new_h, new_w;
  if (orig_h > orig_w) {
    new_h = longer_edge_size;
    new_w = std::round(scale * orig_w);
  } else {
    new_h = std::round(scale * orig_h);
    new_w = longer_edge_size;
  }

  cv::Mat resize;
  resizeOp(src, resize, new_w, new_h);

  // float resize_scale =
  //     std::sqrt(1.0 * resize.rows / orig_h * resize.cols / orig_w);
  // cv::Mat scale_mat = (cv::Mat_<float>(1, 1) << resize_scale);

  cv::Mat dst;
  if (_padding) {
    cv::Mat padded;
    cv::copyMakeBorder(
        resize, padded, 0, longer_edge_size - resize.rows, 0,
        longer_edge_size - resize.cols, cv::BORDER_CONSTANT, {0, 0, 0});
    std::vector<int> shape = {1, longer_edge_size, longer_edge_size, 3};
    dst = cv::Mat(shape, CV_32F, output_buffer);
    padded.convertTo(dst, CV_32F);
    dst = dst.reshape(1, shape);
  } else {
    dst = cv::Mat(resize.rows, resize.cols, CV_32FC3, output_buffer);
    resize.convertTo(dst, CV_32F);
  }

  OCTensor t1(std::move(dst));
  t1.set_pinned();

  // OCTensor t2(std::move(scale_mat));
  // t2.set_shape({1});

  cv::Mat src_scale = (cv::Mat_<int>(1, 3) << orig_h, orig_w, 3);
  cv::Mat resize_scale = (cv::Mat_<int>(1, 3) << new_h, new_w, 3);

  if (_base64dec) {
    context->SetTensor(
        io_names_.output_names,
        {std::move(t1), std::move(src), std::move(resize_scale),
         std::move(src_scale)});
  } else {
    context->SetTensor(
        io_names_.output_names,
        {std::move(t1), std::move(resize_scale), std::move(src_scale)});
  }
  return nullptr;
}

TRITONSERVER_Error*
MaskRCNNV5Post::init(triton::backend::BackendModel* model_state)
{
  MaskRCNNV5Base::init(model_state);
  const int N_POST_THREADS = 10;
  _pool.reset(new ThreadPool(N_POST_THREADS));
  return nullptr;
}

TRITONSERVER_Error*
MaskRCNNV5Post::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  cv::Mat scores, masks, bboxes, boxes_cos, boxes_sin, orig, scale;

  if (_version.compare("trt_v5") == 0) {
    // graph output for trt: masks, detections
    const cv::Mat& pre_masks = inputs[0].m();
    const cv::Mat& detections = inputs[1].m();
    int valid_num = 0;
    // 过滤掉padding部分
    for (int i = 0; i < detections.size[0]; ++i) {
      if (detections.at<float>(i, 5) > 0.01) {
        valid_num += 1;
      } else {
        break;
      }
    }

    scores = detections(cv::Rect(5, 0, 1, valid_num));
    bboxes = detections(cv::Rect(0, 0, 4, valid_num));
    boxes_cos = detections(cv::Rect(6, 0, 1, valid_num));
    boxes_sin = detections(cv::Rect(7, 0, 1, valid_num));
    // valid_num, 1, 28, 28 to valid_num, 28, 28
    std::vector<int> new_size = {
        valid_num, pre_masks.size[2], pre_masks.size[3]};
    masks = cv::Mat(new_size, CV_32F, pre_masks.data);
    orig = inputs[2].GetMat();
    scale = inputs[3].GetMat();
  } else {
    auto size = inputs.size();
    scores = inputs[0].GetMat();
    masks = inputs[1].GetMat();
    bboxes = inputs[2].GetMat();
    orig = inputs[size - 2].GetMat();
    scale = inputs[size - 1].GetMat();
    // version v5
    if (size == 7) {
      boxes_cos = inputs[3].GetMat();
      boxes_sin = inputs[4].GetMat();
    }
  }

  if (scores.rows == 0) {
    context->SetTensor(
        io_names_.output_names,
        {OCTensor({0, 4, 2}, CV_32F), OCTensor({0}, CV_32F)});
    return nullptr;
  }

  // parse params
  nlohmann::json params;
  try {
    OCTensor* params_tensor;
    if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
      auto params_str = params_tensor->GetString(0);
      params = nlohmann::json::parse(
          params_str.data(), params_str.data() + params_str.length());
    }
  }
  catch (nlohmann::json::parse_error& e) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }

  bool enable_huarong_box_adjust = false;
  if (params.contains("enable_huarong_box_adjust")) {
    enable_huarong_box_adjust = params["enable_huarong_box_adjust"].get<bool>();
  }

  bool unify_text_direction = false;
  if (params.contains("unify_text_direction")) {
    unify_text_direction = params["unify_text_direction"].get<bool>();
  }

  std::vector<std::vector<cv::Point2f>> points_vec;
  std::vector<float> scores_vec;
  mask_to_bb(
      scores, masks, bboxes, boxes_cos, boxes_sin, orig, scale,
      enable_huarong_box_adjust, unify_text_direction, points_vec, scores_vec);

  std::vector<cv::Point2f> points;
  for (auto& ps : points_vec) {
    points.insert(points.end(), ps.begin(), ps.end());
  }
  int n = points_vec.size();
  cv::Mat output0 = vec2mat(points, 2, n);
  cv::Mat output1 = cv::Mat(scores_vec).clone();
  auto t1 = OCTensor(std::move(output1));
  t1.set_shape({n});

  context->SetTensor(
      io_names_.output_names, {std::move(output0), std::move(t1)});

  return nullptr;
}

void
MaskRCNNV5Post::mask_to_bb(
    const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
    const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& orig,
    const cv::Mat& scale, const bool& enable_huarong_box_adjust,
    const bool& unify_text_direction, std::vector<Point2fList>& points_vec,
    std::vector<float>& scores_vec)
{
  // calculate rrect for each text mask area
  bool has_angle = (!boxes_cos.empty() && !boxes_sin.empty()) ? true : false;
  // updated: orig is replace with orig_shape, 2022.10.20
  //  scale is replace with scale_shape

  // int orig_h = orig.size[0], orig_w = orig.size[1];
  int orig_h = orig.at<int>(0, 0);
  int orig_w = orig.at<int>(0, 1);

  int resize_h = scale.at<int>(0, 0);
  int resize_w = scale.at<int>(0, 1);

  float resize_scale = std::sqrt(1.0 * resize_h / orig_h * resize_w / orig_w);

  cv::Mat bbs = bboxes / resize_scale;
  // cv::Mat bbs = bboxes / scale.at<float>(0, 0);

  clip_boxes(bbs, orig_h, orig_w);
  int mask_step0 = masks.size[1] * masks.size[2];
  int bbs_cnt = bbs.size[0];

  std::vector<BoolFuture> rets(bbs_cnt);
  std::vector<Point2fList> points_list(bbs_cnt);
  std::vector<std::array<float, 2>> attrs_list(bbs_cnt);
  std::vector<float> score_list(bbs_cnt, -1.0);

  for (int i = 0; i < bbs_cnt; ++i) {
    rets[i] = _pool->enqueue(
        [this, i, &mask_step0, &orig_w, &orig_h, &has_angle, &masks, &bbs,
         &scores, &boxes_cos, &boxes_sin, &points_list, &score_list,
         &attrs_list]() -> bool {
          cv::Mat full_mask = cv::Mat::zeros(orig_h, orig_w, CV_8U);
          int x0 = int(bbs.at<float>(i, 0) + 0.5);
          int y0 = int(bbs.at<float>(i, 1) + 0.5);
          int x1 = int(bbs.at<float>(i, 2) - 0.5);
          int y1 = int(bbs.at<float>(i, 3) - 0.5);

          x1 = std::max(x0, x1);
          y1 = std::max(y0, y1);
          int w = x1 + 1 - x0, h = y1 + 1 - y0;
          // take care of bounding case
          if (x0 >= orig_w || y0 >= orig_h) {
            return false;
          }

          cv::Mat mask1, mask2;
          cv::Mat mask(
              masks.size[1], masks.size[2], CV_32F,
              masks.data + mask_step0 * 4 * i);
          resizeOp(mask, mask1, w, h);
          cv::Mat(mask1 > 0.5).convertTo(mask2, CV_8U);
          mask2.copyTo(full_mask(cv::Rect(x0, y0, w, h)));
          Contours contours;
          findContoursOp(full_mask, contours);
          if (contours.size() == 0) {
            return false;
          }
          int max_area = 0, max_index = 0;
          for (unsigned int idx = 0; idx < contours.size(); idx++) {
            float area = cv::contourArea(contours[idx]);
            if (area > max_area) {
              max_area = area;
              max_index = idx;
            }
          }

          auto rrect = cv::minAreaRect(contours[max_index]);
          int r_w = rrect.size.width, r_h = rrect.size.height;
          auto rect_area = r_w * r_h;
          if (std::min(r_w, r_h) <= 0 || rect_area <= 0) {
            return false;
          }

          cv::Point2f pts[4];
          rrect.points(pts);
          std::vector<cv::Point2f> pts2;
          // auto get_valid = [](float x, float max) {
          //   return (x < 0) ? 0 : ((x >= max) ? max - 1 : x);
          // };
          for (int j = 0; j < 4; j++) {
            // the int transform is due to the lixin's python logic
            // pts2.emplace_back(get_valid((int)pts[j].x, orig_w),
            // get_valid((int)pts[j].y, orig_h)); delete int and get_valid
            pts2.emplace_back(pts[j].x, pts[j].y);
          }
          float cos = 0, sin = 0;
          if (has_angle) {
            cos = boxes_cos.at<float>(i);
            sin = boxes_sin.at<float>(i);
          }

          std::array<float, 2> attrs = {cos, sin};
          points_list[i] = std::move(pts2);
          score_list[i] = scores.at<float>(i, 0);
          attrs_list[i] = std::move(attrs);
          return true;
          // }();
        });
  }
  GetAsyncRets(rets);

  std::vector<Point2fList> points_list2;
  std::vector<std::array<float, 2>> attrs_list2;
  std::vector<float> score_list2;
  for (int i = 0; i < bbs_cnt; i++) {
    if (points_list[i].size() > 0) {
      points_list2.emplace_back(std::move(points_list[i]));
      attrs_list2.emplace_back(std::move(attrs_list[i]));
      score_list2.emplace_back(score_list[i]);
    }
  }
  if (points_list2.size() == 0) {
    return;
  }

  auto keep = lanms::merge_quadrangle_standard_parallel(
      points_list2, score_list2, score_list2.size(), _nms_threshold);

  // reorder point by text direction, first point is the left-top of text line
  for (const auto& j : keep) {
    auto& points = points_list2[j];
    float score = score_list2[j];
    if (has_angle) {
      float cos = attrs_list2[j].at(0), sin = attrs_list2[j].at(1);
      reorder_start_point(points, cos, sin);
    } else {
      auto idxs = reorder_quadrangle_points(points);
      Point2fList points_tmp;
      for (size_t i = 0; i < idxs.size(); i++) {
        points_tmp.emplace_back(points[idxs[i]]);
      }
      points = points_tmp;
    }
    points_vec.emplace_back(std::move(points));
    scores_vec.emplace_back(score);
  }

  // added by hanfeng at 2020.05.20, do box adjust logic from huarong
  //  https://gitlab.4pd.io/cvxy4pd/cvbe/nn-predictor/issues/56
  if (has_angle && enable_huarong_box_adjust) {
    refine_box_orientation(points_vec);
  }
  if (has_angle && unify_text_direction) {
    refine_box_orientation(points_vec, true);
  }
}

}}  // namespace dataelem::alg
