#include "dataelem/alg/ocr_app.h"

#include "dataelem/common/json_utils.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(OcrIntermediate);
REGISTER_ALG_CLASS(OcrPost);
REGISTER_ALG_CLASS(AdjustBboxFromAngle);

void
crop_images(const cv::Mat& src, const cv::Mat& bbs, std::vector<cv::Mat>& rois)
{
  // src: cv::Matcomplete image
  // bbs: cv::Mat bboxes
  // rois : vector to save cropped subimages

  int n = bbs.size[0];
  cv::Mat tmp_bbs(n, 4, CV_32FC2, bbs.data);
  cv::Mat_<cv::Point2f> bbs_(tmp_bbs);

  for (int i = 0; i < n; i++) {
    std::vector<cv::Point2f> v;
    for (int j = 0; j < 4; j++) {
      v.emplace_back(bbs_(i, j));
    }
    // v <cv::Point2f> [pt0, pt1, pt2, pt3]
    float w = round(l2_norm(v[0], v[1]));
    float h = round(l2_norm(v[1], v[2]));

    std::vector<cv::Point2f> src_3points{v[0], v[1], v[2]};
    std::vector<cv::Point2f> dest_3points{{0, 0}, {w, 0}, {w, h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    cv::Mat m;
    cv::warpAffine(src, m, warp_mat, {int(w), int(h)}, cv::INTER_LINEAR);
    rois.emplace_back(std::move(m));

    auto bb = cv::Mat(v);
  }
}

void
transformer_resize_image(
    const cv::Mat& src, int H, int W_min, int W_max, int input_channels, int& W,
    cv::Mat& dst)
{
  cv::Mat img2;
  // very slow here
  if (input_channels == 1) {
    bgr2grayOp(src, img2);
  } else {
    bgr2rgbOp(src, img2);
  }

  cv::Mat img3;
  cv::Rect rect = cv::Rect(0, 0, W, H);
  if (W <= W_max && W >= W_min) {
    resizeOp(img2, img3, W, H);
  } else if (W < W_min) {
    resizeOp(img2, img3, W, H);
    W = W_min;
  } else {
    int h = img2.rows;
    int w = img2.cols;
    int h2 = int(1.0f * W_max / w * h);
    int margin = (H - h2) / 2;
    rect = cv::Rect(0, margin, W_max, h2);
    resizeOp(img2, img3, W_max, h2);
    W = W_max;
  }
  img3.copyTo(dst(rect));
}


TRITONSERVER_Error*
OcrIntermediate::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "ocr_intermediate";

  _long_image_segmentor.reset(new LongImageSegment());

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "fixed_height", &_fixed_height, 32);
  SafeParseParameter(params, "downsample_rate", &_downsample_rate, 8);
  SafeParseParameter(params, "input_channels", &_input_channels, 1);
  SafeParseParameter(params, "W_min", &_W_min, 40);
  SafeParseParameter(params, "W_max", &_W_max, 800);
  SafeParseParameter(params, "version", &_version, 1);
  SafeParseParameter(params, "hw_thrd", &_hw_thrd, 1.5);

  // create buffer for input0
  max_cache_patchs_ = 512;
  int buffer0_size = 512 * _fixed_height * _W_max * _input_channels * 1;
  std::vector<int> input0_shape({512, _fixed_height, _W_max, _input_channels});
  triton::backend::BackendMemory* input0_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU}, 0, buffer0_size,
      &input0_memory));
  input0_buffer_.reset(input0_memory);
  input0_ = cv::Mat(input0_shape, CV_8U, input0_buffer_->MemoryPtr());

  return nullptr;
}

TRITONSERVER_Error*
OcrIntermediate::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  auto& src_ = inputs[0].m();
  cv::Mat src = to_2dmat(src_);
  auto& bboxes = inputs[1].m();
  nlohmann::json params;
  OCTensor* params_tensor;
  if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
    auto data = params_tensor->GetString(0);
    parse_nlo_json(data, params);
  }

  bool support_long_image_segment = false;
  if (_version == 1 && params.contains("support_long_image_segment")) {
    support_long_image_segment =
        params["support_long_image_segment"].get<bool>();
  }

  // refine box(in mrcnn), crop image, split long sentence
  // and resize with fixed height
  IntegerList groups;
  MatList patchs;
  MatList rois;
  std::vector<int> rot_shape = {bboxes.size[0], 1};
  cv::Mat rot = cv::Mat(rot_shape, CV_8U, cv::Scalar(0));
  crop_images(src, bboxes, rois);
  if (_version == 2) {
    for (size_t i = 0; i < rois.size(); i++) {
      int h = rois[i].size[0];
      int w = rois[i].size[1];
      if (1.0 * h / w >= _hw_thrd) {
        cv::rotate(rois[i], rois[i], cv::ROTATE_90_COUNTERCLOCKWISE);
        rot.at<uint8_t>(i, 1) = 1;
      }
    }
  }

  if (_version == 1 && support_long_image_segment) {
    _long_image_segmentor->segment_v2(rois, patchs, groups);
  } else {
    patchs.insert(patchs.end(), rois.begin(), rois.end());
  }

  // convert to image tensor
  int n = patchs.size();
  int max_w = 0;
  std::vector<int> widths_shape = {n, 2};
  cv::Mat widths_(widths_shape, CV_32S);
  auto widths = cv::Mat_<int>(widths_);

  for (int i = 0; i < n; i++) {
    int h = patchs[i].rows;
    int w = patchs[i].cols;
    int new_w = std::max((w * _fixed_height) / h, 1);
    new_w = std::ceil(new_w * 1.0 / _downsample_rate) * _downsample_rate;
    widths(i, 0) = _fixed_height;
    widths(i, 1) = new_w;

    if (new_w > max_w) {
      max_w = new_w;
    }
  }

  auto type = _input_channels == 3 ? CV_8UC3 : CV_8UC1;
  max_w = std::min(max_w, _W_max);
  max_w = std::max(max_w, _W_min);

  std::vector<int> shape = {n, _fixed_height, max_w, _input_channels};
  cv::Mat images;
  if (n <= max_cache_patchs_) {
    images = cv::Mat(shape, CV_8U, input0_.data);
    images.setTo(0);
  } else {
    images = cv::Mat(shape, CV_8U, cv::Scalar_<uint8_t>(0));
  }

  // cv::Mat images(shape, CV_8U, cv::Scalar_<uint8_t>(0));
  for (int i = 0; i < n; i++) {
    if (_version == 2) {
      int w = _W_max;
      widths(i, 1) = w;
      cv::Mat dst(_fixed_height, w, type, images.data + images.step[0] * i);
      resizeOp(patchs[i], dst, w, _fixed_height);
    } else {
      // int j = widths(i, 0);
      int new_w = widths(i, 1);
      cv::Mat dst(_fixed_height, max_w, type, images.data + images.step[0] * i);
      transformer_resize_image(
          patchs[i], _fixed_height, _W_min, _W_max, _input_channels, new_w,
          dst);
      // std::cout << "i,ori_w,new_w:" << i << "," << widths(i, 1) << "," <<
      // new_w
      //          << "\n";
      widths(i, 1) = new_w;
    }
  }

  cv::Mat grp;
  if (groups.size() > 0) {
    grp = cv::Mat(groups).reshape(1, groups.size()).clone();
  } else {
    grp = cv::Mat(0, 1, CV_32S);
  }

  if (_version == 2) {
    context->SetTensor(
        io_names_.output_names,
        {std::move(images), std::move(widths_), std::move(rot)});
  } else {
    context->SetTensor(
        io_names_.output_names,
        {std::move(images), std::move(widths_), std::move(grp)});
  }

  return nullptr;
}

TRITONSERVER_Error*
OcrPost::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "adjust_bbox_from_angle";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "version", &_version, 1);

  return nullptr;
}

TRITONSERVER_Error*
OcrPost::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  // auto& image = inputs[0].m();
  cv::Mat bboxes, bboxes_score, texts_score_, groups_;
  std::vector<absl::string_view> texts;
  if (_version == 1) {
    bboxes = inputs[0].m();
    bboxes_score = inputs[1].m();
    inputs[2].GetStrings(texts);

    texts_score_ = inputs[3].m();
    groups_ = inputs[4].m();
  } else {
    inputs[0].GetStrings(texts);
    texts_score_ = inputs[1].m();
    groups_ = inputs[2].m();
  }


  auto scores = cv::Mat_<float>(texts_score_);
  auto groups = cv::Mat_<int>(groups_);
  int grp_size = groups_.size[0];

  nlohmann::json params;
  OCTensor* params_tensor;
  if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
    auto data = params_tensor->GetString(0);
    parse_nlo_json(data, params);
  }

  bool support_long_image_segment = false;
  if (params.contains("support_long_image_segment")) {
    support_long_image_segment =
        params["support_long_image_segment"].get<bool>();
  }

  bool with_blank = false;
  if (params.contains("split_long_sentence_blank")) {
    with_blank = params["split_long_sentence_blank"].get<bool>();
  }
  std::string SEP = with_blank ? " " : "";

  // segment postprocess, merge small patch result into original
  if (support_long_image_segment && grp_size > 0) {
    StringList out_texts;
    std::vector<float> out_scores;
    std::string curr_val = std::string(texts.at(0));
    auto curr_score = scores(0, 0);
    auto curr_group = groups(0, 0);
    int num_in_grop = 1;
    for (size_t i = 1; i < texts.size(); i++) {
      if (groups(i, 0) == curr_group) {
        absl::StrAppend(&curr_val, SEP, texts.at(i));
        curr_score += scores(i, 0);
      } else {
        out_texts.emplace_back(std::move(curr_val));
        out_scores.emplace_back(curr_score / num_in_grop);
        curr_group = groups(i, 0);
        curr_val = std::string(texts.at(i));
        curr_score = scores(i, 0);
        num_in_grop = 1;
      }
    }
    out_texts.emplace_back(std::move(curr_val));
    out_scores.emplace_back(curr_score / num_in_grop);

    int n = out_texts.size();
    auto mat_scores = cv::Mat(out_scores).reshape(1, n).clone();
    auto t2 = OCTensor(std::move(mat_scores));
    t2.set_shape({n});
    if (_version == 1) {
      context->SetTensor(
          io_names_.output_names, {
                                      std::move(bboxes),
                                      std::move(bboxes_score),
                                      std::move(OCTensor(out_texts, {n})),
                                      std::move(t2),
                                  });
    } else {
      context->SetTensor(
          io_names_.output_names, {
                                      std::move(OCTensor(out_texts, {n})),
                                      std::move(t2),
                                  });
    }
  } else {
    if (_version == 1) {
      context->SetTensor(
          io_names_.output_names, {std::move(inputs[0]), std::move(inputs[1]),
                                   std::move(inputs[2]), std::move(inputs[3])});
    } else {
      context->SetTensor(
          io_names_.output_names, {std::move(inputs[0]), std::move(inputs[1])});
    }
  }


  return nullptr;
}


TRITONSERVER_Error*
AdjustBboxFromAngle::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "adjust_bbox_from_angle";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "thrd", &_thrd, 0.9);

  return nullptr;
}

TRITONSERVER_Error*
AdjustBboxFromAngle::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  auto& bboxes = inputs[0].m();
  auto& angles = inputs[1].m();
  auto& angle_scores = inputs[2].m();
  cv::Mat bboxes_adj = bboxes.clone();
  int n = bboxes.size[0];
  for (int i = 0; i < n; i++) {
    int angle = (int)angles.at<uint8_t>(i, 0);
    float score = angle_scores.at<float>(i, 0);
    if ((angle == 2 && score >= _thrd) || angle == 1 || angle == 3) {
      if (angle == 3 && score < _thrd) {
        angle = 1;
      }
      if (angle == 1) {
        bboxes_adj.at<float>(i, 0, 0) = bboxes.at<float>(i, 1, 0);
        bboxes_adj.at<float>(i, 0, 1) = bboxes.at<float>(i, 1, 1);
        bboxes_adj.at<float>(i, 1, 0) = bboxes.at<float>(i, 2, 0);
        bboxes_adj.at<float>(i, 1, 1) = bboxes.at<float>(i, 2, 1);
        bboxes_adj.at<float>(i, 2, 0) = bboxes.at<float>(i, 3, 0);
        bboxes_adj.at<float>(i, 2, 1) = bboxes.at<float>(i, 3, 1);
        bboxes_adj.at<float>(i, 3, 0) = bboxes.at<float>(i, 0, 0);
        bboxes_adj.at<float>(i, 3, 1) = bboxes.at<float>(i, 0, 1);
      } else if (angle == 2) {
        bboxes_adj.at<float>(i, 0, 0) = bboxes.at<float>(i, 2, 0);
        bboxes_adj.at<float>(i, 0, 1) = bboxes.at<float>(i, 2, 1);
        bboxes_adj.at<float>(i, 1, 0) = bboxes.at<float>(i, 3, 0);
        bboxes_adj.at<float>(i, 1, 1) = bboxes.at<float>(i, 3, 1);
        bboxes_adj.at<float>(i, 2, 0) = bboxes.at<float>(i, 0, 0);
        bboxes_adj.at<float>(i, 2, 1) = bboxes.at<float>(i, 0, 1);
        bboxes_adj.at<float>(i, 3, 0) = bboxes.at<float>(i, 1, 0);
        bboxes_adj.at<float>(i, 3, 1) = bboxes.at<float>(i, 1, 1);
      } else {
        bboxes_adj.at<float>(i, 0, 0) = bboxes.at<float>(i, 3, 0);
        bboxes_adj.at<float>(i, 0, 1) = bboxes.at<float>(i, 3, 1);
        bboxes_adj.at<float>(i, 1, 0) = bboxes.at<float>(i, 0, 0);
        bboxes_adj.at<float>(i, 1, 1) = bboxes.at<float>(i, 0, 1);
        bboxes_adj.at<float>(i, 2, 0) = bboxes.at<float>(i, 1, 0);
        bboxes_adj.at<float>(i, 2, 1) = bboxes.at<float>(i, 1, 1);
        bboxes_adj.at<float>(i, 3, 0) = bboxes.at<float>(i, 2, 0);
        bboxes_adj.at<float>(i, 3, 1) = bboxes.at<float>(i, 2, 1);
      }
    }
  }

  context->SetTensor(io_names_.output_names, {std::move(bboxes_adj)});

  return nullptr;
}

}}  // namespace dataelem::alg
