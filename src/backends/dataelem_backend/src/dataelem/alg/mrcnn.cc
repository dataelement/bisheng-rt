#include "dataelem/alg/mrcnn.h"

#include <array>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(MaskRCNN);


// added by sunjun@dataelem
bool
normalizeimage_by_textdirection(
    std::vector<double>& thetas, int low_index, int up_index, double& theta)
{
  std::vector<double> thetas_sample(
      thetas.begin() + low_index, thetas.begin() + up_index);
  std::vector<double> vector_x, vector_y;
  for (size_t i = 0; i < thetas_sample.size(); i++) {
    auto theta_ = thetas_sample.at(i) * CV_PI / 180.;
    vector_x.push_back(cos(theta_));
    vector_y.push_back(sin(theta_));
  }

  double avg_x =
      (std::accumulate(vector_x.begin(), vector_x.end(), 0.) /
       (up_index - low_index + 1.));

  double avg_y =
      (std::accumulate(vector_y.begin(), vector_y.end(), 0.) /
       (up_index - low_index + 1.));

  if (abs(avg_y) <= 1e-6 && avg_x > 0)
    theta = 0;
  else if (abs(avg_y) <= 1e-6 && avg_x < 0)
    theta = 180;
  else if (abs(avg_x) <= 1e-6 && avg_y > 0)
    theta = 90;
  else if (abs(avg_x) <= 1e-6 && avg_y < 0)
    theta = 270;
  else if (avg_x > 0 && avg_y > 0)
    theta = std::atan(avg_y / avg_x) * 180 / CV_PI;
  else if (avg_x > 0 && avg_y < 0)
    theta = 360 + std::atan(avg_y / avg_x) * 180 / CV_PI;
  else if (avg_x < 0 && avg_y < 0)
    theta = 180 + std::atan(avg_y / avg_x) * 180 / CV_PI;
  else if (avg_x < 0 && avg_y > 0)
    theta = 180 + std::atan(avg_y / avg_x) * 180 / CV_PI;
  return true;
}


TRITONSERVER_Error*
MaskRCNN::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "mrcnn";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

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

  if (_version.compare("trt_v5") == 0) {
    graph_io_names_ =
        StepConfig({"image"}, {"output_masks", "output_detections"});
  } else {
    graph_io_names_ = StepConfig(
        {"image"}, {"output/scores", "output/masks", "output/boxes",
                    "output/boxes_cos", "output/boxes_sin"});
  }

  // create pinned input memory buffer
  size_t total_byte_size = 2560 * 2560 * 3 * 4;
  triton::backend::BackendMemory* input_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
#ifdef TRITON_ENABLE_GPU
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
#else
      {triton::backend::BackendMemory::AllocationType::CPU}, 0,
#endif
      total_byte_size, &input_memory));
  input_buffer_mem_.reset(input_memory);
  input_buffer_ = input_buffer_mem_->MemoryPtr();

  return nullptr;
}


TRITONSERVER_Error*
MaskRCNN::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  cv::Mat src = inputs[0].GetImage();

  // parse params
  rapidjson::Document d;
  OCTensor* params_tensor;
  if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
    auto buffer = params_tensor->GetString(0);
    d.Parse(buffer.data(), buffer.length());
    if (d.HasParseError()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "json parsing error on params");
    }
  }
  APIData params(d);

  bool support_long_rotate_dense = false;
  get_ad_value(params, "support_long_rotate_dense", support_long_rotate_dense);

  MatList outputs;
  if (!support_long_rotate_dense) {
    MatList prep_outs, graph_outs, post_outs;
    RETURN_IF_ERROR(PreprocessStep(params, {src}, prep_outs));
    RETURN_IF_ERROR(GraphStep(context, prep_outs, graph_outs));
    graph_outs.emplace_back(src);
    graph_outs.emplace_back(prep_outs[1]);
    RETURN_IF_ERROR(PostprocessStep(params, graph_outs, outputs));
  } else {
    // Run detection twices
    // pass1
    MatList prep1_outs, graph1_outs, post1_outs;
    RETURN_IF_ERROR(PreprocessStep(params, {src}, prep1_outs));
    RETURN_IF_ERROR(GraphStep(context, prep1_outs, graph1_outs));
    graph1_outs.emplace_back(src);
    graph1_outs.emplace_back(prep1_outs[1]);
    RETURN_IF_ERROR(PostprocessStep1(params, graph1_outs, post1_outs));

    // pass2
    MatList prep2_outs, graph2_outs;
    // no bbox found
    if (post1_outs.size() == 0) {
      outputs.emplace_back(cv::Mat());
      outputs.emplace_back(cv::Mat());
    } else {
      RETURN_IF_ERROR(PreprocessStep(params, post1_outs, prep2_outs));
      RETURN_IF_ERROR(GraphStep(context, prep2_outs, graph2_outs));
      // append: rot_image, rot_scale, src, rot_theta
      graph2_outs.emplace_back(post1_outs[0]);
      graph2_outs.emplace_back(prep2_outs[1]);
      graph2_outs.emplace_back(src);
      graph2_outs.emplace_back(post1_outs[1]);
      RETURN_IF_ERROR(PostprocessStep2(params, graph2_outs, outputs));
    }
  }

  cv::Mat src_scale = (cv::Mat_<int>(1, 3) << src.rows, src.cols, 3);

  if (outputs[0].empty()) {
    context->SetTensor(
        io_names_.output_names, {OCTensor({0, 4, 2}, CV_32F),
                                 OCTensor({0}, CV_32F), std::move(src_scale)});
  } else {
    int n = outputs[1].size[0];
    OCTensorList tensors = {
        std::move(outputs[0]), std::move(outputs[1]), std::move(src_scale)};
    tensors[1].set_shape({n});
    context->SetTensor(io_names_.output_names, std::move(tensors));
  }

  return nullptr;
}

// Preprocess for the mrcnn.
//
// Params:
//  params - parameters
//  inputs - input mats, [image]
//  outputs - output mats, [dst,scale]
// Returns:
TRITONSERVER_Error*
MaskRCNN::PreprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  const cv::Mat& src = inputs[0];
  // print_mat<uint8_t>(src, "prep_in");
  int longer_edge_size = -1;
  if (params.has("longer_edge_size")) {
    get_ad_value(params, "longer_edge_size", longer_edge_size);
    if (longer_edge_size <= 0) {
      longer_edge_size = calc_prop_scale(src, _scale_list);
      // return TRITONSERVER_ErrorNew(
      //     TRITONSERVER_ERROR_INVALID_ARG,
      //     "longer_edge_size must be greater than zero");
    }
  } else {
    longer_edge_size = calc_prop_scale(src, _scale_list);
  }
  
  if(_version.compare("trt_v5") == 0){
    longer_edge_size = 1600;
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

  float resize_scale =
      std::sqrt(1.0 * resize.rows / orig_h * resize.cols / orig_w);
  cv::Mat scale_mat = (cv::Mat_<float>(1, 1) << resize_scale);

  cv::Mat dst;
  if (_padding) {
    cv::Mat padded;
    cv::copyMakeBorder(
        resize, padded, 0, longer_edge_size - resize.rows, 0,
        longer_edge_size - resize.cols, cv::BORDER_CONSTANT, {0, 0, 0});

    std::vector<int> shape = {1, longer_edge_size, longer_edge_size, 3};
    dst = cv::Mat(shape, CV_32F, input_buffer_);
    cv::Mat temp =
        cv::Mat(longer_edge_size, longer_edge_size, CV_32FC3, dst.data);
    padded.convertTo(temp, CV_32F);
  } else {
    dst = cv::Mat(resize.rows, resize.cols, CV_32FC3, input_buffer_);
    resize.convertTo(dst, CV_32F);
  }

  // print_mat<float>(dst, "prep_out");
  outputs.emplace_back(std::move(dst));
  outputs.emplace_back(std::move(scale_mat));

  return nullptr;
}

TRITONSERVER_Error*
MaskRCNN::GraphStep(
    AlgRunContext* context, const MatList& inputs, MatList& outputs)
{
  OCTensorList tensor_inputs = {inputs[0]};
  tensor_inputs[0].set_pinned();
  OCTensorList tensor_outputs;
  RETURN_IF_ERROR(Algorithmer::GraphExecuateStep(
      context, graph_io_names_.input_names, graph_io_names_.output_names,
      tensor_inputs, tensor_outputs));

  // Part postprocess for the tensorrt maskrcnn
  if (_version.compare("trt_v5") == 0) {
    cv::Mat output_0 = tensor_outputs[0].GetMat();
    cv::Mat output_1 = tensor_outputs[1].GetMat();

    int valid_num = 0;
    // 过滤掉padding部分
    for (int i = 0; i < output_1.size[0]; ++i) {
      if (output_1.at<float>(i, 5) > 0.01) {
        valid_num += 1;
      } else {
        break;
      }
    }
    cv::Mat scores = output_1(cv::Rect(5, 0, 1, valid_num));
    cv::Mat bboxes = output_1(cv::Rect(0, 0, 4, valid_num));
    cv::Mat boxes_cos = output_1(cv::Rect(6, 0, 1, valid_num));
    cv::Mat boxes_sin = output_1(cv::Rect(7, 0, 1, valid_num));

    // valid_num, 1, 28, 28 to valid_num, 28, 28
    vector<int> new_size = {valid_num, output_0.size[2], output_0.size[3]};
    cv::Mat masks(new_size, CV_32F, output_0.data);

    for (auto& m : {scores, masks, bboxes, boxes_cos, boxes_sin}) {
      outputs.emplace_back(std::move(m));
    }
  } else {
    for (auto& tensor : tensor_outputs) {
      outputs.emplace_back(std::move(tensor.GetMat()));
    }
  }

  return nullptr;
}

// PostprocessStep for the mrcnn.
//
// Params:
//  params - Parameters
//  inputs - input mats, [score,mask,bbox,bbox_cos,bbox_sin]
//  outputs - output mats, [bboxes, bboxes_score]
// Returns:

// update 2020.04.11
// todo (add by gulixin), 20190801
// 1. before merge_quadrangle_standard, multiply box by 10000 (done 20190801)
// 2. process long size text, calc twice, 1) calc angle, 2) rotate+det again
// 3. remove small bb contained in other bb
//    #comment by hf: this logic will do outside the inference server
// 4. make dst_bbs value in ordered points (done 20190801)
// 5. support the text direction calculation (done, 2020.04.11)
TRITONSERVER_Error*
MaskRCNN::PostprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  auto size = inputs.size();
  const cv::Mat& scores = inputs[0];
  const cv::Mat& masks = inputs[1];
  const cv::Mat& bboxes = inputs[2];
  const cv::Mat& orig = inputs[size - 2];
  const cv::Mat& scale = inputs[size - 1];

  cv::Mat boxes_cos, boxes_sin;
  // version v5
  if (size == 7) {
    boxes_cos = inputs[3];
    boxes_sin = inputs[4];
  }

  // print_mat<float>(scores, "post_in_scores");
  // print_mat<float>(masks, "post_in_masks");
  // print_mat<float>(bboxes, "post_in_bboxes");
  // print_mat<float>(boxes_cos, "post_in_boxes_cos");
  // print_mat<float>(boxes_sin, "post_in_boxes_sin");

  if (scores.rows == 0) {
    outputs.emplace_back(cv::Mat());
    outputs.emplace_back(cv::Mat());
    return nullptr;
  }

  bool enable_huarong_box_adjust = false;
  get_ad_value(params, "enable_huarong_box_adjust", enable_huarong_box_adjust);

  bool unify_text_direction = false;
  get_ad_value(params, "unify_text_direction", unify_text_direction);

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

  // print_mat<float>(output0, "post_out");
  outputs.emplace_back(std::move(output0));
  outputs.emplace_back(std::move(output1));
  return nullptr;
}

// PostprocessStep1 for the mrcnn, normalize the image
//
// Params:
//  Params - Parameters
//  inputs - input mats, [score,mask,bbox,bbox_cos,bbox_sin,orig,scale]
//  outputs: output mats, [rotated_image, rotated_theta]
// Returns
// update 2022.08.24
// Calc the average angle of text block
TRITONSERVER_Error*
MaskRCNN::PostprocessStep1(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  auto size = inputs.size();
  const cv::Mat& scores = inputs[0];
  const cv::Mat& masks = inputs[1];
  const cv::Mat& bboxes = inputs[2];
  const cv::Mat& orig = inputs[size - 2];
  const cv::Mat& scale = inputs[size - 1];

  if (scores.rows == 0) {
    return nullptr;
  }

  bool enable_huarong_box_adjust = false;
  get_ad_value(params, "enable_huarong_box_adjust", enable_huarong_box_adjust);
  bool unify_text_direction = false;

  bool normalize_image_orientation = false;
  get_ad_value(
      params, "normalize_image_orientation", normalize_image_orientation);

  std::vector<std::vector<cv::Point2f>> points_vec;
  std::vector<float> socres_vec;
  cv::Mat boxes_cos, boxes_sin;
  mask_to_bb(
      scores, masks, bboxes, boxes_cos, boxes_sin, orig, scale,
      enable_huarong_box_adjust, unify_text_direction, points_vec, socres_vec);

  if (points_vec.size() == 0) {
    return nullptr;
  }

  // computeAngle: calculate the tilt angle according to bbox.
  // if normalize_image_orientation: true
  //     calculate the angle based on the first point and the second point of
  //     the box and convert it into an interval from 0 to 360.
  // if normalize_image_orientation: false
  //     calculate the angle based on the lowest point of the box and the
  //     point to the right of this point,
  //     the final return value is between -45 and 45.

  double theta_ = 0, theta;
  if (!normalize_image_orientation) {
    std::vector<double> thetas;
    for (const auto& v : points_vec) {
      double angle = computeAngle(v);
      thetas.push_back(angle * 180. / CV_PI);
    }

    // Sort theta values, remove 10% from the beginning and the end,
    // and calculate the remaining mean
    std::sort(
        thetas.begin(), thetas.end(),
        [](const double& v1, const double& v2) { return v1 < v2; });
    int cnt = thetas.size();
    int low_index = std::max(0, int(std::floor(cnt * 0.1) - 1));
    int up_index = std::min(cnt - 1, int(std::ceil(cnt * 0.9) - 1));

    int i = low_index;
    while (i <= up_index) {
      theta_ += thetas[i++];
    }
    theta = theta_ / (up_index - low_index + 1);
  } else {
    // added by sunjun
    std::vector<double> thetas;
    for (const auto& v : points_vec) {
      double angle = computeAngle(v, true);
      thetas.push_back(angle * 180. / CV_PI);
    }
    std::sort(
        thetas.begin(), thetas.end(),
        [](const double& v1, const double& v2) { return v1 < v2; });
    int cnt = thetas.size();
    int low_index = std::max(0, int(std::floor(cnt * 0.1) - 1));
    int up_index = std::min(cnt - 1, int(std::ceil(cnt * 0.9) - 1));

    normalizeimage_by_textdirection(thetas, low_index, up_index, theta);

    if (theta >= 0 && theta < 45) {
      theta = 0;
    } else if (theta >= 45 && theta < 90) {
      theta = 90;
    } else if (theta >= 90 && theta < 135) {
      theta = 90;
    } else if (theta >= 135 && theta < 180) {
      theta = 180;
    } else if (theta >= 180 && theta < 225) {
      theta = 180;
    } else if (theta >= 225 && theta < 270) {
      theta = 270;
    } else if (theta >= 270 && theta < 315) {
      theta = 270;
    } else {
      theta = 0;
    }
  }

  cv::Mat dst;
  rotateOp2(orig, dst, -theta);
  cv::Mat theta_mat = cv::Mat_<double>(1, 1) << theta;
  outputs.emplace_back(std::move(dst));
  outputs.emplace_back(std::move(theta_mat));
  return nullptr;
}

// PostprocessStep2 for the mrcnn
//
// Params:
//  params - parameters
//  inputs - input mats
//           [score,mask,bbox,bbox_cos,bbox_sin,
//            rot_dst,rot_scale,rot_image,rot_theta]
//  outpus - output mats, [bboxes, bboxes_score]
// Returns

TRITONSERVER_Error*
MaskRCNN::PostprocessStep2(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  auto size = inputs.size();
  const cv::Mat& scores = inputs[0];
  const cv::Mat& masks = inputs[1];
  const cv::Mat& bboxes = inputs[2];

  cv::Mat boxes_cos, boxes_sin;
  if (size == 9) {
    boxes_cos = inputs[3];
    boxes_sin = inputs[4];
  }

  const cv::Mat& rotate = inputs[size - 4];
  const cv::Mat& scale = inputs[size - 3];
  const cv::Mat& orig = inputs[size - 2];
  const cv::Mat& theta_mat = inputs[size - 1];

  double theta = theta_mat.at<double>(0, 0);
  if (scores.rows == 0) {
    outputs.emplace_back(cv::Mat());
    outputs.emplace_back(cv::Mat());
    return nullptr;
  }

  bool enable_huarong_box_adjust = false;
  get_ad_value(params, "enable_huarong_box_adjust", enable_huarong_box_adjust);

  bool unify_text_direction = false;
  get_ad_value(params, "unify_text_direction", unify_text_direction);

  std::vector<std::vector<cv::Point2f>> points_vec;
  std::vector<float> scores_vec;
  mask_to_bb(
      scores, masks, bboxes, boxes_cos, boxes_sin, rotate, scale,
      enable_huarong_box_adjust, unify_text_direction, points_vec, scores_vec);

  // remap the bb from rotate into orig image
  cv::Point2f old_center(orig.cols / 2.0, orig.rows / 2.0);
  cv::Point2f new_center(rotate.cols / 2.0, rotate.rows / 2.0);
  readjust_bb(old_center, new_center, theta, points_vec);

  std::vector<cv::Point2f> points;
  for (auto& ps : points_vec) {
    points.insert(points.end(), ps.begin(), ps.end());
  }
  int n = points_vec.size();
  cv::Mat output0 = vec2mat(points, 2, n);
  cv::Mat output1 = cv::Mat(scores_vec).clone();
  outputs.emplace_back(std::move(output0));
  outputs.emplace_back(std::move(output1));
  return nullptr;
}

void
MaskRCNN::mask_to_bb(
    const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
    const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& orig,
    const cv::Mat& scale, const bool& enable_huarong_box_adjust,
    const bool& unify_text_direction, std::vector<Point2fList>& points_vec,
    std::vector<float>& scores_vec)
{
  // calculate rrect for each text mask area
  bool has_angle = (!boxes_cos.empty() && !boxes_sin.empty()) ? true : false;
  int orig_h = orig.rows, orig_w = orig.cols;

  cv::Mat bbs = bboxes / scale.at<float>(0, 0);
  clip_boxes(bbs, orig_h, orig_w);
  int mask_step0 = masks.size[1] * masks.size[2];
  int bbs_cnt = bbs.size[0];

  // std::vector<BoolFuture> rets(bbs_cnt);
  std::vector<Point2fList> points_list(bbs_cnt);
  std::vector<std::array<float, 2>> attrs_list(bbs_cnt);
  std::vector<float> score_list(bbs_cnt, -1.0);

  for (int i = 0; i < bbs_cnt; ++i) {
    // rets[i] = _pool->enqueue(
    [this, i, &mask_step0, &orig_w, &orig_h, &has_angle, &masks, &bbs, &scores,
     &boxes_cos, &boxes_sin, &points_list, &score_list, &attrs_list]() -> bool {
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
    }();
    // });
  }
  // GetAsyncRets(rets);

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

  // added by sunjun, at 2022
  if (has_angle && unify_text_direction) {
    refine_box_orientation(points_vec, true);
  }
}

}}  // namespace dataelem::alg