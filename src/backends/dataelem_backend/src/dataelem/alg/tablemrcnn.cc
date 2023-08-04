#include "dataelem/alg/tablemrcnn.h"

#include <array>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(TableMRCNN);


TRITONSERVER_Error*
TableMRCNN::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "tablemrcnn";

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
                    "output/boxes_cos", "output/boxes_sin", "output/labels"});
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
TableMRCNN::Execute(AlgRunContext* context)
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
  }
  cv::Mat src_scale = (cv::Mat_<int>(1, 3) << src.rows, src.cols, 3);

  // test
  // cv::Mat boxes = outputs[0];
  // print_mat<float>(boxes, "boxes");
  // cv::Mat labels = outputs[2];
  // print_mat<int>(labels, "labels");


  if (outputs[0].empty()) {
    context->SetTensor(
        io_names_.output_names,
        {OCTensor({0, 4, 2}, CV_32F), OCTensor({0}, CV_32F),
         OCTensor({0}, CV_32S), std::move(src_scale)});
  } else {
    int n = outputs[1].size[0];
    OCTensorList tensors = {
        std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2]),
        std::move(src_scale)};
    tensors[1].set_shape({n});
    tensors[2].set_shape({n});
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
TableMRCNN::PreprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  const cv::Mat& src = inputs[0];
  // print_mat<uint8_t>(src, "prep_in");
  int longer_edge_size = -1;
  if (params.has("longer_edge_size")) {
    get_ad_value(params, "longer_edge_size", longer_edge_size);
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
  longer_edge_size = std::min(longer_edge_size, 1600);
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
    padded.convertTo(dst, CV_32F);
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
TableMRCNN::GraphStep(
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

    std::cout << "outputs size:" << outputs.size() << std::endl;
    const cv::Mat& labels = outputs[5];
    print_mat<int>(labels, "graph labels");
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
TableMRCNN::PostprocessStep(
    const APIData& params, const MatList& inputs, MatList& outputs)
{
  auto size = inputs.size();
  const cv::Mat& scores = inputs[0];
  const cv::Mat& masks = inputs[1];
  const cv::Mat& bboxes = inputs[2];
  const cv::Mat& boxes_cos = inputs[3];
  const cv::Mat& boxes_sin = inputs[4];
  const cv::Mat& labels = inputs[5];
  const cv::Mat& orig = inputs[size - 2];
  const cv::Mat& scale = inputs[size - 1];

  // test
  // print_mat<int>(labels, "postStep labels");

  if (scores.rows == 0) {
    outputs.emplace_back(cv::Mat());
    outputs.emplace_back(cv::Mat());
    outputs.emplace_back(cv::Mat());
    return nullptr;
  }

  // bool enable_huarong_box_adjust = false;
  // get_ad_value(params, "enable_huarong_box_adjust",
  // enable_huarong_box_adjust);

  // bool unify_text_direction = false;
  // get_ad_value(params, "unify_text_direction", unify_text_direction);

  std::vector<std::vector<cv::Point2f>> points_vec;
  std::vector<float> scores_vec;
  std::vector<int> labels_vec;
  mask_to_bb(
      scores, masks, bboxes, boxes_cos, boxes_sin, labels, orig, scale,
      points_vec, scores_vec, labels_vec);

  std::vector<cv::Point2f> points;
  for (auto& ps : points_vec) {
    points.insert(points.end(), ps.begin(), ps.end());
  }
  int n = points_vec.size();
  cv::Mat output0 = vec2mat(points, 2, n);
  cv::Mat output1 = cv::Mat(scores_vec).clone();
  cv::Mat output2 = cv::Mat(labels_vec).clone();

  // print_mat<float>(output0, "post_out");
  outputs.emplace_back(std::move(output0));
  outputs.emplace_back(std::move(output1));
  outputs.emplace_back(std::move(output2));
  return nullptr;
}

void
TableMRCNN::mask_to_bb(
    const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
    const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& labels,
    const cv::Mat& orig, const cv::Mat& scale,
    // const bool& enable_huarong_box_adjust,
    // const bool& unify_text_direction,
    std::vector<Point2fList>& points_vec, std::vector<float>& scores_vec,
    std::vector<int>& labels_vec)
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
  std::vector<int> label_list(bbs_cnt, 0);

  for (int i = 0; i < bbs_cnt; ++i) {
    // rets[i] = _pool->enqueue(
    [this, i, &mask_step0, &orig_w, &orig_h, &has_angle, &masks, &bbs, &scores,
     &labels, &boxes_cos, &boxes_sin, &points_list, &score_list, &attrs_list,
     &label_list]() -> bool {
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
      label_list[i] = labels.at<int>(i, 0);
      return true;
    }();
    // });
  }
  // GetAsyncRets(rets);

  std::vector<Point2fList> points_list2;
  std::vector<std::array<float, 2>> attrs_list2;
  std::vector<float> score_list2;
  std::vector<int> label_list2;
  for (int i = 0; i < bbs_cnt; i++) {
    if (points_list[i].size() > 0) {
      points_list2.emplace_back(std::move(points_list[i]));
      attrs_list2.emplace_back(std::move(attrs_list[i]));
      score_list2.emplace_back(score_list[i]);
      label_list2.emplace_back(label_list[i]);
    }
    OCTensorList inputs;
  }
  if (points_list2.size() == 0) {
    return;
  }

  // tEnd = std::chrono::high_resolution_clock::now();
  // time_cost = std::chrono::duration<float, std::milli>(tEnd -
  // tStart).count(); std::cout << "mask_to_bbox_time:" << time_cost <<
  // std::endl;

  // tStart = std::chrono::high_resolution_clock::now();
  // reorder point by text direction, first point is the left-top of text line
  for (uint j = 0; j < points_list2.size(); j++) {
    auto& points = points_list2[j];
    float score = score_list2[j];
    int label = label_list2[j];

    points_vec.emplace_back(std::move(points));
    scores_vec.emplace_back(score);
    labels_vec.emplace_back(label);
  }
  // tEnd = std::chrono::high_resolution_clock::now();
  // time_cost = std::chrono::duration<float, std::milli>(tEnd -
  // tStart).count(); std::cout << "reorder_start_point_time:" << time_cost <<
  // std::endl;
}

}}  // namespace dataelem::alg