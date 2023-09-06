#include "dataelem/alg/ppdet.h"

#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "ext/ppocr/postprocess_op.h"
#include "nlohmann/json.hpp"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(PPDetDBPrep);
REGISTER_ALG_CLASS(PPDetDBPost);

TRITONSERVER_Error*
PPDetDBPrep::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "det_db_prep";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);

  SafeParseParameter(params, "max_side_len", &max_side_len_);
  SafeParseParameter(params, "fixed_shape", &fixed_shape_);
  std::string str_min_side_lens;
  SafeParseParameter(params, "min_side_lens", &str_min_side_lens);
  if (str_min_side_lens.size() > 0) {
    size_t pos = 0;
    std::string token;
    std::string delimiter = ",";
    int side_len = 0;
    while ((pos = str_min_side_lens.find(delimiter)) != std::string::npos) {
      token = str_min_side_lens.substr(0, pos);
      side_len = atoi(token.c_str());
      min_side_lens_.push_back(side_len);
      str_min_side_lens.erase(0, pos + delimiter.length());
    }
    side_len = atoi(str_min_side_lens.c_str());
    min_side_lens_.push_back(side_len);
  }

  SafeParseParameter(params, "version", &version_, 1);

  // std::cout<<"det: min_side_lens_:";
  // for(size_t i=0; i<min_side_lens_.size(); i++){
  //   std::cout<<min_side_lens_[i]<<" ";
  // }
  // std::cout<<std::endl;

  return nullptr;
}

TRITONSERVER_Error*
PPDetDBPrep::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  cv::Mat img0, img1, imgs, shape_list;
  if (version_ == 1) {
    imgs = inputs[0].m();
    img0 = cv::Mat(imgs.size[1], imgs.size[2], CV_8UC3, imgs.data);
  } else {
    cv::Mat img = inputs[0].m();
    img0 = cv::Mat(img.size[0], img.size[1], CV_8UC3, img.data);
    std::vector<int> ori_imgs_shape = {1, img0.size[0], img0.size[1], 3};
    imgs = cv::Mat(ori_imgs_shape, CV_8U, img0.data).clone();

    float h = float(img0.size[0]);
    float w = float(img0.size[1]);
    float r = max_side_len_ / std::max(h, w);

    float resize_h = std::floor(h * r);
    float resize_w = std::floor(w * r);
    resize_h = std::max(std::nearbyint(resize_h / 32.0) * 32.0, 32.0);
    resize_w = std::max(std::nearbyint(resize_w / 32.0) * 32.0, 32.0);

    float ratio_h = resize_h / h;
    float ratio_w = resize_w / w;

    std::vector<float> shape_vec = {h, w, ratio_h, ratio_w};
    std::vector<int> shape = {1, 4};
    shape_list = cv::Mat(shape, CV_32FC1, shape_vec.data()).clone();
  }

  img0.convertTo(img1, CV_32FC3, 1.0, 0.0);

  cv::Mat resized_img;
  int resize_h0 = imgs.size[1];
  int resize_w0 = imgs.size[2];

  float h = float(imgs.size[1]);
  float w = float(imgs.size[2]);
  float r = max_side_len_ / std::max(h, w);

  float resize_h = std::floor(h * r);
  float resize_w = std::floor(w * r);
  resize_h = std::max(std::nearbyint(resize_h / 32.0) * 32.0, 32.0);
  resize_w = std::max(std::nearbyint(resize_w / 32.0) * 32.0, 32.0);
  resize_h0 = int(resize_h);
  resize_w0 = int(resize_w);
  cv::resize(img1, resized_img, {resize_w0, resize_h0}, 0, 0, cv::INTER_LINEAR);

  // std::cout<<"resize_w0:"<<resize_w0<<" resize_h0:"<<resize_h0<<std::endl;
  int output_h = fixed_shape_ ? max_side_len_ : resize_h0;
  int output_w = fixed_shape_ ? max_side_len_ : resize_w0;
  if (min_side_lens_.size() > 0) {
    min_side_lens_.push_back(max_side_len_);
    if (output_h < max_side_len_) {
      int min_h = 0;
      for (size_t i = 0; i < min_side_lens_.size(); i++) {
        if (output_h > min_h && output_h <= min_side_lens_[i]) {
          output_h = min_side_lens_[i];
          break;
        } else {
          min_h = min_side_lens_[i];
        }
      }
    } else if (output_w < max_side_len_) {
      int min_w = 0;
      for (size_t i = 0; i < min_side_lens_.size(); i++) {
        if (output_w > min_w && output_w <= min_side_lens_[i]) {
          output_w = min_side_lens_[i];
          break;
        } else {
          min_w = min_side_lens_[i];
        }
      }
    }
  }

  // std::cout<<"output_w:"<<output_w<<" output_h:"<<output_h<<std::endl;

  std::vector<int> batched_imgs_shape = {1, 3, output_h, output_w};
  cv::Mat batched_imgs =
      cv::Mat(batched_imgs_shape, CV_32FC1, cv::Scalar(0.0f));
  cv::Mat resized_img_pad =
      cv::Mat(output_h, output_w, CV_32FC3, cv::Scalar(0.0f));

  // print_mat<float>(resized_img, "resized_img");

  double e = 1.0 / 255.0;
  (resized_img).convertTo(resized_img, CV_32FC3, e);
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(resized_img, bgr_channels);
  for (size_t i = 0; i < bgr_channels.size(); i++) {
    bgr_channels[i].convertTo(
        bgr_channels[i], CV_32FC1, 1.0 * scale_[i],
        (0.0 - mean_[i]) * scale_[i]);
  }
  cv::merge(bgr_channels, resized_img);
  // print_mat<float>(resized_img, "resized_img");
  resized_img.copyTo(resized_img_pad(cv::Rect(0, 0, resize_w0, resize_h0)));
  // print_mat<float>(resized_img_pad, "resized_img_pad");

  cv::Mat img_dst =
      cv::Mat(3, output_h * output_w, CV_32FC1, batched_imgs.data);
  cv::Mat img_src =
      cv::Mat(output_h * output_w, 3, CV_32FC1, resized_img_pad.data);
  cv::transpose(img_src, img_dst);

  // print_mat<float>(batched_imgs, "batched_imgs");
  // float *p = (float*)batched_imgs.data;
  // float temp = 0.0f;
  // for(int i=0; i<960*960*3; i++){
  //   temp += *(p+i);
  // }
  // std::cout<<"dataelem in:"<<temp<<std::endl;

  if (version_ == 1) {
    context->SetTensor(io_names_.output_names, {std::move(batched_imgs)});
  } else {
    context->SetTensor(
        io_names_.output_names,
        {std::move(batched_imgs), std::move(shape_list)});
  }

  return nullptr;
}

TRITONSERVER_Error*
PPDetDBPost::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "det_db_post";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

  SafeParseParameter(params, "det_db_thresh", &det_db_thresh_);
  SafeParseParameter(params, "det_db_box_thresh", &det_db_box_thresh_);
  SafeParseParameter(params, "det_db_unclip_ratio", &det_db_unclip_ratio_);
  SafeParseParameter(params, "det_db_score_mode", &det_db_score_mode_);
  SafeParseParameter(params, "use_dilation", &use_dilation_);
  SafeParseParameter(params, "delta_w", &delta_w_);
  SafeParseParameter(params, "delta_h", &delta_h_);
  SafeParseParameter(params, "version", &version_, 1);

  return nullptr;
}

cv::Mat
getroi(cv::Mat featmap, int h, int w)
{
  int featH = featmap.size[2];
  int featW = featmap.size[3];
  cv::Mat m0 = cv::Mat(featH, featW, CV_32FC1, featmap.data);
  cv::Mat m1 = m0(cv::Rect(0, 0, w, h)).clone();
  std::vector<int> shape = {1, 1, h, w};
  cv::Mat m2 = cv::Mat(shape, CV_32FC1, m1.data);
  return m2.clone();
}

void
PPDetDBPost::EnlageBbox(
    std::vector<std::vector<std::vector<int>>>& bbox, int delta_w0,
    int delta_h0)
{
  float hw_thrd = 1.5;
  float w = 0;
  float h = 0;
  for (size_t i = 0; i < bbox.size(); i++) {
    std::vector<cv::Point2f> v = {
        cv::Point2f(bbox[i][0][0], bbox[i][0][1]),
        cv::Point2f(bbox[i][1][0], bbox[i][1][1]),
        cv::Point2f(bbox[i][2][0], bbox[i][2][1]),
        cv::Point2f(bbox[i][3][0], bbox[i][3][1])};
    w += std::max(l2_norm(v[0], v[1]), l2_norm(v[2], v[3]));
    h += std::max(l2_norm(v[1], v[2]), l2_norm(v[0], v[3]));
  }

  float delta_w = delta_w0;
  float delta_h = delta_h0;
  if (w > 0 && h / w >= hw_thrd) {
    delta_w = delta_h0;
    delta_h = delta_w0;
  }

  for (size_t i = 0; i < bbox.size(); i++) {
    std::vector<cv::Point2f> src_points, dest_points;
    for (size_t j = 0; j < bbox[i].size(); j++) {
      src_points.push_back(cv::Point2f(bbox[i][j][0], bbox[i][j][1]));
    }

    enLargeRRectOp(src_points, dest_points, delta_w, delta_h);
    for (size_t j = 0; j < bbox[i].size(); j++) {
      bbox[i][j][0] = dest_points[j].x;
      bbox[i][j][1] = dest_points[j].y;
    }

    bbox[i] = post_processor_.OrderPointsClockwise(bbox[i]);
  }
}

TRITONSERVER_Error*
PPDetDBPost::Execute(AlgRunContext* context)
{
  int delta_w = delta_w_;
  int delta_h = delta_h_;
  if (optional_inputs_.size() > 0) {
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

    if (params.contains("delta_w")) {
      delta_w = params["delta_w"].get<int>();
      if (delta_w < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "delta_w must be greater than zero");
      }
    }
    if (params.contains("delta_h")) {
      delta_h = params["delta_h"].get<int>();
      if (delta_h < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "delta_h must be greater than zero");
      }
    }
  }

  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  const cv::Mat& out_feamap = inputs[0].m();
  const cv::Mat& shape_list = inputs[1].m();  // (1, 4)
  // print_mat<float>(out_feamap, "out_feamap");
  // print_mat<float>(shape_list, "shape_list");

  int ori_h = (int)shape_list.at<float>(0, 0);
  int ori_w = (int)shape_list.at<float>(0, 1);
  float ratio_h = shape_list.at<float>(0, 2);
  float ratio_w = shape_list.at<float>(0, 3);

  int featH = out_feamap.size[2];
  int featW = out_feamap.size[3];
  int realH = (int)(shape_list.at<float>(0, 0) * ratio_h);
  int realW = (int)(shape_list.at<float>(0, 1) * ratio_w);
  // std::cout<<"featW:"<<featW<<" featH:"<<featH<<" realW:"<<realW<<"
  // realH:"<<realH<<std::endl;
  cv::Mat new_featmap;
  if (realH < featH || realW < featW) {
    new_featmap = getroi(out_feamap, realH, realW);
  } else {
    new_featmap = out_feamap;
  }

  float* out_data = reinterpret_cast<float*>(new_featmap.data);
  int n2 = realH;
  int n3 = realW;
  int n = n2 * n3;

  std::vector<float> pred(n, 0.0);
  std::vector<uint8_t> mask(n, 0);
  // std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    // cbuf[i] = (unsigned char)((out_data[i]) * 255);
    if (pred[i] > det_db_thresh_) {
      mask[i] = 255;
    } else {
      mask[i] = 0;
    }
  }

  // cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char*)cbuf.data());
  cv::Mat bit_map(n2, n3, CV_8UC1, mask.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float*)pred.data());

  /*const double threshold = det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map, bit_map_u8;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (use_dilation_) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }*/

  std::vector<std::vector<std::vector<int>>> boxes;

  // print_mat<uint8_t>(bit_map, "bit_map");

  // boxes: (n, 4, 2)
  std::vector<float> scores;
  boxes = post_processor_.BoxesFromBitmap(
      pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
      det_db_score_mode_, scores, ori_w, ori_h);

  std::vector<float> filter_scores;
  boxes = post_processor_.FilterTagDetRes(
      boxes, ratio_h, ratio_w, ori_h, ori_w, scores, filter_scores);

  // std::cout<<"bboxes:"<<std::endl;
  // for(size_t i=0; i<boxes.size(); i++){
  //   std::cout<<boxes[i][0][0]<<" "<<boxes[i][0][1]<<" ";
  //   std::cout<<boxes[i][1][0]<<" "<<boxes[i][1][1]<<" ";
  //   std::cout<<boxes[i][2][0]<<" "<<boxes[i][2][1]<<" ";
  //   std::cout<<boxes[i][3][0]<<" "<<boxes[i][3][1]<<std::endl;
  // }

  if (delta_w > 0 || delta_h > 0) {
    EnlageBbox(boxes, delta_w, delta_h);
  }

  int bb_cnt = boxes.size();
  std::vector<int> bbox_shape = {1, bb_cnt, 4, 2};
  std::vector<int> score_shape = {1, bb_cnt};
  cv::Mat output0 = cv::Mat(bbox_shape, CV_32S);
  if (version_ == 2) {
    bbox_shape = {bb_cnt, 4, 2};
  }
  if (version_ == 1) {
    auto* ptr = reinterpret_cast<int*>(output0.data);
    for (int i = 0; i < bb_cnt; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
          *(ptr + 8 * i + j * 2 + k) = boxes[i][j][k];
        }
      }
    }
  } else {
    output0 = cv::Mat(bbox_shape, CV_32F);
    auto* ptr = reinterpret_cast<float*>(output0.data);
    for (int i = 0; i < bb_cnt; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
          *(ptr + 8 * i + j * 2 + k) = (float)boxes[i][j][k];
        }
      }
    }
  }


  cv::Mat output1;
  if (bb_cnt == 0) {
    output1 = cv::Mat(score_shape, CV_32F);
  } else {
    output1 = cv::Mat(score_shape, CV_32F, filter_scores.data()).clone();
  }

  if (version_ == 1) {
    context->SetTensor(
        io_names_.output_names, {std::move(output0), std::move(output1)});
  } else {
    auto t1 = OCTensor(std::move(output1));
    t1.set_shape({bb_cnt});

    cv::Mat src_scale = (cv::Mat_<int>(1, 3) << ori_h, ori_w, 3);
    context->SetTensor(
        io_names_.output_names,
        {std::move(output0), std::move(t1), std::move(src_scale)});
  }

  return nullptr;
}

}}  // namespace dataelem::alg
