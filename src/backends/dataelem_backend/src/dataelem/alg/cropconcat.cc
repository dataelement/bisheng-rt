#include "dataelem/alg/cropconcat.h"

#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(CropConcat);

TRITONSERVER_Error*
CropConcat::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "crop_concat";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "max_w_", &max_w_);
  SafeParseParameter(params, "img_h_", &img_h_);
  SafeParseParameter(params, "rot_bbox", &rot_bbox_);

  return nullptr;
}

TRITONSERVER_Error*
CropConcat::Execute(AlgRunContext* context)
{
  bool rot_bbox = rot_bbox_;
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

    if (params.contains("rot_bbox")) {
      rot_bbox = params["rot_bbox"].get<int>();
    }
  }

  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  auto org_imgs = inputs[0].m();
  cv::Mat ori_img =
      cv::Mat(org_imgs.size[1], org_imgs.size[2], CV_8UC3, org_imgs.data);
  auto bboxes = inputs[1].m();
  auto bboxes_score = inputs[2].m();
  /*if(bboxes.empty() || bboxes_score.empty()){
    std::vector<int> bboxes_shape = {1, 0, 4, 2};
    std::vector<int> bboxes_score_shape = {1, 0};
    std::vector<int> processed_imgs_shape = {1, 0, img_h_, max_w, 3};
    cv::Mat bboxes_new = cv::Mat(bboxes_shape, CV_32S, cv::Scalar(0));
    cv::Mat bboxes_score_new = cv::Mat(bboxes_score_shape, CV_32F,
  cv::Scalar(0.0f)); cv::Mat processed_rot = cv::Mat(bboxes_score_shape, CV_8U,
  cv::Scalar(0)); cv::Mat processed_imgs_width = cv::Mat(bboxes_score_shape,
  CV_32S, cv::Scalar(0)); cv::Mat processed_imgs = cv::Mat(processed_imgs_shape,
  CV_8U, cv::Scalar(0));

    context->SetTensor(
      io_names_.output_names, {std::move(processed_imgs),
  std::move(processed_imgs_width), std::move(bboxes_new),
  std::move(bboxes_score_new), std::move(processed_rot)});

    return nullptr;
  }*/

  int n = bboxes.size[1];

  cv::Mat bbmf;
  cv::Mat_<cv::Point2f> bbs;
  if (n > 0) {
    bboxes.convertTo(bbmf, CV_32FC1);
    bbs = cv::Mat_<cv::Point2f>(bbmf);
  }

  // sort bboxes by w/h ratio
  std::vector<std::pair<int, int>> temp_widths;
  std::vector<cv::Mat> img_list;
  std::vector<uint8_t> rot_list;
  if (rot_bbox) {
    for (int i = 0; i < n; i++) {
      std::vector<cv::Point2f> v = {
          bbs(0, i, 0), bbs(0, i, 1), bbs(0, i, 2), bbs(0, i, 3)};
      float w = floor(std::max(l2_norm(v[0], v[1]), l2_norm(v[2], v[3])));
      float h = floor(std::max(l2_norm(v[1], v[2]), l2_norm(v[0], v[3])));

      if (h / w >= hw_thrd_) {
        float temp = h;
        h = w;
        w = temp;
        rot_list.push_back(1);
      } else {
        rot_list.push_back(0);
      }

      int new_w = int(ceil(w * img_h_ / h));
      temp_widths.emplace_back(make_pair(i, new_w));
    }
  } else {
    std::vector<float> w_list;
    std::vector<float> h_list;
    float w = 0;
    float h = 0;
    uint8_t rot = 0;
    for (int i = 0; i < n; i++) {
      std::vector<cv::Point2f> v = {
          bbs(0, i, 0), bbs(0, i, 1), bbs(0, i, 2), bbs(0, i, 3)};

      float w0 = floor(std::max(l2_norm(v[0], v[1]), l2_norm(v[2], v[3])));
      float h0 = floor(std::max(l2_norm(v[1], v[2]), l2_norm(v[0], v[3])));
      w_list.push_back(w0);
      h_list.push_back(h0);
      w += w0;
      h += h0;
    }

    bool flag = h / w >= hw_thrd_;
    if (flag) {
      rot = 1;
    }

    for (int i = 0; i < n; i++) {
      rot_list.push_back(rot);
      w = w_list[i];
      h = h_list[i];
      if (flag) {
        float temp = h;
        h = w;
        w = temp;
      }

      int new_w = int(ceil(w * img_h_ / h));
      temp_widths.emplace_back(make_pair(i, new_w));
    }
  }

  if (n > 0) {
    std::stable_sort(
        temp_widths.begin(), temp_widths.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
          return p1.second < p2.second;
        });
  }

  /*for(int i=0; i<n; i++){
    std::cout<<temp_widths[i].first<<" ";
  }
  std::cout<<std::endl;

  for(int i=0; i<n; i++){
    std::cout<<temp_widths[i].second<<" ";
  }
  std::cout<<std::endl;*/

  int max_w = n > 0 ? std::min(max_w_, temp_widths[n - 1].second) : 0;
  std::vector<int> bboxes_shape = {1, n, 4, 2};
  std::vector<int> bboxes_score_shape = {1, n};
  std::vector<int> processed_imgs_shape = {1, n, img_h_, max_w, 3};
  cv::Mat bboxes_new = cv::Mat(bboxes_shape, CV_32S, cv::Scalar(0));
  cv::Mat bboxes_score_new =
      cv::Mat(bboxes_score_shape, CV_32F, cv::Scalar(0.0f));
  cv::Mat processed_rot = cv::Mat(bboxes_score_shape, CV_8U, cv::Scalar(0));
  cv::Mat processed_imgs_width =
      cv::Mat(bboxes_score_shape, CV_32S, cv::Scalar(0));
  cv::Mat processed_imgs = cv::Mat(processed_imgs_shape, CV_8U, cv::Scalar(0));
  auto* bbox_ptr = reinterpret_cast<int*>(bboxes.data);
  auto* bbox_new_ptr = reinterpret_cast<int*>(bboxes_new.data);
  int step = 3 * img_h_ * max_w;
  for (int i = 0; i < n; i++) {
    int m_index = temp_widths[i].first;
    int w = std::min(max_w, temp_widths[i].second);


    for (int j = 0; j < 8; j++) {
      *(bbox_new_ptr + 8 * i + j) = (int)*(bbox_ptr + 8 * m_index + j);
    }

    bboxes_score_new.at<float>(0, i) = bboxes_score.at<float>(0, m_index);
    processed_rot.at<uint8_t>(0, i) = rot_list[m_index];
    processed_imgs_width.at<int>(0, i) = w;

    /*std::vector<cv::Point2f> src_3points = {bbs(0, m_index, 0), bbs(0,
    m_index, 1), bbs(0, m_index, 2)}; if(rot_list[i] == 1){ src_3points =
    {bbs(0, m_index, 3), bbs(0, m_index, 0), bbs(0, m_index, 1)};
    }

    std::vector<cv::Point2f> dst_3points = {{0, 0}, {(float)w, 0}, {(float)w,
    (float)img_h_}}; cv::Mat warp_mat = cv::getAffineTransform(src_3points,
    dst_3points);*/

    std::vector<cv::Point2f> src_4points = {
        bbs(0, m_index, 0), bbs(0, m_index, 1), bbs(0, m_index, 2),
        bbs(0, m_index, 3)};
    /*if(rot_list[i] == 1){
      src_4points = {bbs(0, m_index, 3), bbs(0, m_index, 0), bbs(0, m_index, 1),
    bbs(0, m_index, 2)};
    }*/

    // float raw_w = floor(l2_norm(src_4points[0], src_4points[1]));
    // float raw_h = floor(l2_norm(src_4points[1], src_4points[2]));
    float raw_w = floor(std::max(
        l2_norm(src_4points[0], src_4points[1]),
        l2_norm(src_4points[2], src_4points[3])));
    float raw_h = floor(std::max(
        l2_norm(src_4points[1], src_4points[2]),
        l2_norm(src_4points[0], src_4points[3])));

    std::vector<cv::Point2f> dst_4points = {
        {0, 0}, {raw_w, 0}, {raw_w, raw_h}, {0, raw_h}};

    // std::cout<<"src_4points:"<<src_4points[0].x<<" "<<src_4points[0].y<<"
    // "<<src_4points[1].x<<" "<<src_4points[1].y<<" "<<src_4points[2].x<<"
    // "<<src_4points[2].y<<" "<<src_4points[3].x<<"
    // "<<src_4points[3].y<<std::endl;
    // std::cout<<"dst_4points:"<<dst_4points[0].x<<" "<<dst_4points[0].y<<"
    // "<<dst_4points[1].x<<" "<<dst_4points[1].y<<" "<<dst_4points[2].x<<"
    // "<<dst_4points[2].y<<" "<<dst_4points[3].x<<"
    // "<<dst_4points[3].y<<std::endl;

    cv::Mat warp_mat = cv::getPerspectiveTransform(src_4points, dst_4points);

    // print_mat<float>(warp_mat, "warp_mat");
    cv::Mat img =
        cv::Mat(img_h_, max_w, CV_8UC3, processed_imgs.data + i * step);
    cv::Mat m;
    // cv::warpAffine(ori_img, m, warp_mat, {w, img_h_}, cv::INTER_LINEAR);
    // cv::warpAffine(ori_img, m, warp_mat, {w, img_h_}, cv::INTER_CUBIC,
    // cv::BORDER_REPLICATE);
    cv::warpPerspective(
        ori_img, m, warp_mat, cv::Size((int)raw_w, (int)raw_h), cv::INTER_CUBIC,
        cv::BORDER_REPLICATE);
    // print_mat<uint8_t>(m, "m0");
    // m.convertTo(img(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 2.0 / 255.0, -1.0);
    // m.copyTo(img(cv::Rect(0, 0, w, img_h_)));
    // m.convertTo(m, CV_32FC3, 1.0, 0);
    // print_mat<float>(m, "m1");
    if (rot_list[m_index] == 1) {
      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
      // std::cout<<"rotate"<<std::endl;
    }

    // std::cout<<"rot:"<<(int)rot_list[m_index]<<std::endl;

    // print_mat<uint8_t>(m, "m");
    cv::resize(m, m, {w, img_h_}, 0, 0, cv::INTER_LINEAR);
    m.copyTo(img(cv::Rect(0, 0, w, img_h_)));
    // print_mat<uint8_t>(img(cv::Rect(0, 0, w, img_h_)), "img");
    /*if(src_4points[0].x == 265 && src_4points[0].y == 540){
      print_mat<uint8_t>(img(cv::Rect(0, 0, w, img_h_)), "img");
      std::cout<<"i:"<<i<<" mindex:"<<m_index<<std::endl;
    }*/
  }

  /*print_mat<uint8_t>(processed_rot, "cropconcat processed_rot");

  print_mat<uint8_t>(processed_imgs, "cropconcat processed_imgs");
  std::cout<<"data ptr:"<<(void*)processed_imgs.data<<std::endl;
  */
  context->SetTensor(
      io_names_.output_names,
      {std::move(processed_imgs), std::move(processed_imgs_width),
       std::move(bboxes_new), std::move(bboxes_score_new),
       std::move(processed_rot)});

  return nullptr;
}

}}  // namespace dataelem::alg
