#include "dataelem/alg/ppcls.h"

#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(PPClsAngle);

TRITONSERVER_Error*
PPClsAngle::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "cls_angle";

  JValue params;
  auto& model_config = model_state->ModelConfig();
  model_config.Find("parameters", &params);
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, sub_graph_names_);

  std::string input_names_str;
  std::vector<std::string> input_names_vec;
  SafeParseParameter(params, "graph_input_name", &input_names_str, "");
  ParseArrayFromString(input_names_str, input_names_vec);
  if (input_names_vec.size() > 0 && input_names_vec[0].size() > 0) {
    sub_graph_io_names_.input_names = input_names_vec;
  }

  std::string output_names_str;
  std::vector<std::string> output_names_vec;
  SafeParseParameter(params, "graph_output_name", &output_names_str, "");
  ParseArrayFromString(output_names_str, output_names_vec);
  if (output_names_vec.size() > 0 && output_names_vec[0].size() > 0) {
    sub_graph_io_names_.output_names = output_names_vec;
  }

  SafeParseParameter(params, "max_w", &max_w_);
  SafeParseParameter(params, "min_w", &min_w_);
  SafeParseParameter(params, "img_h", &img_h_);
  SafeParseParameter(params, "batch_size", &batch_size_);
  SafeParseParameter(params, "fixed_batch", &fixed_batch_);
  SafeParseParameter(params, "version", &version_, 1);

  std::string str_batch_sizes;
  SafeParseParameter(params, "batch_sizes", &str_batch_sizes);
  if (str_batch_sizes.size() > 0) {
    size_t pos = 0;
    std::string token;
    std::string delimiter = ",";
    int side_len = 0;
    while ((pos = str_batch_sizes.find(delimiter)) != std::string::npos) {
      token = str_batch_sizes.substr(0, pos);
      side_len = atoi(token.c_str());
      batch_sizes_.push_back(side_len);
      str_batch_sizes.erase(0, pos + delimiter.length());
    }
    side_len = atoi(str_batch_sizes.c_str());
    batch_sizes_.push_back(side_len);
  }


  // std::cout<<"cls: batch_sizes_:";
  // for(size_t i=0; i<batch_sizes_.size(); i++){
  //   std::cout<<batch_sizes_[i]<<" ";
  // }
  // std::cout<<std::endl;

  return nullptr;
}

TRITONSERVER_Error*
PPClsAngle::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  cv::Mat processed_imgs = inputs[0].m();
  // print_mat_uint8(processed_imgs, "cls processed_imgs", false);
  // std::cout<<"data ptr:"<<(void*)inputs[0].data_ptr()<<std::endl;
  auto processed_imgs_width = inputs[1].m();
  // print_mat<int>(processed_imgs_width, "cls processed_imgs_width");
  auto processed_rot = inputs[2].m();

  int n = (version_ == 1) ? processed_imgs.size[1] : processed_imgs.size[0];
  int imgs_h =
      (version_ == 1) ? processed_imgs.size[2] : processed_imgs.size[1];
  int imgs_max_w =
      (version_ == 1) ? processed_imgs.size[3] : processed_imgs.size[2];
  int batchs = std::ceil(n * 1.0 / batch_size_);
  int step0 = 3 * imgs_h * imgs_max_w;
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(batchs);
  std::vector<cv::Mat> batched_imgs_vec;
  std::vector<int> real_batch_list;
  for (int j = 0; j < batchs; j++) {
    int s0 = j * batch_size_;
    int e0 = j == (batchs - 1) ? n : (j + 1) * batch_size_;
    int sn0 = e0 - s0;
    // std::cout<<"sn0:"<<sn0<<std::endl;
    real_batch_list.push_back(sn0);
    if (fixed_batch_) {
      sn0 = batch_size_;
    } else if (batch_sizes_.size() > 0) {
      int min_batch = 0;
      batch_sizes_.push_back(batch_size_);
      for (size_t i = 0; i < batch_sizes_.size(); i++) {
        if (sn0 > min_batch && sn0 <= batch_sizes_[i]) {
          sn0 = batch_sizes_[i];
          break;
        } else {
          min_batch = batch_sizes_[i];
        }
      }
    }
    // std::cout<<"sn0:"<<sn0<<std::endl;
    int max_w = (version_ == 1) ? processed_imgs_width.at<int>(0, e0 - 1)
                                : processed_imgs_width.at<int>(e0 - 1, 1);
    max_w = int(1.0 * max_w * img_h_ / imgs_h);
    int real_w = std::min(max_w_, std::max(max_w, min_w_));
    std::vector<int> batched_imgs_shape = {sn0, 3, img_h_, real_w};
    std::vector<int> img_shape = {3, img_h_, real_w};
    int step1 = 4 * 3 * img_h_ * real_w;
    cv::Mat batched_imgs =
        cv::Mat(batched_imgs_shape, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat img_src0 = cv::Mat(img_h_, real_w, CV_32FC3, cv::Scalar(0.0f));
    for (int i = s0; i < e0; i++) {
      int w = (version_ == 1) ? processed_imgs_width.at<int>(0, i)
                              : processed_imgs_width.at<int>(i, 1);
      cv::Mat img_hw =
          cv::Mat(imgs_h, imgs_max_w, CV_8UC3, processed_imgs.data + i * step0);
      cv::Mat imgroi0 = img_hw(cv::Rect(0, 0, w, imgs_h));
      // cv::imwrite(std::to_string(i)+".png", imgroi0);
      cv::Mat imgroi;
      w = int(1.0 * w * img_h_ / imgs_h);
      if (imgs_h != img_h_) {
        cv::resize(imgroi0, imgroi, {w, img_h_}, 0, 0, cv::INTER_LINEAR);
      } else {
        imgroi0.copyTo(imgroi);
      }
      /*if(w > max_w_ || w < min_w_){
        cv::resize(imgroi, img_src0(cv::Rect(0,0,real_w,img_h_)), {real_w,
      img_h_}, 0, 0, cv::INTER_LINEAR); w = real_w;
      }
      else{
        imgroi.copyTo(img_src0(cv::Rect(0,0,w,img_h_)));
      }*/

      // print_mat_uint8(imgroi, "cls imgroi 0", false);

      if (w > max_w_) {
        cv::resize(imgroi, imgroi, {real_w, img_h_}, 0, 0, cv::INTER_LINEAR);
        w = real_w;
      }
      /*else if(w < min_w_){
        cv::resize(imgroi, imgroi, {min_w_, img_h_}, 0, 0, cv::INTER_LINEAR);
        w = min_w_;
      }*/

      /*print_mat_uint8(imgroi, "cls imgroi", false);
      cv::Mat showdata0 = imgroi.clone();
      cv::Mat img_show2 = cv::Mat({3, img_h_, w}, CV_8UC1, cv::Scalar(0.0f));
      cv::Mat img_show0 = cv::Mat(img_h_*w, 3, CV_8UC1, showdata0.data);
      cv::Mat img_show1 = cv::Mat(3, img_h_*w, CV_8UC1, img_show2.data);
      cv::transpose(img_show0, img_show1);
      print_mat_uint8(img_show2, "cls img_show2", false);
      */

      // imgroi.convertTo(img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 2.0 /
      // 255.0, -1.0);
      imgroi.convertTo(img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 1.0, 0.0);
      float ratio = 2.0 / 255.0;
      img_src0(cv::Rect(0, 0, w, img_h_))
          .convertTo(
              img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, ratio, -1.0);

      /*cv::Mat showdata1 = img_src0(cv::Rect(0, 0, w, img_h_)).clone();

      cv::Mat img_show5 = cv::Mat({3, img_h_, w}, CV_32FC1, cv::Scalar(0.0f));
      cv::Mat img_show3 = cv::Mat(img_h_*w, 3, CV_32FC1, showdata1.data);
      cv::Mat img_show4 = cv::Mat(3, img_h_*w, CV_32FC1, img_show5.data);
      cv::transpose(img_show3, img_show4);
      print_mat<float>(img_show5, "cls img_show5");
      */
      cv::Mat img_dst = cv::Mat(
          3, img_h_ * real_w, CV_32FC1, batched_imgs.data + (i - s0) * step1);
      cv::Mat img_src = cv::Mat(img_h_ * real_w, 3, CV_32FC1, img_src0.data);
      cv::transpose(img_src, img_dst);
    }

    batched_imgs_vec.push_back(batched_imgs);
    /*if(j == 0){
      cv::FileStorage file("batched_imgs_cls_cpp.ext", cv::FileStorage::WRITE);
      file << "batched_imgs" << batched_imgs;
      file.release();
    }
    */

    // print_mat<float>(batched_imgs, "cls batched_imgs");

    OCTensorList inputs = {OCTensor(std::move(batched_imgs))};
    // construct request and exec internal infer
    TRITONSERVER_InferenceRequest* req = nullptr;
    RETURN_IF_ERROR(CreateServerRequestWithTensors(
        context->GetBackendRequestInfo(), graph_executor_->GetServer(),
        sub_graph_names_[0].c_str(), &inputs, sub_graph_io_names_.input_names,
        sub_graph_io_names_.output_names, &req));
    RETURN_IF_ERROR(graph_executor_->AsyncExecute(req, context, &futures[j]));
  }

  std::vector<int> shape = {1, n};
  if (version_ == 2) {
    shape = {n};
  }
  cv::Mat angles = cv::Mat(shape, CV_8U, cv::Scalar(0));
  cv::Mat angles_prob = cv::Mat(shape, CV_32F, cv::Scalar(0.0f));

  for (int j = 0; j < batchs; j++) {
    auto* resp = futures[j].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, sub_graph_io_names_.output_names, &outputs));
    // GraphInferResponseDelete(resp);
    auto feat_map = outputs[0].m().clone();
    int batch_size = real_batch_list[j];
    for (int64_t i = 0; i < batch_size; i++) {
      uint8_t rot = (version_ == 1)
                        ? processed_rot.at<uint8_t>(0, j * batch_size_ + i)
                        : processed_rot.at<uint8_t>(j * batch_size_ + i, 0);
      if (feat_map.at<float>(i, 0) >= feat_map.at<float>(i, 1)) {
        if (version_ == 1) {
          angles.at<uint8_t>(0, j * batch_size_ + i) = rot;
          angles_prob.at<float>(0, j * batch_size_ + i) =
              feat_map.at<float>(i, 0);
        } else {
          angles.at<uint8_t>(j * batch_size_ + i, 0) = rot;
          angles_prob.at<float>(j * batch_size_ + i, 0) =
              feat_map.at<float>(i, 0);
        }
      } else {
        if (version_ == 1) {
          angles.at<uint8_t>(0, j * batch_size_ + i) = 2 + rot;
          angles_prob.at<float>(0, j * batch_size_ + i) =
              feat_map.at<float>(i, 1);
        } else {
          angles.at<uint8_t>(j * batch_size_ + i, 0) = 2 + rot;
          angles_prob.at<float>(j * batch_size_ + i, 0) =
              feat_map.at<float>(i, 1);
        }
      }
    }
    GraphInferResponseDelete(resp);
  }

  context->SetTensor(
      io_names_.output_names, {std::move(angles), std::move(angles_prob)});

  return nullptr;
}

}}  // namespace dataelem::alg
