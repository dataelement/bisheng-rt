#include "dataelem/alg/pprec.h"

#include <array>

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"
#include "triton/common/local_filesystem.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(PPRecCh);
REGISTER_ALG_CLASS(PPRecLatin);

std::vector<std::string>
PPRec::ReadDict(const std::string& path)
{
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

TRITONSERVER_Error*
PPRec::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);

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
  SafeParseParameter(params, "thresh", &thresh_);
  SafeParseParameter(params, "output_matrix", &output_matrix_);
  SafeParseParameter(params, "fixed_batch", &fixed_batch_);
  SafeParseParameter(params, "process_type", &process_type_);
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
  std::string str_w_sizes;
  SafeParseParameter(params, "w_sizes", &str_w_sizes);
  if (str_w_sizes.size() > 0) {
    size_t pos = 0;
    std::string token;
    std::string delimiter = ",";
    int side_len = 0;
    while ((pos = str_w_sizes.find(delimiter)) != std::string::npos) {
      token = str_w_sizes.substr(0, pos);
      side_len = atoi(token.c_str());
      w_sizes_.push_back(side_len);
      str_w_sizes.erase(0, pos + delimiter.length());
    }
    side_len = atoi(str_w_sizes.c_str());
    w_sizes_.push_back(side_len);
  }


  // std::cout<<"rec: batch_sizes_:";
  // for(size_t i=0; i<batch_sizes_.size(); i++){
  //   std::cout<<batch_sizes_[i]<<" ";
  // }
  // std::cout<<std::endl;

  // std::cout<<"rec: w_sizes_:";
  // for(size_t i=0; i<w_sizes_.size(); i++){
  //   std::cout<<w_sizes_[i]<<" ";
  // }
  // std::cout<<std::endl;

  std::string charset_name = "character_dict.txt";
  SafeParseParameter(params, "charset_name", &charset_name);
  auto repo_path = model_state->RepositoryPath();

  std::string label_path =
      triton::common::JoinPath({repo_path, "..", "resource", charset_name});

  bool is_exist = false;
  triton::common::FileExists(label_path, &is_exist);
  if (!is_exist) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "character file not exists.");
  }

  label_list_ = ReadDict(label_path);
  label_list_.insert(label_list_.begin(), "#");
  label_list_.push_back(" ");

  return nullptr;
}

// static int CNT0 = 0;
TRITONSERVER_Error*
PPRec::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  auto processed_imgs = inputs[0].m();
  auto processed_imgs_width = inputs[1].m();
  cv::Mat angles;
  cv::Mat angles_prob;
  if (version_ == 1) {
    angles = inputs[2].m();
    angles_prob = inputs[3].m();
  }

  int n = (version_ == 1) ? processed_imgs.size[1] : processed_imgs.size[0];
  int imgs_h =
      (version_ == 1) ? processed_imgs.size[2] : processed_imgs.size[1];
  int imgs_max_w =
      (version_ == 1) ? processed_imgs.size[3] : processed_imgs.size[2];
  // std::cout<<"imgs_max_w:"<<imgs_max_w<<" n:"<<n<<"
  // imgs_h:"<<imgs_h<<std::endl;
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
    // 如果输入图片不是按w排序，需要下面逻辑
    for (int j = s0; j < e0 - 1; j++) {
      int w_temp = (version_ == 1) ? processed_imgs_width.at<int>(0, j)
                                   : processed_imgs_width.at<int>(j, 1);
      max_w = std::max(max_w, w_temp);
    }
    max_w = int(1.0 * max_w * img_h_ / imgs_h);
    // std::cout<<"max_w:"<<max_w<<std::endl;
    int real_w = std::min(max_w_, std::max(max_w, min_w_));
    // std::cout<<"real_w:"<<real_w<<std::endl;
    if (real_w < max_w_ && w_sizes_.size() > 0) {
      int min_w = 0;
      w_sizes_.push_back(max_w_);
      for (size_t i = 0; i < w_sizes_.size(); i++) {
        if (real_w > min_w && real_w <= w_sizes_[i]) {
          real_w = w_sizes_[i];
          break;
        } else {
          min_w = w_sizes_[i];
        }
      }
    }
    // std::cout<<"real_w:"<<real_w<<std::endl;
    std::vector<int> batched_imgs_shape = {sn0, 3, img_h_, real_w};
    std::vector<int> img_shape = {3, img_h_, real_w};
    int step1 = 4 * 3 * img_h_ * real_w;
    cv::Mat batched_imgs =
        cv::Mat(batched_imgs_shape, CV_32FC1, cv::Scalar(0.0f));
    // cv::Mat img_src0 = cv::Mat(img_h_, real_w, CV_32FC3, cv::Scalar(0.0f));
    cv::Mat imgroi;
    for (int i = s0; i < e0; i++) {
      cv::Mat img_src0 = cv::Mat(img_h_, real_w, CV_32FC3, cv::Scalar(0.0f));
      int w0 = (version_ == 1) ? processed_imgs_width.at<int>(0, i)
                               : processed_imgs_width.at<int>(i, 1);
      // std::cout<<"w0:"<<w0<<" imgs_h:"<<imgs_h<<"
      // imgs_max_w:"<<imgs_max_w<<std::endl;
      cv::Mat img_hw =
          cv::Mat(imgs_h, imgs_max_w, CV_8UC3, processed_imgs.data + i * step0);
      cv::Mat imgroi0 = img_hw(cv::Rect(0, 0, w0, imgs_h));
      // cv::imwrite(std::to_string(i+100)+".png", imgroi0);
      int w = int(1.0 * w0 * img_h_ / imgs_h);
      if (imgs_h != img_h_) {
        cv::resize(imgroi0, imgroi, {w, img_h_}, 0, 0, cv::INTER_LINEAR);
      } else {
        imgroi0.copyTo(imgroi);
      }

      if (version_ == 1) {
        auto angle = (int)angles.at<uint8_t>(0, i);
        auto angle_prob = angles_prob.at<float>(0, i);
        if (angle >= 2 && angle_prob >= thresh_) {
          cv::rotate(imgroi, imgroi, cv::ROTATE_180);
        }
      }

      if (w > max_w_) {
        cv::resize(imgroi, imgroi, {real_w, img_h_}, 0, 0, cv::INTER_LINEAR);
        w = real_w;
      }

      // cv::imwrite(std::to_string(i)+".png", imgroi);

      /*print_mat_uint8(imgroi, "rec imgroi", false);
      cv::Mat showdata0 = imgroi.clone();
      cv::Mat img_show2 = cv::Mat({3, img_h_, w}, CV_8UC1, cv::Scalar(0));
      cv::Mat img_show0 = cv::Mat(img_h_*w, 3, CV_8UC1, showdata0.data);
      cv::Mat img_show1 = cv::Mat(3, img_h_*w, CV_8UC1, img_show2.data);
      cv::transpose(img_show0, img_show1);
      print_mat_uint8(img_show2, "rec img_show2", false);
      */
      // imgroi.convertTo(img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 2.0 /
      // 255.0, -1.0);

      /*if(j==0){
        std::cout<<"w:"<<w<<" real_w:"<<real_w<<std::endl;
      }*/
      if (process_type_ == 0) {
        imgroi.convertTo(
            img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 1.0, 0.0);
        float ratio = 2.0 / 255.0;
        img_src0(cv::Rect(0, 0, w, img_h_))
            .convertTo(
                img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, ratio, -1.0);
      } else {
        imgroi.convertTo(
            img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 1.0, 0.0);
        float ratio = 1.0 / 255.0;
        img_src0(cv::Rect(0, 0, w, img_h_))
            .convertTo(
                img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, ratio, 0.0);
      }

      /*if(w == 90){
        std::cout<<i<<std::endl;
        cv::Mat showdata1 = img_src0(cv::Rect(0, 0, w, img_h_)).clone();
        cv::Mat img_show5 = cv::Mat({3, img_h_, w}, CV_32FC1, cv::Scalar(0.0f));
        cv::Mat img_show3 = cv::Mat(img_h_*w, 3, CV_32FC1, showdata1.data);
        cv::Mat img_show4 = cv::Mat(3, img_h_*w, CV_32FC1, img_show5.data);
        cv::transpose(img_show3, img_show4);
        print_mat<float>(img_show5, "rec img_show5");
      }*/
      /*cv::Mat showdata1 = img_src0(cv::Rect(0, 0, w, img_h_)).clone();
      cv::Mat img_show5 = cv::Mat({3, img_h_, w}, CV_32FC1, cv::Scalar(0.0f));
      cv::Mat img_show3 = cv::Mat(img_h_*w, 3, CV_32FC1, showdata1.data);
      cv::Mat img_show4 = cv::Mat(3, img_h_*w, CV_32FC1, img_show5.data);
      cv::transpose(img_show3, img_show4);
      print_mat<float>(img_show5, "rec img_show5");*/

      cv::Mat img_dst = cv::Mat(
          3, img_h_ * real_w, CV_32FC1, batched_imgs.data + (i - s0) * step1);
      cv::Mat img_src = cv::Mat(img_h_ * real_w, 3, CV_32FC1, img_src0.data);
      cv::transpose(img_src, img_dst);
    }
    /*if(j == 2){
      cv::FileStorage file("batched_imgs0.ext", cv::FileStorage::WRITE);
      file << "batched_imgs" << batched_imgs;
      file.release();
    }*/

    // cv::FileStorage
    // file("/home/public/rec_test/rec_"+std::to_string(CNT0)+"_"+std::to_string(j)+".ext",
    // cv::FileStorage::WRITE); file << "imgs" << batched_imgs; file.release();

    batched_imgs_vec.push_back(batched_imgs);
    // print_mat<float>(batched_imgs, "rec batched_imgs");

    OCTensorList inputs = {OCTensor(std::move(batched_imgs))};
    // construct request and exec internal infer
    TRITONSERVER_InferenceRequest* req = nullptr;
    RETURN_IF_ERROR(CreateServerRequestWithTensors(
        context->GetBackendRequestInfo(), graph_executor_->GetServer(),
        sub_graph_names_[0].c_str(), &inputs, sub_graph_io_names_.input_names,
        sub_graph_io_names_.output_names, &req));
    RETURN_IF_ERROR(graph_executor_->AsyncExecute(req, context, &futures[j]));
  }

  // CNT0++;

  std::vector<int> shape = {1, n};
  std::vector<std::string> rec_texts(n, "");
  cv::Mat rec_probs = cv::Mat(shape, CV_32F, cv::Scalar(0.0f));
  for (int k = 0; k < batchs; k++) {
    auto* resp = futures[k].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, sub_graph_io_names_.output_names, &outputs));
    // GraphInferResponseDelete(resp);
    int batch_size = real_batch_list[k];
    cv::Mat feat_ind, feat_prob;
    if (output_matrix_) {
      auto featmap = outputs[0].m().clone();
      float* featmap_data = (float*)featmap.data;
      feat_ind =
          cv::Mat(featmap.size[0], featmap.size[1], CV_32S, cv::Scalar(0));
      feat_prob =
          cv::Mat(featmap.size[0], featmap.size[1], CV_32F, cv::Scalar(0.0f));
      int step0 = featmap.size[1] * featmap.size[2];
      int step1 = featmap.size[2];
      for (int i = 0; i < batch_size; i++) {
        float* data0 = featmap_data + i * step0;
        for (int j = 0; j < featmap.size[1]; j++) {
          float* data1 = data0 + j * step1;
          double min_val, max_val;
          cv::Point2i min_loc;
          cv::Point2i max_loc;
          cv::Mat m = cv::Mat(1, (int)featmap.size[2], CV_32FC1, data1);
          cv::minMaxLoc(m, &min_val, &max_val, &min_loc, &max_loc);
          feat_ind.at<int>(i, j) = max_loc.x;
          feat_prob.at<float>(i, j) = (float)max_val;
        }
      }
    } else {
      feat_ind = outputs[0].m().clone();
      feat_prob = outputs[1].m().clone();
    }

    // print_mat<int>(feat_ind, "feat_ind");
    // print_mat<float>(feat_prob, "feat_prob");

    // print_mat<float>(batched_imgs_vec[k], "batched_imgs");
    for (int64_t j = 0; j < batch_size; j++) {
      float prob = 0.0f;
      int cnt = 0;
      std::string text;
      for (int64_t i = 0; i < feat_ind.size[1]; i++) {
        int ind = feat_ind.at<int>(j, i);
        if (ind == 0 || (i > 0 && ind == feat_ind.at<int>(j, i - 1))) {
          continue;
        }
        text += label_list_[ind];
        prob += feat_prob.at<float>(j, i);
        cnt++;
      }

      rec_texts[k * batch_size_ + j] = text;
      if (cnt == 0) {
        rec_probs.at<float>(0, k * batch_size_ + j) = 0.0;
      } else {
        rec_probs.at<float>(0, k * batch_size_ + j) = prob / cnt;
      }

      // std::cout<<"text:"<<text<<" score:"<<rec_probs.at<float>(0,
      // k*batch_size_+j)<<std::endl;
      /*for(size_t i=0; i<show_prob.size(); i++){
        std::cout<<show_prob[i]<<" ";
      }
      std::cout<<std::endl;*/
    }

    GraphInferResponseDelete(resp);
  }

  if (version_ == 1) {
    context->SetTensor(
        io_names_.output_names,
        {std::move(OCTensor(rec_texts, {1, n})), std::move(rec_probs)});
  } else {
    auto t1 = OCTensor(std::move(rec_probs));
    t1.set_shape({n});

    context->SetTensor(
        io_names_.output_names,
        {std::move(OCTensor(rec_texts, {n})), std::move(t1)});
  }


  return nullptr;
}

TRITONSERVER_Error*
PPRecCh::init(triton::backend::BackendModel* model_state)
{
  PPRec::init(model_state);
  alg_name_ = "rec_ch";
  // if (output_matrix_) {
  //   sub_graph_io_names_ = {{"x"}, {"softmax_5.tmp_0"}};
  // }
  return nullptr;
}

TRITONSERVER_Error*
PPRecLatin::init(triton::backend::BackendModel* model_state)
{
  PPRec::init(model_state);
  alg_name_ = "rec_latin";
  // if (output_matrix_) {
  //   sub_graph_io_names_ = {{"x"}, {"softmax_2.tmp_0"}};
  // }
  return nullptr;
}


}}  // namespace dataelem::alg
