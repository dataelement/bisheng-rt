#include "dataelem/common/mat_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace dataelem::alg;

std::vector<std::string>
ReadDict(const std::string& path)
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

void pprec(){
  int max_w_ = 1200;
  int min_w_ = 320;
  int img_h_ = 48;
  int batch_size_ = 32;
  float thresh_ = 0.9;

  std::vector<std::string> label_list_;
  std::string label_path = "./test/data/ppocr_keys_v1.txt";
  label_list_ = ReadDict(label_path);
  label_list_.insert(label_list_.begin(), "#");
  label_list_.push_back(" ");

  std::string src_data = "./test/output/inference_results/pprec.cvfs";
  cv::FileStorage fs(src_data, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  std::cout<<"num:"<<num<<std::endl;
  for(int k=0; k<num; k++){
    cv::Mat processed_imgs, processed_imgs_width, angles, angles_prob;
    fs["imgs"+std::to_string(k)] >> processed_imgs;
    print_mat<uint8_t>(processed_imgs, "processed_imgs");
    fs["widths"+std::to_string(k)] >> processed_imgs_width;
    print_mat<int>(processed_imgs_width, "processed_imgs_width");
    fs["cls_ind"+std::to_string(k)] >> angles;
    print_mat<uint8_t>(angles, "angles");
    fs["cls_prob"+std::to_string(k)] >> angles_prob;
    print_mat<float>(angles, "angles_prob");
    
    int n = processed_imgs.size[0];
    int imgs_max_w = processed_imgs.size[2];
    int batchs = std::ceil(n * 1.0 / batch_size_);
    int step0 = 3 * img_h_ * imgs_max_w;
    for (int j = 0; j < batchs; j++) {
      int s0 = j * batch_size_;
      int e0 = j == (batchs - 1) ? n : (j + 1) * batch_size_;
      int sn0 = e0 - s0;
      int max_w = processed_imgs_width.at<int>(0, e0 - 1);
      int real_w = std::min(max_w_, std::max(max_w, min_w_));
      std::vector<int> batched_imgs_shape = {sn0, 3, img_h_, real_w};
      std::vector<int> img_shape = {3, img_h_, real_w};
      int step1 = 4 * 3 * img_h_ * real_w;
      cv::Mat batched_imgs =
        cv::Mat(batched_imgs_shape, CV_32FC1, cv::Scalar(0.0f));
      cv::Mat img_src0 = cv::Mat(img_h_, real_w, CV_32FC3, cv::Scalar(0.0f));
      for (int i = s0; i < e0; i++) {
        int w = processed_imgs_width.at<int>(0, i);
        cv::Mat img_hw =
            cv::Mat(img_h_, imgs_max_w, CV_8UC3, processed_imgs.data + i * step0);
        cv::Mat imgroi = img_hw(cv::Rect(0, 0, w, img_h_));

        auto angle = (int)angles.at<uint8_t>(0, i);
        auto angle_prob = angles_prob.at<float>(0, i);
        if (angle >= 2 && angle_prob >= thresh_) {
          cv::rotate(imgroi, imgroi, cv::ROTATE_180);
        }

        if (w > max_w_) {
          cv::resize(imgroi, imgroi, {real_w, img_h_}, 0, 0, cv::INTER_LINEAR);
          w = real_w;
        }

        imgroi.convertTo(img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, 1.0, 0.0);
        float ratio = 2.0 / 255.0;
        img_src0(cv::Rect(0, 0, w, img_h_))
            .convertTo(
                img_src0(cv::Rect(0, 0, w, img_h_)), CV_32FC3, ratio, -1.0);

        cv::Mat img_dst = cv::Mat(
            3, img_h_ * real_w, CV_32FC1, batched_imgs.data + (i - s0) * step1);
        cv::Mat img_src = cv::Mat(img_h_ * real_w, 3, CV_32FC1, img_src0.data);
        cv::transpose(img_src, img_dst);
      }
      cv::Mat graph_in;
      fs["graph_in"+std::to_string(k)+"_"+std::to_string(j)] >> graph_in;

      print_mat<float>(graph_in, "graph_in");
      print_mat<float>(batched_imgs, "batched_imgs");
    }

    std::vector<int> shape = {1, n};
    std::vector<std::string> rec_texts(n, "");
    cv::Mat rec_probs = cv::Mat(shape, CV_32F, cv::Scalar(0.0f));
    for (int k0 = 0; k0 < batchs; k0++) {
      cv::Mat graph_out;
      fs["graph_out"+std::to_string(k)+"_"+std::to_string(k0)] >> graph_out;

      auto featmap = graph_out;
      float* featmap_data = (float*)featmap.data;
      cv::Mat feat_ind = cv::Mat(featmap.size[0], featmap.size[1], CV_32S, cv::Scalar(0));
      cv::Mat feat_prob = cv::Mat(featmap.size[0], featmap.size[1], CV_32F, cv::Scalar(0.0f));
      int step0 = featmap.size[1] * featmap.size[2];
      int step1 = featmap.size[2];
      for(int i=0; i<featmap.size[0]; i++){
        float* data0 = featmap_data + i * step0;
        for(int j=0; j<featmap.size[1]; j++){
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

      for (int64_t j = 0; j < feat_ind.size[0]; j++) {
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
      }
    }

    std::string rec_text_py;
    fs["rec_text"+std::to_string(k)] >> rec_text_py;
    std::cout<<"rec_text_py:"<<rec_text_py<<std::endl;
    std::cout<<"rec_text   :";
    for(int i=0; i<rec_texts.size(); i++){
      std::cout<<rec_texts[i];
      if(i<rec_texts.size()-1){
        std::cout<<"###";
      }
    }
    std::cout<<std::endl;

    cv::Mat text_probs_py;
    fs["rec_prob"+std::to_string(k)] >> text_probs_py;
    print_mat<float>(text_probs_py, "text_probs_py");
    print_mat<float>(rec_probs, "rec_probs");
  }

  fs.release();
}

int main(int argc, char** argv){
  pprec();
  return 0;
}