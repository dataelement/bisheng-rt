#include "dataelem/common/mat_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace dataelem::alg;

void ppcls(){
  int max_w_ = 192;
  int min_w_ = 192;
  int img_h_ = 48;
  int batch_size_ = 32;

  std::string src_data = "./test/output/inference_results/ppcls.cvfs";
  cv::FileStorage fs(src_data, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  std::cout<<"num:"<<num<<std::endl;
  for(int k=0; k<num; k++){
    cv::Mat imgs, widths, rots;
    fs["imgs"+std::to_string(k)] >> imgs;
    print_mat<uint8_t>(imgs, "imgs");
    fs["widths"+std::to_string(k)] >> widths;
    print_mat<int>(widths, "widths");
    fs["rots"+std::to_string(k)] >> rots;
    print_mat<uint8_t>(rots, "rots");
    
    cv::Mat processed_imgs = imgs;
    auto processed_imgs_width = widths;
    auto processed_rot = rots;

    int n = processed_imgs.size[0];
    int imgs_max_w = processed_imgs.size[2];
    int batchs = std::ceil(n * 1.0 / batch_size_);
    int step0 = 3 * img_h_ * imgs_max_w;
    for (int j = 0; j < batchs; j++) {
      int s0 = j * batch_size_;
      int e0 = j == (batchs - 1) ? n : (j + 1) * batch_size_;
      int sn0 = e0 - s0;
      int max_w = processed_imgs_width.at<int>(e0 - 1, 0);
      int real_w = std::min(max_w_, std::max(max_w, min_w_));
      std::vector<int> batched_imgs_shape = {sn0, 3, img_h_, real_w};
      std::vector<int> img_shape = {3, img_h_, real_w};
      int step1 = 4 * 3 * img_h_ * real_w;
      cv::Mat batched_imgs =
        cv::Mat(batched_imgs_shape, CV_32FC1, cv::Scalar(0.0f));
      cv::Mat img_src0 = cv::Mat(img_h_, real_w, CV_32FC3, cv::Scalar(0.0f));
      for (int i = s0; i < e0; i++) {
        int w = processed_imgs_width.at<int>(i, 0);
        cv::Mat img_hw =
            cv::Mat(img_h_, imgs_max_w, CV_8UC3, processed_imgs.data + i * step0);
        cv::Mat imgroi = img_hw(cv::Rect(0, 0, w, img_h_));
        
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
    cv::Mat angles = cv::Mat(shape, CV_8U, cv::Scalar(0));
    cv::Mat angles_prob = cv::Mat(shape, CV_32F, cv::Scalar(0.0f));
    for (int j = 0; j < batchs; j++) {
      cv::Mat graph_out;
      fs["graph_out"+std::to_string(k)+"_"+std::to_string(j)] >> graph_out;
      auto feat_map = graph_out;
      for (int64_t i = 0; i < feat_map.size[0]; i++) {
        if (feat_map.at<float>(i, 0) >= feat_map.at<float>(i, 1)) {
          angles.at<uint8_t>(0, j * batch_size_ + i) =
            processed_rot.at<uint8_t>(0, j * batch_size_ + i);
          angles_prob.at<float>(0, j * batch_size_ + i) =
            feat_map.at<float>(i, 0);
        } else {
          angles.at<uint8_t>(0, j * batch_size_ + i) =
            2 + processed_rot.at<uint8_t>(0, j * batch_size_ + i);
          angles_prob.at<float>(0, j * batch_size_ + i) =
            feat_map.at<float>(i, 1);
        }
      }
    }

    cv::Mat angles_py, angles_prob_py;
    fs["cls_ind"+std::to_string(k)] >> angles_py;
    fs["cls_prob"+std::to_string(k)] >> angles_prob_py;
    print_mat<uint8_t>(angles_py, "angles_py");
    print_mat<uint8_t>(angles, "angles");
    print_mat<float>(angles_prob_py, "angles_prob_py");
    print_mat<float>(angles_prob, "angles_prob");
  }

  fs.release();
}

int main(int argc, char** argv){
  ppcls();
  return 0;
}