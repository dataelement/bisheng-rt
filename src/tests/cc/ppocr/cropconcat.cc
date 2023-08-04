#include "dataelem/common/mat_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace dataelem::alg;

void cropconcat(){
  int max_w_ = 1200;
  int img_h_ = 48;
  float hw_thrd_ = 1.5;
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;

  std::string src_data = "./test/output/inference_results/cropconcat.cvfs";
  cv::FileStorage fs(src_data, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  std::cout<<"num:"<<num<<std::endl;
  for(int k=0; k<num; k++){
    cv::Mat ori_img, bboxes;
    fs["input"+std::to_string(k)] >> ori_img;
    fs["bboxes"+std::to_string(k)] >> bboxes;

    cv::Mat imgs, widths, rots;
    fs["imgs"+std::to_string(k)] >> imgs;
    print_mat<uint8_t>(imgs, "imgs");
    fs["widths"+std::to_string(k)] >> widths;
    print_mat<int>(widths, "widths");
    fs["rots"+std::to_string(k)] >> rots;
    print_mat<uint8_t>(rots, "rots");
    
    int n = bboxes.size[0];
    cv::Mat bbmf;
    cv::Mat_<cv::Point2f> bbs;
    if (n > 0) {
      bboxes.convertTo(bbmf, CV_32FC1);
      bbs = cv::Mat_<cv::Point2f>(bbmf);
    }
    
    std::vector<std::pair<int, int>> temp_widths;
    std::vector<cv::Mat> img_list;
    std::vector<uint8_t> rot_list;
    for (int i = 0; i < n; i++) {
      std::vector<cv::Point2f> v = {
        bbs(i, 0), bbs(i, 1), bbs(i, 2), bbs(i, 3)};
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

    if (n > 0) {
      std::stable_sort(
        temp_widths.begin(), temp_widths.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
          return p1.second < p2.second;
        });
    }

    int max_w = n > 0 ? std::min(max_w_, temp_widths[n - 1].second) : 0;
    std::vector<int> bboxes_shape = {1, n, 4, 2};
    std::vector<int> shape0 = {1, n};
    std::vector<int> processed_imgs_shape = {1, n, img_h_, max_w, 3};
    cv::Mat processed_rot = cv::Mat(shape0, CV_8U, cv::Scalar(0));
    cv::Mat processed_imgs_width =
        cv::Mat(shape0, CV_32S, cv::Scalar(0));
    cv::Mat processed_imgs = cv::Mat(processed_imgs_shape, CV_8U, cv::Scalar(0));
    int step = 3 * img_h_ * max_w;
    for (int i = 0; i < n; i++) {
      int m_index = temp_widths[i].first;
      int w = std::min(max_w, temp_widths[i].second);
      
      processed_rot.at<uint8_t>(0, i) = rot_list[m_index];
      processed_imgs_width.at<int>(0, i) = w;
      std::vector<cv::Point2f> src_4points = {
        bbs(m_index, 0), bbs(m_index, 1), bbs(m_index, 2),
        bbs(m_index, 3)};
      float raw_w = floor(std::max(
        l2_norm(src_4points[0], src_4points[1]),
        l2_norm(src_4points[2], src_4points[3])));
      float raw_h = floor(std::max(
        l2_norm(src_4points[1], src_4points[2]),
        l2_norm(src_4points[0], src_4points[3])));

      std::vector<cv::Point2f> dst_4points = {
        {0, 0}, {raw_w, 0}, {raw_w, raw_h}, {0, raw_h}};

      cv::Mat warp_mat = cv::getPerspectiveTransform(src_4points, dst_4points);
      cv::Mat img =
        cv::Mat(img_h_, max_w, CV_8UC3, processed_imgs.data + i * step);
      cv::Mat m;
      cv::warpPerspective(
        ori_img, m, warp_mat, cv::Size((int)raw_w, (int)raw_h), cv::INTER_CUBIC,
        cv::BORDER_REPLICATE);
      if (rot_list[m_index] == 1) {
        cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
      }

      cv::resize(m, m, {w, img_h_}, 0, 0, cv::INTER_LINEAR);
      m.copyTo(img(cv::Rect(0, 0, w, img_h_)));
    }

    print_mat<uint8_t>(processed_imgs, "processed_imgs");
    print_mat<int>(processed_imgs_width, "processed_imgs_width");
    print_mat<uint8_t>(processed_rot, "processed_rot");
  }
  fs.release();
}

int main(int argc, char** argv){
  cropconcat();
  return 0;
}