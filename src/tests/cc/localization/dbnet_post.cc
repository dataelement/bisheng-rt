#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "ext/ppocr/postprocess_op.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace dataelem::alg;

double det_db_thresh_ = 0.3;
double det_db_box_thresh_ = 0.6;
double det_db_unclip_ratio_ = 1.5;
std::string det_db_score_mode_ = "fast";
bool use_dilation_ = false;
void
postprocess(
    PaddleOCR::PostProcessor ppocr_post, const cv::Mat& feamap,
    const cv::Mat& shape_list, cv::Mat& out_bboxes, cv::Mat& out_scores)
{
  int ori_h = (int)shape_list.at<float>(0, 0);
  int ori_w = (int)shape_list.at<float>(0, 1);
  float ratio_h = shape_list.at<float>(0, 2);
  float ratio_w = shape_list.at<float>(0, 3);
  auto output_shape = feamap.size;
  int n2 = (int)output_shape[2];
  int n3 = (int)output_shape[3];
  int n = n2 * n3;

  float* out_data = reinterpret_cast<float*>(feamap.data);

  std::vector<float> pred(n, 0.0);
  std::vector<uint8_t> mask(n, 0);

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    if (pred[i] > det_db_thresh_) {
      mask[i] = 255;
    } else {
      mask[i] = 0;
    }
  }

  cv::Mat bit_map(n2, n3, CV_8UC1, mask.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float*)pred.data());

  std::vector<std::vector<std::vector<int>>> boxes;

  // print_mat<uint8_t>(bit_map, "bit_map");
  // print_mat<float>(pred_map, "pred_map");

  // boxes: (n, 4, 2)
  std::vector<float> scores;
  boxes = ppocr_post.BoxesFromBitmap(
      pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
      det_db_score_mode_, scores, ori_w, ori_h);

  std::vector<float> filter_scores;
  boxes = ppocr_post.FilterTagDetRes(
      boxes, ratio_h, ratio_w, ori_h, ori_w, scores, filter_scores);

  // std::cout<<"bboxes:"<<std::endl;
  // for(size_t i=0; i<boxes.size(); i++){
  //   std::cout<<boxes[i][0][0]<<" "<<boxes[i][0][1]<<" ";
  //   std::cout<<boxes[i][1][0]<<" "<<boxes[i][1][1]<<" ";
  //   std::cout<<boxes[i][2][0]<<" "<<boxes[i][2][1]<<" ";
  //   std::cout<<boxes[i][3][0]<<" "<<boxes[i][3][1]<<std::endl;
  // }

  int bb_cnt = boxes.size();
  std::vector<int> bbox_shape = {bb_cnt, 4, 2};
  std::vector<int> score_shape = {1, bb_cnt};
  out_bboxes = cv::Mat(bbox_shape, CV_32S);
  auto* ptr = reinterpret_cast<int*>(out_bboxes.data);
  for (int i = 0; i < bb_cnt; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 2; k++) {
        *(ptr + 8 * i + j * 2 + k) = boxes[i][j][k];
      }
    }
  }

  if (bb_cnt == 0) {
    out_scores = cv::Mat(score_shape, CV_32F);
  } else {
    out_scores = cv::Mat(score_shape, CV_32F, filter_scores.data()).clone();
  }
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

int
main(int argc, char** argv)
{
  // std::string data_dir = "/home/public/data/test_data/";
  // std::string read_name = "ppdet_graph.cvfs";
  // std::string write_name = "ppdet_post_cc.cvfs";

  std::string data_dir = argv[1];
  std::string read_name = argv[2];
  std::string write_name = argv[3];
  cv::FileStorage fs_rd(data_dir + read_name, cv::FileStorage::READ);
  cv::FileStorage fs_wr(data_dir + write_name, cv::FileStorage::WRITE);
  int num = 0;
  fs_rd["num"] >> num;
  fs_wr << "num" << num;
  std::cout << "num:" << num << std::endl;

  PaddleOCR::PostProcessor ppocr_post;
  for (int k = 0; k < num; k++) {
    std::string name;
    fs_rd["image_name" + std::to_string(k)] >> name;
    fs_wr << "image_name" + std::to_string(k) << name;
    cv::Mat featmap, shape_list, orig_shape, gt;
    fs_rd["featmap" + std::to_string(k)] >> featmap;
    fs_rd["shape_list" + std::to_string(k)] >> shape_list;
    fs_rd["orig_shape" + std::to_string(k)] >> orig_shape;
    fs_rd["gt" + std::to_string(k)] >> gt;
    print_mat<float>(shape_list, "shape_list");
    print_mat<int>(orig_shape, "orig_shape");
    std::cout << "name:" << name << std::endl;
    print_mat<float>(featmap, "featmap");

    int featH = featmap.size[2];
    int featW = featmap.size[3];
    int realH = (int)(shape_list.at<float>(0, 0) * shape_list.at<float>(0, 2));
    int realW = (int)(shape_list.at<float>(0, 1) * shape_list.at<float>(0, 3));
    cv::Mat new_featmap;
    if (realH < featH || realW < featW) {
      new_featmap = getroi(featmap, realH, realW);
    } else {
      new_featmap = featmap;
    }

    print_mat<float>(new_featmap, "new_featmap");

    cv::Mat out_bboxes;
    cv::Mat out_scores;
    postprocess(ppocr_post, new_featmap, shape_list, out_bboxes, out_scores);
    int n = out_bboxes.size[0];
    fs_wr << "num_bbox" + std::to_string(k) << n;
    std::cout << "num_bbox:" << n << std::endl;
    for (int i = 0; i < n; i++) {
      std::cout << "i"
                << ":";
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
          std::cout << out_bboxes.at<int>(i, j, k) << " ";
        }
      }
      std::cout << out_scores.at<float>(0, i) << std::endl;
    }

    fs_wr << "gt" + std::to_string(k) << gt;
    fs_wr << "bboxes" + std::to_string(k) << out_bboxes;
    fs_wr << "scores" + std::to_string(k) << out_scores;
  }
  fs_rd.release();
  fs_wr.release();
  return 0;
}