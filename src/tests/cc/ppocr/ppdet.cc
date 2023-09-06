#include "ext/ppocr/postprocess_op.h"
#include "dataelem/common/mat_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace dataelem::alg;

void det_postprocess(){
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.6;
  double det_db_unclip_ratio_ = 1.5;
  std::string det_db_score_mode_ = "fast";
  bool use_dilation_ = false;
  PaddleOCR::PostProcessor post_processor_;

  std::string src_data = "./test/output/inference_results/ppdet.cvfs";
  cv::FileStorage fs(src_data, cv::FileStorage::READ);

  int num = 0;
  fs["num"] >> num;
  for(int i=0; i<num; i++){
    cv::Mat featmap;
    fs["graphout"+std::to_string(i)] >> featmap;
    cv::Mat shape_list;
    fs["shape_list"+std::to_string(i)] >> shape_list;
    cv::Mat bboxes_py;
    fs["bboxes"+std::to_string(i)] >> bboxes_py;

    print_mat<int>(bboxes_py, "bboxes_py");

    int ori_h = (int)shape_list.at<float>(0, 0);
    int ori_w = (int)shape_list.at<float>(0, 1);
    float ratio_h = shape_list.at<float>(0, 2);
    float ratio_w = shape_list.at<float>(0, 3);

    std::vector<int> output_shape;
    for (int i = 0; i < featmap.size.dims(); i++) {
      output_shape.emplace_back(featmap.size.p[i]);
    }

    int n2 = output_shape[2];
    int n3 = output_shape[3];
    int n = n2 * n3;

    float* out_data = reinterpret_cast<float*>(featmap.data);

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
    
    std::vector<float> scores;
    boxes = post_processor_.BoxesFromBitmap(
        pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
        det_db_score_mode_, scores, ori_w, ori_h);
    
    std::vector<float> filter_scores;
    boxes = post_processor_.FilterTagDetRes(
        boxes, ratio_h, ratio_w, ori_h, ori_w, scores, filter_scores);

    int bb_cnt = boxes.size();
    std::vector<int> bbox_shape = {bb_cnt, 4, 2};
    std::vector<int> score_shape = {bb_cnt};
    cv::Mat bbox_cc = cv::Mat(bbox_shape, CV_32S);
    auto* ptr = reinterpret_cast<int*>(bbox_cc.data);
    for (int i = 0; i < bb_cnt; i++) {
        for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 2; k++) {
            *(ptr + 8 * i + j * 2 + k) = boxes[i][j][k];
        }
        }
    }

    cv::Mat bbox_prob;
    if (bb_cnt == 0) {
        bbox_prob = cv::Mat(score_shape, CV_32F);
    } else {
        bbox_prob = cv::Mat(score_shape, CV_32F, filter_scores.data()).clone();
    }

    print_mat<int>(bbox_cc, "bbox_cc");
    print_mat<float>(bbox_prob, "bbox_prob");
  }
  
  fs.release();

}

int main(int argc, char** argv){
  det_postprocess();
  return 0;
}