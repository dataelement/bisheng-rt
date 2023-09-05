#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace dataelem::alg;

void
mask_to_bb(
    const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
    const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& orig,
    const float& scale, const bool& enable_huarong_box_adjust,
    std::vector<Point2fList>& points_vec, std::vector<float>& scores_vec)
{
  // calculate rrect for each text mask area
  bool has_angle = (!boxes_cos.empty() && !boxes_sin.empty()) ? true : false;
  // updated: orig is replace with orig_shape, 2022.10.20
  //  scale is replace with scale_shape

  // int orig_h = orig.size[0], orig_w = orig.size[1];
  int orig_h = orig.at<int>(0, 0);
  int orig_w = orig.at<int>(1, 0);

  cv::Mat bbs = bboxes / scale;

  clip_boxes(bbs, orig_h, orig_w);
  int c0 = masks.dims == 2 ? masks.channels() : masks.size[2];
  int mask_step0 = masks.size[1] * c0;
  int bbs_cnt = bbs.size[0];

  // std::vector<BoolFuture> rets(bbs_cnt);
  std::vector<Point2fList> points_list(bbs_cnt);
  std::vector<std::array<float, 2>> attrs_list(bbs_cnt);
  std::vector<float> score_list(bbs_cnt, -1.0);
  for (int i = 0; i < bbs_cnt; ++i) {
    // rets[i] = _pool->enqueue(
    //[this, i, &mask_step0, &orig_w, &orig_h, &has_angle, &masks, &bbs,
    [i, &mask_step0, &orig_w, &orig_h, &has_angle, &masks, &bbs, &scores,
     &boxes_cos, &boxes_sin, &points_list, &score_list, &attrs_list]() -> bool {
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
      int c = masks.dims == 2 ? masks.channels() : masks.size[2];
      cv::Mat mask(masks.size[1], c, CV_32F, masks.data + mask_step0 * 4 * i);
      resizeOp(mask, mask1, w, h);
      cv::Mat(mask1 > 0.5).convertTo(mask2, CV_8U);
      mask2.copyTo(full_mask(cv::Rect(x0, y0, w, h)));
      Contours contours;
      findContoursOp(full_mask, contours);
      if (contours.size() == 0) {
        return false;
      }
      int max_area = 0;
      int max_index = 0;
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
        cos = boxes_cos.at<float>(i, 0);
        sin = boxes_sin.at<float>(i, 0);
        std::cout << cos << " " << sin << std::endl;
      }

      std::array<float, 2> attrs = {cos, sin};
      points_list[i] = std::move(pts2);
      score_list[i] = scores.at<float>(i, 0);
      attrs_list[i] = std::move(attrs);
      return true;
    }();
    //});
  }
  // GetAsyncRets(rets);

  std::vector<Point2fList> points_list2;
  std::vector<std::array<float, 2>> attrs_list2;
  std::vector<float> score_list2;
  for (int i = 0; i < bbs_cnt; i++) {
    if (points_list[i].size() > 0) {
      points_list2.emplace_back(std::move(points_list[i]));
      attrs_list2.emplace_back(std::move(attrs_list[i]));
      score_list2.emplace_back(score_list[i]);
    }
  }
  if (points_list2.size() == 0) {
    return;
  }

  float _nms_threshold = 0.2f;
  auto keep = lanms::merge_quadrangle_standard_parallel(
      points_list2, score_list2, score_list2.size(), _nms_threshold);

  // reorder point by text direction, first point is the left-top of text line
  for (const auto& j : keep) {
    auto& points = points_list2[j];
    float score = score_list2[j];
    if (has_angle) {
      float cos = attrs_list2[j].at(0), sin = attrs_list2[j].at(1);
      reorder_start_point(points, cos, sin);
    } else {
      auto idxs = reorder_quadrangle_points(points);
      Point2fList points_tmp;
      for (size_t i = 0; i < idxs.size(); i++) {
        points_tmp.emplace_back(points[idxs[i]]);
      }
      points = points_tmp;
    }
    points_vec.emplace_back(std::move(points));
    scores_vec.emplace_back(score);
  }

  // added by hanfeng at 2020.05.20, do box adjust logic from huarong
  //  https://gitlab.4pd.io/cvxy4pd/cvbe/nn-predictor/issues/56
  if (has_angle && enable_huarong_box_adjust) {
    refine_box_orientation(points_vec);
  }
}

int
main(int argc, char** argv)
{
  std::string data_dir = "/home/public/data/test_data/";
  std::string read_name = "mrcnn_v5.1_graph.cvfs";
  std::string write_name = "mrcnn_v5.1_post_cc.cvfs";
  cv::FileStorage fs_rd(data_dir + read_name, cv::FileStorage::READ);
  cv::FileStorage fs_wr(data_dir + write_name, cv::FileStorage::WRITE);
  int num = 0;
  fs_rd["num"] >> num;
  fs_wr << "num" << num;
  std::cout << "num:" << num << std::endl;
  for (int k = 0; k < num; k++) {
    cv::Mat scores, bboxes, masks, boxes_cos, boxes_sin, orig;
    fs_rd["pre_boxes" + std::to_string(k)] >> bboxes;
    fs_rd["pre_scores" + std::to_string(k)] >> scores;
    fs_rd["pre_masks" + std::to_string(k)] >> masks;
    fs_rd["pre_boxes_cos" + std::to_string(k)] >> boxes_cos;
    fs_rd["pre_boxes_sin" + std::to_string(k)] >> boxes_sin;
    fs_rd["orig_shape" + std::to_string(k)] >> orig;
    float scale;
    fs_rd["scale" + std::to_string(k)] >> scale;
    std::string name;
    fs_rd["image_name" + std::to_string(k)] >> name;
    print_mat<float>(bboxes, "bboxes");
    print_mat<float>(scores, "scores");
    print_mat<float>(masks, "masks");
    print_mat<float>(boxes_cos, "boxes_cos");
    print_mat<float>(boxes_sin, "boxes_sin");
    print_mat<int>(orig, "orig");
    std::cout << "scale:" << scale << std::endl;
    std::cout << "name:" << name << std::endl;

    std::vector<std::vector<cv::Point2f>> points_vec;
    std::vector<float> scores_vec;
    bool enable_huarong_box_adjust = true;
    mask_to_bb(
        scores, masks, bboxes, boxes_cos, boxes_sin, orig, scale,
        enable_huarong_box_adjust, points_vec, scores_vec);
    int n = points_vec.size();
    for (int i = 0; i < n; i++) {
      std::cout << "i"
                << ":" << i << " " << points_vec[i][0].x << " "
                << points_vec[i][0].y << " " << points_vec[i][1].x << " "
                << points_vec[i][1].y << " ";
      std::cout << points_vec[i][2].x << " " << points_vec[i][2].y << " "
                << points_vec[i][3].x << " " << points_vec[i][3].y << std::endl;
    }
  }
  fs_rd.release();
  fs_wr.release();
  return 0;
}