// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Suppress warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#include "ext/ppocr/postprocess_op.h"

#include "ext/clipper/clipper.hpp"

namespace PaddleOCR {

void
PostProcessor::GetContourArea(
    const std::vector<std::vector<float>>& box, float unclip_ratio,
    double& distance)
{
  int pts_num = 4;
  double area = 0.0;
  double dist = 0.0;
  for (int i = 0; i < pts_num; i++) {
    area += (double)box[i][0] * (double)box[(i + 1) % pts_num][1] -
            (double)box[i][1] * (double)box[(i + 1) % pts_num][0];
    dist += sqrt(
        ((double)box[i][0] - (double)box[(i + 1) % pts_num][0]) *
            ((double)box[i][0] - (double)box[(i + 1) % pts_num][0]) +
        ((double)box[i][1] - (double)box[(i + 1) % pts_num][1]) *
            ((double)box[i][1] - (double)box[(i + 1) % pts_num][1]));
  }
  area = fabs(area / 2.0);

  distance = area * unclip_ratio / dist;
  // std::cout<<"distance:"<<distance<<" area:"<<area<<"
  // dist:"<<dist<<std::endl;
}

cv::RotatedRect
PostProcessor::UnClip(
    std::vector<std::vector<float>> box, const float& unclip_ratio)
{
  double distance = 1.0;

  for (size_t j = 0; j < box.size(); j++) {
    for (size_t i = 0; i < box[j].size(); i++) {
      box[j][i] = std::nearbyint(box[j][i]);
    }
  }

  /*std::cout<<"bbox:";
  std::cout<<box[0][0]<<" "<<box[0][1]<<" ";
  std::cout<<box[1][0]<<" "<<box[1][1]<<" ";
  std::cout<<box[2][0]<<" "<<box[2][1]<<" ";
  std::cout<<box[3][0]<<" "<<box[3][1]<<std::endl;*/

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
    << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
    << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
    << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;

  // std::cout<<"points:";
  for (int j = 0; j < soln.size(); j++) {
    for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
      // std::cout<<soln[j][i].X<<" "<<soln[j][i].Y<<" ";
    }
  }

  // std::cout<<std::endl;

  cv::RotatedRect res;
  if (points.size() <= 0) {
    res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
  } else {
    res = cv::minAreaRect(points);
  }
  return res;
}

float**
PostProcessor::Mat2Vec(cv::Mat mat)
{
  auto** array = new float*[mat.rows];
  for (int i = 0; i < mat.rows; ++i) array[i] = new float[mat.cols];
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

std::vector<std::vector<int>>
PostProcessor::OrderPointsClockwise(std::vector<std::vector<int>> pts)
{
  std::vector<std::vector<int>> box = pts;
  std::stable_sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {
      leftmost[0], rightmost[0], rightmost[1], leftmost[1]};
  return rect;
}

std::vector<std::vector<float>>
PostProcessor::Mat2Vector(cv::Mat mat)
{
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

bool
PostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b)
{
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

bool
PostProcessor::XsortInt(std::vector<int> a, std::vector<int> b)
{
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

std::vector<std::vector<float>>
PostProcessor::GetMiniBoxes(cv::RotatedRect box, float& ssid)
{
  ssid = std::min(box.size.width, box.size.height);
  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::stable_sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

float
PostProcessor::PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred)
{
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (int i = 0; i < contour.size(); ++i) {
    box_x.push_back(contour[i].x);
    box_y.push_back(contour[i].y);
  }

  int xmin = clamp(
      int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
      width - 1);
  int xmax = clamp(
      int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
      width - 1);
  int ymin = clamp(
      int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
      height - 1);
  int ymax = clamp(
      int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
      height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point* rook_point = new cv::Point[contour.size()];

  for (int i = 0; i < contour.size(); ++i) {
    rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
  }
  const cv::Point* ppt[1] = {rook_point};
  int npt[] = {int(contour.size())};

  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  float score = cv::mean(croppedImg, mask)[0];

  delete[] rook_point;
  return score;
}

float
PostProcessor::BoxScoreFast(
    std::vector<std::vector<float>> box_array, cv::Mat pred)
{
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(
      int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0, width - 1);
  int xmax = clamp(
      int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0, width - 1);
  int ymin = clamp(
      int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0, height - 1);
  int ymax = clamp(
      int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0, height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point* ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

std::vector<std::vector<std::vector<int>>>
PostProcessor::BoxesFromBitmap(
    const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh,
    const float& det_db_unclip_ratio, const std::string& det_db_score_mode,
    std::vector<float>& scores, int dest_width, int dest_height)
{
  const int min_size = 3;
  const int max_candidates = 1000;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(
      bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::vector<int>>> boxes;

  for (int _i = 0; _i < num_contours; _i++) {
    /*if (contours[_i].size() <= 2) {
      continue;
    }*/
    float ssid;
    /*int sumc = 0;
    int cnt = 0;
    for(size_t k=0; k<contours[_i].size(); k++){
      std::cout<<contours[_i][k].x<<" "<<contours[_i][k].y<<" ";
      sumc += contours[_i][k].x;
      sumc += contours[_i][k].y;
      cnt += 1;
    }
    std::cout<<"sum:"<<sumc<<" cnt:"<<cnt<<std::endl;*/

    // std::cout<<std::endl;
    cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    // std::cout<<"box:";
    // std::cout<<box.center.x<<" "<<box.center.y<<" "<<box.size.width<<"
    // "<<box.size.height<<" "<<box.angle<<std::endl;
    auto array = GetMiniBoxes(box, ssid);

    /*std::cout<<"ssid:"<<ssid<<std::endl;
    std::cout<<array[0][0]<<" "<<array[0][1]<<" ";
    std::cout<<array[1][0]<<" "<<array[1][1]<<" ";
    std::cout<<array[2][0]<<" "<<array[2][1]<<" ";
    std::cout<<array[3][0]<<" "<<array[3][1]<<std::endl;*/

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    if (det_db_score_mode == "slow")
      /* compute using polygon*/
      score = PolygonScoreAcc(contours[_i], pred);
    else
      score = BoxScoreFast(array, pred);

    if (score < box_thresh)
      continue;

    // start for unclip
    cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);

    /*if (points.size.height < 1.001 && points.size.width < 1.001) {
      continue;
    }*/
    // end for unclip

    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);
    /*std::cout<<"ssid:"<<ssid<<std::endl;
    std::cout<<"cliparray:";
    std::cout<<cliparray[0][0]<<" "<<cliparray[0][1]<<" ";
    std::cout<<cliparray[1][0]<<" "<<cliparray[1][1]<<" ";
    std::cout<<cliparray[2][0]<<" "<<cliparray[2][1]<<" ";
    std::cout<<cliparray[3][0]<<" "<<cliparray[3][1]<<std::endl;
    */

    if (ssid < min_size + 2)
      continue;

    // int dest_width = pred.cols;
    // int dest_height = pred.rows;
    // std::cout<<"bbox:";
    // std::cout<<"width:"<<width<<" dest_width:"<<dest_width<<"
    // height:"<<height<<" dest_height:"<<dest_height<<std::endl;
    std::vector<std::vector<int>> intcliparray;
    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{
          int(clampf(
              nearbyint(
                  float(dest_width) * cliparray[num_pt][0] / float(width)),
              0, float(dest_width))),
          int(clampf(
              nearbyint(
                  float(dest_height) * cliparray[num_pt][1] / float(height)),
              0, float(dest_height)))};
      intcliparray.push_back(a);
      // std::cout<<a[0]<<" "<<a[1]<<" ";
    }
    // std::cout<<std::endl;
    boxes.push_back(intcliparray);
    scores.emplace_back(score);

  }  // end for
  return boxes;
}

std::vector<std::vector<std::vector<int>>>
PostProcessor::FilterTagDetRes(
    std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
    float ratio_w, int oriimg_h, int oriimg_w, const std::vector<float>& scores,
    std::vector<float>& filter_scores)
{
  // int oriimg_h = srcimg.rows;
  // int oriimg_w = srcimg.cols;

  std::vector<std::vector<std::vector<int>>> root_points;
  for (int n = 0; n < boxes.size(); n++) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (int m = 0; m < boxes[0].size(); m++) {
      // boxes[n][m][0] /= ratio_w;
      // boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width = int(sqrt(
        pow(boxes[n][0][0] - boxes[n][1][0], 2) +
        pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = int(sqrt(
        pow(boxes[n][0][0] - boxes[n][3][0], 2) +
        pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 3 || rect_height <= 3)
      continue;
    root_points.push_back(boxes[n]);
    filter_scores.emplace_back(scores[n]);
  }
  return root_points;
}

}  // namespace PaddleOCR

#pragma GCC diagnostic pop