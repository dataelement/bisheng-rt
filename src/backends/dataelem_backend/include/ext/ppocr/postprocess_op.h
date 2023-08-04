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

#pragma once

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "ext/clipper/clipper.hpp"
#include "ext/ppocr/utility.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

namespace PaddleOCR {

class PostProcessor {
 public:
  void GetContourArea(
      const std::vector<std::vector<float>>& box, float unclip_ratio,
      double& distance);

  cv::RotatedRect UnClip(
      std::vector<std::vector<float>> box, const float& unclip_ratio);

  float** Mat2Vec(cv::Mat mat);

  std::vector<std::vector<int>> OrderPointsClockwise(
      std::vector<std::vector<int>> pts);

  std::vector<std::vector<float>> GetMiniBoxes(
      cv::RotatedRect box, float& ssid);

  float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
  float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

  std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
      const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh,
      const float& det_db_unclip_ratio, const std::string& det_db_score_mode,
      std::vector<float>& scores, int dest_width, int dest_height);

  std::vector<std::vector<std::vector<int>>> FilterTagDetRes(
      std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
      float ratio_w, int oriimg_h, int oriimg_w,
      const std::vector<float>& scores, std::vector<float>& filter_scores);

 private:
  static bool XsortInt(std::vector<int> a, std::vector<int> b);

  static bool XsortFp32(std::vector<float> a, std::vector<float> b);

  std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

  inline int _max(int a, int b) { return a >= b ? a : b; }

  inline int _min(int a, int b) { return a >= b ? b : a; }

  template <class T>
  inline T clamp(T x, T min, T max)
  {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline float clampf(float x, float min, float max)
  {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
};

}  // namespace PaddleOCR
