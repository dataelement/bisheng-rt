#include <mutex>
#include "absl/strings/escaping.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "dataelem/common/mat_utils.h"

namespace dataelem { namespace alg {

// using namespace std;

CV_MATCH_TYPE_AND_ENUM(uchar, CV_8U);
CV_MATCH_TYPE_AND_ENUM(schar, CV_8S);
CV_MATCH_TYPE_AND_ENUM(int, CV_32S);
CV_MATCH_TYPE_AND_ENUM(float, CV_32F);

CV_MATCH_TYPE_AND_ENUM(long long int, -1);
CV_MATCH_TYPE_AND_ENUM(unsigned long long, -2);
CV_MATCH_TYPE_AND_ENUM(unsigned int, -3);

void
imNormOp(const cv::Mat& src, cv::Mat& dest, cv::Vec3f mean, cv::Vec3f std)
{
  cv::Mat temp1;
  auto f_cond = [](float v) { return abs(v - 1) > 1e-6 && abs(v) > 1e-6; };

  if (src.channels() == 1) {
    cv::subtract(src, mean[0], temp1, cv::noArray(), CV_32FC1);
    dest = f_cond(std[0]) ? temp1 / std[0] : temp1;
  } else {
    std::vector<cv::Mat> planes;
    cv::Mat temp1;
    if (src.type() == CV_32FC1) {
      temp1 = src.clone();
    } else {
      src.convertTo(temp1, CV_32FC1);
    }
    cv::split(temp1, planes);
    for (int i = 0; i < 3; i++) {
      planes[i] = f_cond(std[i]) ? (planes[i] - mean[i]) / std[i]
                                 : (planes[i] - mean[i]);
    }
    cv::merge(planes, dest);
  }
}

// rotate image
void
rotateOp(const cv::Mat& src, cv::Mat& dest, float angle)
{
  cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  double sinv = abs(rot.at<double>(0, 1));
  double cosv = abs(rot.at<double>(0, 0));
  int bound_w = int(src.rows * sinv + src.cols * cosv);
  int bound_h = int(src.rows * cosv + src.cols * sinv);
  // adjust transformation matrix
  rot.at<double>(0, 2) += (bound_w - 1) / 2.0 - center.x;
  rot.at<double>(1, 2) += (bound_h - 1) / 2.0 - center.y;
  cv::warpAffine(src, dest, rot, {bound_w, bound_h});
}

// normalize positions of points
vector<size_t>
reorder_and_nique_points(const vector<cv::Point2f>& v)
{
  // use 3 rules to rerank the points
  // rule1: point0 and point3 is left side, point1, point2 is right
  // rule2: point0 is top than point3, point1 is top than point2
  // rule3 if distance(point0, point1) is large than distance(Point1, Point2)
  //  then, shift the points, new order is point1, point2, point3, point0
  vector<size_t> idx = {0, 1, 2, 3};
  sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
    return v[i1].x < v[i2].x;
  });
  size_t p0 = idx[0], p3 = idx[1];
  size_t p1 = idx[2], p2 = idx[3];
  // make sure p1.y is small then p3.y if p1.x equals p3.x
  if (v[p1].x == v[p3].x && v[p1].y > v[p3].y) {
    swap(p1, p3);
  }
  // make sure p0.y is small then p3.y
  if (v[p0].y > v[p3].y) {
    swap(p0, p3);
  }
  // make sure p1.y is small then p2.y
  if (v[p1].y > v[p2].y) {
    swap(p1, p2);
  }
  // width / height < 0.4, the logic come from (@jingtao)
  if (l2_norm(v[p0], v[p1]) / l2_norm(v[p1], v[p2]) < 0.4) {
    idx.assign({p1, p2, p3, p0});
  } else {
    idx.assign({p0, p1, p2, p3});
  }
  return idx;
}

vector<size_t>
reorder_quadrangle_points(const vector<cv::Point2f>& v)
{
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
    return v[i1].x < v[i2].x;
  });
  size_t p0 = idx[0], p3 = idx[1];
  size_t p1 = idx[2], p2 = idx[3];
  // make sure p1.y is small then p3.y if p1.x equals p3.x
  if (v[p1].x == v[p3].x && v[p1].y > v[p3].y) {
    swap(p1, p3);
  }
  // make sure p0.y is small then p3.y
  if (v[p0].y > v[p3].y) {
    swap(p0, p3);
  }
  // make sure p1.y is small then p2.y
  if (v[p1].y > v[p2].y) {
    swap(p1, p2);
  }
  return {p0, p1, p2, p3};
}

// width / height < 0.4, the logic comes from (@jingtao)
void
nique_points(const vector<cv::Point2f>& v, vector<size_t>& idx, float thres)
{
  if (l2_norm(v[idx[0]], v[idx[1]]) / l2_norm(v[idx[1]], v[idx[2]]) < thres) {
    idx.assign({idx[1], idx[2], idx[3], idx[0]});
  }
}

// get rotated rectangle roi
void
getRRectRoiOp(
    const cv::Mat& src, cv::Mat& dest, const vector<cv::Point2f>& src_points,
    int fixed_h)
{
  vector<size_t> idx = move(reorder_and_nique_points(src_points));
  float ori_w = round(l2_norm(src_points[idx[0]], src_points[idx[1]]));
  float ori_h = round(l2_norm(src_points[idx[1]], src_points[idx[2]]));
  float new_w = CvRound(fixed_h / ori_h * ori_w);
  float new_h = float(fixed_h);
  vector<cv::Point2f> src_3points{src_points[idx[0]], src_points[idx[1]],
                                  src_points[idx[2]]};
  vector<cv::Point2f> dest_3points = {{0, 0}, {new_w, 0}, {new_w, new_h}};
  cv::Mat warp_m = cv::getAffineTransform(src_3points, dest_3points);
  cv::warpAffine(src, dest, warp_m, {int(new_w), fixed_h}, cv::INTER_LINEAR);
}

// get rotated rectangle roi
void
getRRectRoiOp(
    const cv::Mat& src, cv::Mat& dst, const vector<cv::Point2f>& points,
    int new_w, int new_h)
{
  vector<size_t> idx = move(reorder_and_nique_points(points));
  vector<cv::Point2f> src_3points{points[idx[0]], points[idx[1]],
                                  points[idx[2]]};
  float new_w_f = float(new_w), new_h_f = float(new_h);
  vector<cv::Point2f> dest_3points = {{0, 0}, {new_w_f, 0}, {new_w_f, new_h_f}};
  cv::Mat warp_m = cv::getAffineTransform(src_3points, dest_3points);
  cv::warpAffine(src, dst, warp_m, {new_w, new_h}, cv::INTER_LINEAR);
}

// get rotated rectangle rois
void
getRRectRoisWithPaddingOp(
    const cv::Mat& src, vector<cv::Mat>& dest, vector<float>& dest_widths,
    const vector<vector<cv::Point2f>>& points_vec, int fixed_h)
{
  vector<vector<size_t>> idx_vec;
  vector<cv::Size> dest_size_vec;
  vector<cv::Mat> temp_mats;

  for (const auto& v : points_vec) {
    vector<size_t> idx = move(reorder_and_nique_points(v));
    float ori_w = round(l2_norm(v[idx[0]], v[idx[1]]));
    float ori_h = round(l2_norm(v[idx[1]], v[idx[2]]));
    float new_w = CvRound(fixed_h / ori_h * ori_w);
    float new_h = float(fixed_h);
    vector<cv::Point2f> src_3points{v[idx[0]], v[idx[1]], v[idx[2]]};
    vector<cv::Point2f> dest_3points{{0, 0}, {new_w, 0}, {new_w, new_h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    int out_w = int(new_w);
    cv::Mat m;
    cv::warpAffine(src, m, warp_mat, {out_w, fixed_h}, cv::INTER_LINEAR);
    dest_size_vec.push_back({out_w, fixed_h});
    temp_mats.emplace_back(m);
  }

  int max_width = -1;
  for (const auto v : dest_size_vec) {
    dest_widths.push_back(v.width);
    if (v.width > max_width) {
      max_width = v.width;
    }
  }
  for (const auto m : temp_mats) {
    if (m.cols < max_width) {
      cv::Mat new_m = cv::Mat::zeros(fixed_h, max_width, src.type());
      m.copyTo(new_m(cv::Rect(0, 0, m.cols, fixed_h)));
      dest.emplace_back(new_m);
    } else {
      dest.emplace_back(m);
    }
  }
}

// added 2020.06.09, get rois
void
getRRectRoisWithPaddingOp4(
    const cv::Mat& src, const cv::Mat& bbs, int fixed_h, float nique_threshold,
    int output_channels, vector<cv::Mat>& rois)
{
  cv::Mat tgt;
  if (output_channels == 1) {
    bgr2grayOp(src, tgt);
  } else {
    tgt = src;
  }
  int n = bbs.rows;
  cv::Mat_<cv::Point2f> bbs_(bbs);
  for (int i = 0; i < n; i++) {
    std::vector<cv::Point2f> v;
    for (int j = 0; j < 4; j++) {
      v.emplace_back(bbs_(i, j));
    }
    float ori_w = round(l2_norm(v[0], v[1]));
    float ori_h = round(l2_norm(v[1], v[2]));
    int i1 = 0, i2 = 1, i3 = 2;
    if (ori_w / ori_h < nique_threshold) {
      // std::swap(ori_w, ori_h);
      ori_w = round(l2_norm(v[1], v[2]));
      ori_h = round(l2_norm(v[2], v[3]));
      i1 = 1;
      i2 = 2;
      i3 = 3;
    }
    float new_w = CvRound(fixed_h / ori_h * ori_w);
    float new_h = float(fixed_h);
    vector<cv::Point2f> src_3points{v[i1], v[i2], v[i3]};
    vector<cv::Point2f> dest_3points{{0, 0}, {new_w, 0}, {new_w, new_h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    int out_w = int(new_w);
    cv::Mat m;
    cv::warpAffine(tgt, m, warp_mat, {out_w, fixed_h}, cv::INTER_LINEAR);
    rois.emplace_back(m);
  }
}

// added 2020.02.29, change bbs type
// get rotated rectangle rois for transformer v2 and ctc-revised 1.1
void
getRRectRoisWithPaddingOp3(
    const cv::Mat& src, const cv::Mat& bbs, int batch_size, int fixed_h,
    float nique_threshold, int output_channels, bool use_min_width_limit,
    int device_count, PairMatList& dst, std::vector<int>& dst_indexes)
{
  // ctc not use min_width_limit
  int n = bbs.rows;
  const int MIN_WIDTH = 40;
  std::vector<std::pair<int, int>> temp_widths;
  cv::Mat gray;
  if (output_channels == 1) {
    bgr2grayOp(src, gray);
  } else {
    gray = src;
  }

  vector<cv::Mat> temp_mats(n);
  cv::Mat_<cv::Point2f> bbs_(bbs);
  for (int i = 0; i < n; i++) {
    std::vector<cv::Point2f> v;
    for (int j = 0; j < 4; j++) {
      v.emplace_back(bbs_(i, j));
    }
    float ori_w = round(l2_norm(v[0], v[1]));
    float ori_h = round(l2_norm(v[1], v[2]));
    int i1 = 0, i2 = 1, i3 = 2;
    if (ori_w / ori_h < nique_threshold) {
      // std::swap(ori_w, ori_h);
      ori_w = round(l2_norm(v[1], v[2]));
      ori_h = round(l2_norm(v[2], v[3]));
      i1 = 1;
      i2 = 2;
      i3 = 3;
    }
    float new_w = CvRound(fixed_h / ori_h * ori_w);
    float new_h = float(fixed_h);
    vector<cv::Point2f> src_3points{v[i1], v[i2], v[i3]};
    vector<cv::Point2f> dest_3points{{0, 0}, {new_w, 0}, {new_w, new_h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    int out_w = int(new_w);
    cv::Mat m;
    cv::warpAffine(gray, m, warp_mat, {out_w, fixed_h}, cv::INTER_LINEAR);
    temp_widths.emplace_back(out_w, i);
    m.convertTo(temp_mats[i], CV_32F, 1 / 255.0);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first < p2.first;
      });
  for (const auto& v : temp_widths) {
    dst_indexes.push_back(v.second);
  }

  // this logic will changed to dynamic config
  int selected_batchs, selected_batch_size;
  if (device_count <= 1) {
    // cpu mode: device_count = 0; gpu mode, only one device
    selected_batch_size = batch_size;
    selected_batchs = std::ceil(n * 1.0 / batch_size);
  } else {
    // gpu mode
    if (n <= batch_size * device_count) {
      selected_batch_size = std::ceil(n * 1.0 / device_count);
      selected_batchs = n <= device_count ? n : device_count;
    } else {
      selected_batch_size = batch_size;
      selected_batchs = std::ceil(n * 1.0 / batch_size);
    }
  }

  // int batchs = std::ceil(n * 1.0 / batch_size);
  int max_width = -1;
  for (int k = 0; k < selected_batchs; k++) {
    int s = k * selected_batch_size;
    int e = k == (selected_batchs - 1) ? n : (k + 1) * selected_batch_size;

    int sn = e - s;
    cv::Mat widths = cv::Mat::zeros(sn, 2, CV_32S);
    cv::Mat_<int> widths_(sn, 2, reinterpret_cast<int*>(widths.data));
    max_width = temp_widths[e - 1].first;
    if (use_min_width_limit && max_width < MIN_WIDTH) {
      max_width = MIN_WIDTH;
    }

    // this is very tricky, bns will broadcast noise
    max_width += fixed_h;

    std::vector<int> size = {sn, fixed_h, max_width, output_channels};
    auto rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));
    int step0 = 1;
    for (unsigned int j = 1; j < size.size(); j++) {
      step0 *= size[j];
    }
    auto type = output_channels > 1 ? CV_32FC3 : CV_32FC1;
    for (int i = 0; i < sn; ++i) {
      int m_index = temp_widths[i + s].second;
      int new_w = temp_widths[i + s].first;
      cv::Mat mapped_mat(fixed_h, max_width, type, rois.data + step0 * 4 * i);

      if (use_min_width_limit) {
        int pad_l = 0, pad_r = 0;
        if (new_w < MIN_WIDTH) {
          pad_l = (MIN_WIDTH - new_w) / 2;
          pad_r = (MIN_WIDTH - new_w - pad_l);
          new_w = MIN_WIDTH;
        }
        cv::Mat roi = mapped_mat(cv::Rect(0, 0, new_w, fixed_h));
        cv::copyMakeBorder(
            temp_mats[m_index], roi, 0, 0, pad_l, pad_r, cv::BORDER_CONSTANT,
            1.0);
      } else {
        temp_mats[m_index].copyTo(mapped_mat(cv::Rect(0, 0, new_w, fixed_h)));
      }
      widths_(i, 1) = new_w;
    }
    dst.emplace_back(rois, widths);
  }
}

// get rotated rectangle rois for transformer
void
getRRectRoisWithPaddingOp2(
    const cv::Mat& src, cv::Mat& rois, cv::Mat& widths,
    const vector<vector<cv::Point2f>>& points_vec, int fixed_h,
    vector<float>& bbs, float nique_threshold)
{
  int max_width = -1, n = points_vec.size();
  const int MIN_WIDTH = 40;
  widths = cv::Mat::zeros(n, 2, CV_32S);
  cv::Mat_<int> widths_(n, 2, reinterpret_cast<int*>(widths.data));
  cv::Mat gray;
  bgr2grayOp(src, gray);
  for (const auto& v : points_vec) {
    vector<size_t> idx = {0, 1, 2, 3};
    for (const auto& i : idx) {
      bbs.emplace_back(v[i].x);
      bbs.emplace_back(v[i].y);
    }
  }
  vector<cv::Mat> temp_mats(n);
  // #pragma omp parallel for num_threads(THREAD_NUM_FOR_OPENCV)
  for (int i = 0; i < n; i++) {
    const auto& v = points_vec[i];
    vector<size_t> idx = {0, 1, 2, 3};
    nique_points(v, idx, nique_threshold);
    float ori_w = round(l2_norm(v[idx[0]], v[idx[1]]));
    float ori_h = round(l2_norm(v[idx[1]], v[idx[2]]));
    float new_w = CvRound(fixed_h / ori_h * ori_w);
    float new_h = float(fixed_h);
    vector<cv::Point2f> src_3points{v[idx[0]], v[idx[1]], v[idx[2]]};
    vector<cv::Point2f> dest_3points{{0, 0}, {new_w, 0}, {new_w, new_h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    int out_w = int(new_w);
    cv::Mat m;
    cv::warpAffine(gray, m, warp_mat, {out_w, fixed_h}, cv::INTER_LINEAR);
    m.convertTo(temp_mats[i], CV_32F);
    widths_(i, 1) = out_w;
  }
  for (int i = 0; i < n; i++) {
    if (widths_(i, 1) > max_width) {
      max_width = widths_(i, 1);
    }
  }
  if (max_width < MIN_WIDTH) {
    max_width = MIN_WIDTH;
  }
  const int c = 1;
  std::vector<int> size = {n, fixed_h, max_width, c};
  rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }

  auto cv_type = c > 1 ? CV_32FC3 : CV_32FC1;
  // #pragma omp parallel for num_threads(THREAD_NUM_FOR_OPENCV)
  for (unsigned int i = 0; i < temp_mats.size(); ++i) {
    int new_w = widths_(i, 1);
    cv::Mat mapped_mat(fixed_h, max_width, cv_type, rois.data + step0 * 4 * i);
    // be careful, enlarge the marge of the input
    if (new_w < MIN_WIDTH) {
      mapped_mat(cv::Rect(0, 0, MIN_WIDTH, fixed_h)) = 1.0;
      widths_(i, 1) = MIN_WIDTH;
    }
    int start_x = new_w < MIN_WIDTH ? (MIN_WIDTH - new_w) / 2 : 0;
    auto roi = mapped_mat(cv::Rect(start_x, 0, new_w, fixed_h));
    temp_mats[i].copyTo(roi);
    roi.convertTo(roi, cv_type, 1.0 / 255.0);
  }
}

// get rectangle roi, approx value comes from net inference, not defined by user
void
getRectRoiOp(
    const cv::Mat& src, cv::Mat& dest, const vector<cv::Point2f> points,
    int fixed_h)
{
  // input bbox is 2 points represents left-top, right-bottom position
  auto r = cv::Rect(points[0], points[1]);
  cv::Mat crop = src(r);
  if (r.height != fixed_h) {
    int new_w = CvRoundI(1.0 * r.width * fixed_h / r.height);
    resizeOp(crop, dest, new_w, fixed_h);
  } else {
    dest = std::move(crop.clone());
  }
}

// get rectangle rois, the op is used for ctc model provided by liuqinjie@
//  modify by hf, 2020.02.29, change the 2nd param rects to bbs
void
getRectRoisWithPaddingOp(
    const cv::Mat& src, const cv::Mat& bbs, cv::Mat& rois, cv::Mat& widths,
    int fixed_h, bool align_4factor)
{
  int index = 0, max_width = -1, n = bbs.size[0];
  auto rects = std::vector<cv::Rect>(bbs);
  int width_size[] = {n, 1, 1};
  widths = cv::Mat(3, width_size, CV_32S);
  cv::Mat_<int> widths_(n, 1, reinterpret_cast<int*>(widths.data));
  for (const auto& r : rects) {
    int new_w =
        (align_4factor
             ? int(std::ceil(1. * r.width * fixed_h / r.height / 4.0) * 4)
             : int(std::ceil(1. * r.width * fixed_h / r.height)));
    widths_(index++, 0) = new_w;
    if (max_width < new_w) {
      max_width = new_w;
    }
  }

  int c = src.channels();
  std::vector<int> size = {n, fixed_h, max_width};
  if (c > 1) {
    size.push_back(c);
  }
  rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0, 0, 0));

  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }

  auto cv_type = c > 1 ? CV_32FC3 : CV_32FC1;
  for (unsigned int i = 0; i < rects.size(); ++i) {
    auto& r = rects[i];
    int new_w = widths_(i, 0);
    cv::Mat crop = src(r);
    cv::Mat mapped_mat(fixed_h, max_width, cv_type, rois.data + step0 * 4 * i);
    auto roi = mapped_mat(cv::Rect(0, 0, new_w, fixed_h));
    cv::Mat temp;
    if (r.width == new_w && r.height == fixed_h) {
      crop.copyTo(roi);
    } else {
      resizeOp(crop, roi, new_w, fixed_h);
    }
    roi.convertTo(roi, cv_type, 1.0 / 255.0, -0.5);
    widths_(i, 0) = widths_(i, 0) / 4;
  }
}

// get rectangle rois for transoformer model provided by chenfeng@
//  modify by hf, 2020.02.29, change the 2nd param rects to bbs
void
getRectRoisWithPaddingOp2(
    const cv::Mat& src, const cv::Mat& bbs, cv::Mat& rois, cv::Mat& widths,
    int fixed_h)
{
  int index = 0, max_width = -1, n = bbs.size[0];
  const int MIN_WIDTH = 40;
  widths = cv::Mat::zeros(n, 2, CV_32S);
  cv::Mat_<int> widths_(n, 2, reinterpret_cast<int*>(widths.data));
  // cv::Mat_<cv::Rect> rects(bbs);
  auto rects = std::vector<cv::Rect>(bbs);
  for (const auto& r : rects) {
    float coef = fixed_h * 1.0 / r.height;
    int new_w = int(coef * r.width);

    widths_(index++, 1) = new_w;
    if (max_width < new_w) {
      max_width = new_w;
    }
  }

  if (max_width < MIN_WIDTH) {
    max_width = MIN_WIDTH;
  }
  const int c = 1;
  std::vector<int> size = {n, fixed_h, max_width, c};
  rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }

  auto cv_type = c > 1 ? CV_32FC3 : CV_32FC1;
  cv::Mat gray;
  bgr2grayOp(src, gray);
  for (unsigned int i = 0; i < rects.size(); ++i) {
    auto& r = rects[i];
    int new_w = widths_(i, 1);
    cv::Mat crop = gray(r);
    cv::Mat mapped_mat(fixed_h, max_width, cv_type, rois.data + step0 * 4 * i);
    // be careful, enlarge the marge of the input
    if (new_w < MIN_WIDTH) {
      mapped_mat(cv::Rect(0, 0, MIN_WIDTH, fixed_h)) = 1.0;
      widths_(i, 1) = MIN_WIDTH;
    }
    int start_x = new_w < MIN_WIDTH ? (MIN_WIDTH - new_w) / 2 : 0;
    auto roi = mapped_mat(cv::Rect(start_x, 0, new_w, fixed_h));
    if (r.width == new_w && r.height == fixed_h) {
      crop.copyTo(roi);
    } else {
      resizeOp(crop, roi, new_w, fixed_h);
    }

    roi.convertTo(roi, cv_type, 1.0 / 255.0);
  }
}

void
transformerPreprocessOp(
    const std::vector<cv::Mat>& mat_list, cv::Mat& rois, cv::Mat& widths,
    int fixed_h)
{
  int index = 0, max_width = -1, n = mat_list.size();
  const int MIN_WIDTH = 40;
  widths = cv::Mat::zeros(n, 2, CV_32S);
  cv::Mat_<int> widths_(n, 2, reinterpret_cast<int*>(widths.data));
  for (const auto& mat : mat_list) {
    float coef = fixed_h * 1.0 / mat.rows;
    int new_w = int(coef * mat.cols);
    widths_(index++, 1) = new_w;
    if (max_width < new_w) {
      max_width = new_w;
    }
  }
  if (max_width < MIN_WIDTH) {
    max_width = MIN_WIDTH;
  }

  const int c = 1;
  std::vector<int> size = {n, fixed_h, max_width, c};
  rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));

  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }

  auto cv_type = c > 1 ? CV_32FC3 : CV_32FC1;
  for (int i = 0; i < n; ++i) {
    int new_w = widths_(i, 1);
    cv::Mat mapped_mat(fixed_h, max_width, cv_type, rois.data + step0 * 4 * i);
    // be careful, enlarge the marge of the input
    if (new_w < MIN_WIDTH) {
      mapped_mat(cv::Rect(0, 0, MIN_WIDTH, fixed_h)) = 1.0;
      widths_(i, 1) = MIN_WIDTH;
    }
    int start_x = new_w < MIN_WIDTH ? (MIN_WIDTH - new_w) / 2 : 0;
    auto roi = mapped_mat(cv::Rect(start_x, 0, new_w, fixed_h));
    // input mat is uint8 type
    cv::Mat gray, resize, src;
    bgr2grayOp(mat_list[i], gray);
    if (gray.cols != new_w || gray.rows != fixed_h) {
      resizeOp(gray, resize, new_w, fixed_h);
    } else {
      resize = gray;
    }
    resize.convertTo(src, CV_32F);
    src.copyTo(roi);
    roi.convertTo(roi, CV_32F, 1.0 / 255.0);
  }
}

// create batch input from mats for transformer v2 and ctc-revised 1.1
//  updated by hf, 2020.03.29 make sure the output_channels is the same of mat
//  in srcs
//   otherwise it will different with getRRectRoisWithPaddingOp3
void
transformerCTCPreprocessOp(
    const std::vector<cv::Mat>& srcs, int batch_size, int fixed_h,
    int output_channels, bool use_min_width_limit, int device_count,
    PairMatList& dst, std::vector<int>& dst_indexes)
{
  int n = srcs.size();
  const int MIN_WIDTH = 40;
  std::vector<std::pair<int, int>> temp_widths;
  for (int i = 0; i < n; i++) {
    const auto& mat = srcs[i];
    int new_w = int(fixed_h * 1.0 / mat.rows * mat.cols);
    temp_widths.emplace_back(new_w, i);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first < p2.first;
      });
  for (const auto& v : temp_widths) {
    dst_indexes.push_back(v.second);
  }

  // this logic will changed to dynamic config
  int selected_batchs, selected_batch_size;
  if (device_count <= 1) {
    // cpu mode: device_count = 0; gpu mode, only one device
    selected_batch_size = batch_size;
    selected_batchs = std::ceil(n * 1.0 / batch_size);
  } else {
    // gpu mode
    if (n <= batch_size * device_count) {
      selected_batch_size = std::ceil(n * 1.0 / device_count);
      selected_batchs = n <= device_count ? n : device_count;
    } else {
      selected_batch_size = batch_size;
      selected_batchs = std::ceil(n * 1.0 / batch_size);
    }
  }

  // int batchs = std::ceil(n * 1.0 / batch_size);
  int max_width = -1;
  for (int k = 0; k < selected_batchs; k++) {
    int s = k * selected_batch_size;
    int e = k == (selected_batchs - 1) ? n : (k + 1) * selected_batch_size;
    int sn = e - s;
    cv::Mat widths = cv::Mat::zeros(sn, 2, CV_32S);
    cv::Mat_<int> widths_(widths);
    max_width = temp_widths[e - 1].first;

    if (use_min_width_limit && max_width < MIN_WIDTH) {
      max_width = MIN_WIDTH;
    }
    // see getRRectRoisWithPaddingOp3
    max_width += 32;
    const int& c = output_channels;
    std::vector<int> size = {sn, fixed_h, max_width, c};
    auto rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));
    int step0 = 1;
    for (unsigned int j = 1; j < size.size(); j++) {
      step0 *= size[j];
    }
    auto type = c > 1 ? CV_32FC3 : CV_32FC1;
    for (int i = 0; i < sn; ++i) {
      int m_index = temp_widths[s + i].second;
      int new_w = temp_widths[s + i].first;
      cv::Mat mapped_mat(fixed_h, max_width, type, rois.data + step0 * 4 * i);
      // be careful, enlarge the marge of the input
      int start_x = 0;
      if (use_min_width_limit && new_w < MIN_WIDTH) {
        mapped_mat(cv::Rect(0, 0, MIN_WIDTH, fixed_h)) = 1.0;
        widths_(i, 1) = MIN_WIDTH;
        start_x = (MIN_WIDTH - new_w) / 2;
      } else {
        widths_(i, 1) = new_w;
      }

      auto roi = mapped_mat(cv::Rect(start_x, 0, new_w, fixed_h));
      const cv::Mat& m = srcs[m_index];
      cv::Mat resize;
      if (m.cols != new_w || m.rows != fixed_h) {
        resizeOp(m, resize, new_w, fixed_h);
      } else {
        resize = m;
      }
      resize.convertTo(roi, type, 1.0 / 255.0);
    }
    dst.emplace_back(rois, widths);
  }
}

// 20201027 hjt update
int
resize_image(
    const cv::Mat& img1, cv::Mat& img5, int H, int W_min, int W_max,
    bool is_grayscale, int downsample_rate, int extra_padding_length)
{
  if (W_min % downsample_rate != 0 || W_max % downsample_rate != 0) {
    return -1;
  }
  // int type = is_grayscale ? CV_8U : CV_8UC3;

  cv::Mat img2;
  // very slow here
  if (is_grayscale) {
    bgr2grayOp(img1, img2);
  } else {
    bgr2rgbOp(img1, img2);
  }

  int h = img2.rows;
  int w = img2.cols;

  int w2 = std::max((H * w) / h, 1);
  int W = std::ceil(w2 * 1.0 / downsample_rate) * downsample_rate;

  cv::Mat img3;
  if (W <= W_max && W >= W_min) {
    resizeOp(img2, img3, W, H);  // img3.shape [H, W, 1]

  } else if (W < W_min) {
    cv::Mat img2prime;
    resizeOp(img2, img2prime, W, H);
    cv::copyMakeBorder(
        img2prime, img3, 0, 0, 0, W_min - W, cv::BORDER_CONSTANT, 0);
    // cv::Mat tail = cv::Mat::zeros(H, W_min - W, type);
    // cv::hconcat(img2prime, tail, img3); // pad with 0
  } else {
    W = W_max;
    /*
    img3 = cv::Mat::zeros(H, W, type);
    int h2 = std::max((W * h) / w, 1);
    cv::Mat img2prime;
    resizeOp(img2, img2prime, W, h2);
    int margin = (H - h2) / 2;
    img2prime.copyTo(img3(cv::Rect(0, margin, W, h2)));
    */
    cv::Mat img2prime;
    int h2 = std::max((W * h) / w, 1);
    resizeOp(img2, img2prime, W, h2);
    int margin = (H - h2) / 2;
    int remainder = (H - h2) % 2;
    int bottom = margin + remainder;
    int top = margin;
    cv::copyMakeBorder(
        img2prime, img3, top, bottom, 0, 0, cv::BORDER_CONSTANT, 0);
  }

  // extra tail padding
  // cv::Mat tail = cv::Mat::zeros(H, extra_padding_length, type);
  cv::Mat img4;
  // cv::hconcat(img3, tail, img4);
  // cvhelp::Timer timer2 = cvhelp::Timer();
  // timer2.tic();
  cv::copyMakeBorder(
      img3, img4, 0, 0, 0, extra_padding_length, cv::BORDER_CONSTANT, 0);
  img4.convertTo(img5, 5, 1.0 / 255.0);
  // int elapse = timer2.toc();
  // std::cout << "reizeOp time: " << elapse << " ms" << std::endl;
  return 0;
}

// 20201027 hjt update
void
preprocess_recog_batch_v2(
    const std::vector<cv::Mat>& imgs, cv::Mat& rois, cv::Mat& shapes, int H,
    int W_min, int W_max, bool is_grayscale, int downsample_rate,
    int extra_padding_length)
{
  int channels = is_grayscale ? 1 : 3;
  std::vector<cv::Mat> imgs_preprocessed;
  std::vector<int> shapes_preprocessed;

  int shape_max = 0;
  for (auto img0 : imgs) {
    // cv::Mat img1, img5;

    // cvhelp::Timer timer2 = cvhelp::Timer();
    // timer2.tic();

    // bgr2rgbOp(img0, img1);

    // int elapse = timer2.toc();
    // std::cout << "bgr2rbg time: " << elapse << " ms" << std::endl;
    cv::Mat img5;
    resize_image(
        img0, img5, H, W_min, W_max, is_grayscale, downsample_rate,
        extra_padding_length);

    shape_max = std::max(shape_max, img5.cols);
    shapes_preprocessed.emplace_back(img5.rows);
    shapes_preprocessed.emplace_back(img5.cols - extra_padding_length);
    imgs_preprocessed.emplace_back(std::move(img5));
  }

  int batchsize = imgs_preprocessed.size();
  std::vector<int> size = {batchsize, H, shape_max, channels};
  rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));

  int volumn_bytes = H * shape_max * channels * 4;
  int cv_type = is_grayscale ? CV_32FC1 : CV_32FC3;

  for (int i = 0; i < batchsize; i++) {
    cv::Mat mapped_mat(H, shape_max, cv_type, rois.data + volumn_bytes * i);
    imgs_preprocessed[i].copyTo(mapped_mat(
        cv::Rect(0, 0, imgs_preprocessed[i].cols, imgs_preprocessed[i].rows)));
  }

  shapes = cv::Mat(shapes_preprocessed);
  shapes = shapes.reshape(0, batchsize).clone();
}

// 20201027 hjt modification #3 and #4
void
transformerCTCPreprocessOp2(
    const std::vector<cv::Mat>& srcs, int batch_size, int fixed_h,
    int output_channels, int downsample_rate, int W_min, int W_max,
    int device_count, PairMatList& dst, std::vector<int>& dst_indexes)
{
  // #3 stable sort
  int n = srcs.size();
  // const int MIN_WIDTH = 40;
  std::vector<std::pair<int, int>> temp_widths;
  for (int i = 0; i < n; i++) {
    const auto& mat = srcs[i];
    int new_w = int(fixed_h * 1.0 / mat.rows * mat.cols);
    temp_widths.emplace_back(new_w, i);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first > p2.first;
      });
  for (const auto& v : temp_widths) {
    dst_indexes.push_back(v.second);
  }

  // this logic will changed to dynamic config
  int selected_batchs, selected_batch_size;
  if (device_count <= 1) {
    // cpu mode: device_count = 0; gpu mode, only one device
    selected_batch_size = batch_size;
    selected_batchs = std::ceil(n * 1.0 / batch_size);
  } else {
    // gpu mode
    if (n <= batch_size * device_count) {
      selected_batch_size = std::ceil(n * 1.0 / device_count);
      selected_batchs = n <= device_count ? n : device_count;
    } else {
      selected_batch_size = batch_size;
      selected_batchs = std::ceil(n * 1.0 / batch_size);
    }
  }

  // #4 new preprocessing function
  for (int k = 0; k < selected_batchs; k++) {
    int s = k * selected_batch_size;
    int e = k == (selected_batchs - 1) ? n : (k + 1) * selected_batch_size;
    int sn = e - s;

    // cv::Mat widths = cv::Mat::zeros(sn, 2, CV_32S);
    // cv::Mat_<int> widths_(widths);
    cv::Mat rois;
    cv::Mat widths;
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < sn; ++i) {
      // fix a bug: temp_widths[s + i].second
      imgs.emplace_back(srcs[temp_widths[s + i].second]);
    }
    int extra_padding_length = 108;
    // int H = fixed_h; passed by outside parameter
    // int W_min = 40; passed by outside parameter
    // int W_max = 800; passed by outside parameter
    // int downsample_rate = 8; passed by outside parameter
    bool is_grayscale = output_channels == 1 ? true : false;

    preprocess_recog_batch_v2(
        imgs, rois, widths, fixed_h, W_min, W_max, is_grayscale,
        downsample_rate, extra_padding_length);
    dst.emplace_back(rois, widths);
  }
}

void
enLargeRRectOp(
    const vector<cv::Point2f>& src_points, vector<cv::Point2f>& dest_points,
    float deltaW, float deltaH)
{
  // references from https://github.com/opencv/opencv/blob/master/
  //     modules/core/src/types.cpp
  vector<size_t> idx = move(reorder_and_nique_points(src_points));
  auto _point1 = src_points[idx[0]];
  auto _point2 = src_points[idx[1]];
  auto _point3 = src_points[idx[2]];
  cv::Point2f _center = 0.5f * (_point1 + _point3);
  cv::Vec2f vecs[2];
  vecs[0] = cv::Vec2f(_point1 - _point2);
  vecs[1] = cv::Vec2f(_point2 - _point3);
  // wd_i stores which vector (0,1) or (1,2) will make the width
  // One of them will definitely have slope within -1 to 1
  int wd_i = 0;
  if (fabs(vecs[1][1]) < fabs(vecs[1][0]))
    wd_i = 1;
  int ht_i = (wd_i + 1) % 2;
  float _angle =
      (std::atan(vecs[wd_i][1] / vecs[wd_i][0]) * 180.0f / (float)CV_PI);
  float _width = (float)cv::norm(vecs[wd_i]);
  float _height = (float)cv::norm(vecs[ht_i]);

  cv::Point2f pts[4];
  cv::RotatedRect(_center, {_width + deltaW, _height + deltaH}, _angle)
      .points(pts);
  dest_points.assign(pts, pts + 4);
}

void
ctcPreprocessOp(
    const std::vector<cv::Mat>& mat_list, int fixed_h, cv::Mat& dst,
    cv::Mat& width_mat)
{
  // output mat: n,w,h,c
  int n = mat_list.size();
  // width_mat = cv::Mat(n, 1, CV_32S);

  int width_size[] = {n, 1, 1};
  width_mat = cv::Mat(3, width_size, CV_32S);
  cv::Mat_<int> widths_(n, 1, reinterpret_cast<int*>(width_mat.data));

  std::vector<int> new_width_list;
  int max_width = 0;
  for (int i = 0; i < n; i++) {
    const auto& mat = mat_list[i];
    int new_w = int(std::ceil(mat.cols * 1.0 * fixed_h / mat.rows / 4.0) * 4);
    widths_(i, 0) = int(std::floor(new_w / 4.0));
    if (new_w > max_width) {
      max_width = new_w;
    }
    new_width_list.push_back(new_w);
  }

  int c = mat_list[0].channels();
  std::vector<int> size = {n, max_width, fixed_h};
  if (c > 1) {
    size.push_back(c);
  }
  int dims = size.size();
  dst = cv::Mat(dims, size.data(), CV_32F, cv::Scalar(0, 0, 0));
  int step0 = 1;
  for (int j = 1; j < dims; j++) {
    step0 *= size[j];
  }

  auto cv_type = c > 1 ? CV_32FC3 : CV_32FC1;
  for (int i = 0; i < n; i++) {
    const auto& mat = mat_list[i];
    int new_w = new_width_list[i];
    cv::Mat mat2, mat3;
    resizeOp(mat, mat2, new_w, fixed_h);
    cv::transpose(mat2, mat3);
    // mat3: w',h',c
    cv::Mat mapped_mat(max_width, fixed_h, cv_type, dst.data + step0 * 4 * i);
    if (new_w < max_width) {
      mat3.convertTo(
          mapped_mat(cv::Rect(0, 0, fixed_h, new_w)), cv_type, 1.0 / 255.0,
          -0.5);
    } else {
      mat3.convertTo(mapped_mat, cv_type, 1.0 / 255.0, -0.5);
    }
  }
}

// modified by liuqingjie & hanfeng, 2020.03.22
// change dst depth by input
void
mergeBatchMatOp(const std::vector<cv::Mat>& mat_list, cv::Mat& dst)
{
  auto m0 = mat_list[0];
  int n = mat_list.size(), h = m0.rows, w = m0.cols, c = m0.channels();
  std::vector<int> size = {n, h, w};
  if (c > 1) {
    size.push_back(c);
  }
  int dims = size.size();
  dst = cv::Mat(dims, size.data(), m0.depth(), cv::Scalar(0, 0, 0));
  int step0 = 1;
  for (int j = 1; j < dims; j++) {
    step0 *= size[j];
  }
  for (int i = 0; i < n; i++) {
    cv::Mat mapped_mat(h, w, m0.type(), dst.data + step0 * m0.elemSize1() * i);
    mat_list[i].copyTo(mapped_mat);
  }
}

void
transposeHW(const cv::Mat& src, cv::Mat& dst)
{
  int n = src.size[0], h = src.size[1], w = src.size[2], c = src.size[3];
  int size[] = {n, w, h, c};
  int step0 = w * h * c;
  dst = cv::Mat(4, size, CV_32F);
  auto* src_ptr = reinterpret_cast<float*>(src.data);
  auto* dst_ptr = reinterpret_cast<float*>(dst.data);
  for (int i = 0; i < n; i++) {
    cv::Mat src_(h, w, CV_32FC3, src_ptr + step0 * i);
    cv::Mat dst_(w, h, CV_32FC3, dst_ptr + step0 * i);
    cv::transpose(src_, dst_);
  }
}

void
drawRectangle(const cv::Mat& src, const cv::Mat& r, cv::Mat& dst)
{
  // rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1,
  //   int lineType=8, int shift=0 )
  dst = src.clone();
  for (int i = 0; i < r.rows; i++) {
    int x0 = (int)r.at<float>(i, 0);
    int y0 = (int)r.at<float>(i, 1);
    int x1 = (int)r.at<float>(i, 2);
    int y1 = (int)r.at<float>(i, 3);
    cv::Rect rect(x0, y0, x1 - x0, y1 - y0);
    cv::rectangle(dst, rect, cv::Scalar(0, 255, 0));
  }
}

void
compute_centerness_targets(const cv::Mat& geometry, cv::Mat& centerness)
{
  int n = geometry.size[0];
  cv::Mat left_side = geometry(cv::Rect(3, 0, 1, n));
  cv::Mat right_side = geometry(cv::Rect(1, 0, 1, n));
  cv::Mat top_side = geometry(cv::Rect(0, 0, 1, n));
  cv::Mat bottom_side = geometry(cv::Rect(2, 0, 1, n));

  cv::Mat left_right_min, top_bottom_min;
  cv::Mat left_right_max, top_bottom_max;
  cv::min(left_side, right_side, left_right_min);
  cv::min(top_side, bottom_side, top_bottom_min);
  cv::max(left_side, right_side, left_right_max);
  cv::max(top_side, bottom_side, top_bottom_max);

  cv::Mat centerness_square =
      (left_right_min / left_right_max).mul(top_bottom_min / top_bottom_max);

  cv::sqrt(centerness_square, centerness);
}

void
compute_sideness_targets(
    const cv::Mat& geometry, const std::string& lr_or_tb, cv::Mat& sideness)
{
  int n = geometry.size[0];
  cv::Mat left_side = geometry(cv::Rect(3, 0, 1, n));
  cv::Mat right_side = geometry(cv::Rect(1, 0, 1, n));
  cv::Mat top_side = geometry(cv::Rect(0, 0, 1, n));
  cv::Mat bottom_side = geometry(cv::Rect(2, 0, 1, n));
  cv::Mat left_right_min, top_bottom_min;
  cv::Mat left_right_max, top_bottom_max;
  cv::min(left_side, right_side, left_right_min);
  cv::min(top_side, bottom_side, top_bottom_min);
  cv::max(left_side, right_side, left_right_max);
  cv::max(top_side, bottom_side, top_bottom_max);
  cv::Mat sideness_square;
  if (lr_or_tb.compare("lr") == 0) {
    cv::Mat leftness = (right_side / (left_side + right_side))
                           .mul(top_bottom_min / top_bottom_max);
    cv::Mat rightness = (left_side / (left_side + right_side))
                            .mul(top_bottom_min / top_bottom_max);
    cv::hconcat(leftness, rightness, sideness_square);
  } else {
    cv::Mat topness = (bottom_side / (top_side + bottom_side))
                          .mul(left_right_min / left_right_max);
    cv::Mat bottomness = (top_side / (top_side + bottom_side))
                             .mul(left_right_min / left_right_max);
    cv::hconcat(topness, bottomness, sideness_square);
  }
  cv::sqrt(sideness_square, sideness);
}

void
point_inside_of_box(
    const std::vector<cv::Point2f>& xy_text,
    const std::vector<cv::Point2f>& box, std::vector<int>& idex)
{
  std::vector<cv::Point2f> box_edge_vertor;
  std::vector<cv::Point2f> box_yx;
  for (uint i = 0; i < box.size() - 1; i++) {
    box_edge_vertor.emplace_back(box[i + 1] - box[i]);
    box_yx.emplace_back(box[i].y, box[i].x);
  }
  box_edge_vertor.emplace_back(box[0] - box[box.size() - 1]);
  box_yx.emplace_back(box[box.size() - 1].y, box[box.size() - 1].x);

  std::vector<std::vector<float>> cross_mul_vec;
  for (uint i = 0; i < xy_text.size(); i++) {
    std::vector<float> cross_mul;
    for (uint j = 0; j < box_yx.size(); j++) {
      cv::Point2f point_sub;
      point_sub = cv::Point2f(xy_text[i].y, xy_text[i].x) - box_yx[j];
      cross_mul.emplace_back(
          box_edge_vertor[j].x * point_sub.x -
          box_edge_vertor[j].y * point_sub.y);
    }
    cross_mul_vec.emplace_back(std::move(cross_mul));
  }

  for (uint i = 0; i < cross_mul_vec.size(); i++) {
    float max_value =
        *max_element(cross_mul_vec[i].begin(), cross_mul_vec[i].end());
    float min_value =
        *min_element(cross_mul_vec[i].begin(), cross_mul_vec[i].end());
    if ((max_value <= 0) || (min_value >= 0)) {
      idex.emplace_back(i);
    }
  }
}

//# point1 point2 两个向量是否是一个方向，如果大于0 一个方向，小于0，反方向
bool
check_angle_length(const cv::Point2f& point1, const cv::Point2f& point2)
{
  if ((point1.dot(point2) > 0) &&
      (point_norm(point1) < 3 * point_norm(point2)) &&
      (point_norm(point1) > 0.3 * point_norm(point2))) {
    return true;
  } else {
    return false;
  }
}

void
refine_box(
    std::vector<cv::Point2f>& box, const std::vector<cv::Point2f>& xy_text,
    const cv::Mat& geometry, const std::vector<std::vector<cv::Point2f>>& boxes,
    float& ratio_h, float& ratio_w)
{
  if (max(point_norm(box[1] - box[0]), point_norm(box[2] - box[1])) >= 500) {
    std::vector<int> idex;
    point_inside_of_box(xy_text, box, idex);
    std::string left_right;
    if (point_norm(box[1] - box[0]) > point_norm(box[2] - box[1])) {
      left_right = "lr";
    } else {
      left_right = "tb";
    }
    cv::Mat geometry_inside_of_box;
    cv::Mat geometry_tmp;
    for (uint i = 0; i < idex.size(); i++) {
      cv::Mat select_geometry = geometry(cv::Rect(0, idex[i], 4, 1));
      if (i == 0) {
        geometry_inside_of_box = select_geometry;
      } else {
        cv::vconcat(geometry_tmp, select_geometry, geometry_inside_of_box);
      }
      geometry_tmp = geometry_inside_of_box;
    }
    cv::Mat sideness;
    compute_sideness_targets(geometry_inside_of_box, left_right, sideness);
    cv::Mat one_side = sideness(cv::Rect(0, 0, 1, sideness.size[0]));
    cv::Mat other_side = sideness(cv::Rect(1, 0, 1, sideness.size[0]));
    double minv, maxv;
    cv::Point one_side_min_index, one_side_max_index;
    cv::minMaxLoc(
        one_side, &minv, &maxv, &one_side_min_index, &one_side_max_index);
    cv::Point other_side_min_index, other_side_max_index;
    cv::minMaxLoc(
        other_side, &minv, &maxv, &other_side_min_index, &other_side_max_index);

    if (left_right.compare("lr") == 0) {
      std::vector<cv::Point2f> left_box = boxes[idex[one_side_max_index.y]];
      std::vector<cv::Point2f> right_box = boxes[idex[other_side_max_index.y]];
      if (check_angle_length(left_box[3] - left_box[0], box[3] - box[0])) {
        box[0] = left_box[0];
        box[3] = left_box[3];
      }
      if (check_angle_length(right_box[2] - right_box[1], box[2] - box[1])) {
        box[1] = right_box[1];
        box[2] = right_box[2];
      }
    } else {
      std::vector<cv::Point2f> top_box = boxes[idex[one_side_max_index.y]];
      std::vector<cv::Point2f> bottom_box = boxes[idex[other_side_max_index.y]];
      if (check_angle_length(top_box[1] - top_box[0], box[1] - box[0])) {
        box[0] = top_box[0];
        box[1] = top_box[1];
      }
      if (check_angle_length(bottom_box[3] - bottom_box[2], box[3] - box[2])) {
        box[2] = bottom_box[2];
        box[3] = bottom_box[3];
      }
    }
  }

  for (auto& point_xy : box) {
    point_xy.x = point_xy.x / ratio_w;
    point_xy.y = point_xy.y / ratio_h;
  }
}

double
computeAngle(
    const std::vector<cv::Point2f>& poly,
    bool normalize_image_based_on_text_orientation)
{
  double angle = 0.;
  double dy = poly[1].y - poly[0].y;
  double dx = poly[1].x - poly[0].x;

  if (dx == 0. and dy == 0.) {
    angle = 0.;
  }
  if (dx == 0. and dy != 0) {
    if (dy > 0.) {
      angle = CV_PI / 2.;
    } else {
      angle = 3.0 * CV_PI / 2.0;
    }
  }
  if (dy == 0. and dx != 0) {
    if (dx > 0.) {
      angle = 0;
    } else {
      angle = CV_PI;
    }
  }

  if (dx > 0. && dy > 0.) {
    angle = std::atan(dy / dx);
  } else if (dx > 0. && dy < 0.) {
    angle = CV_PI * 2 - std::atan(-1 * dy / dx);
  } else if (dx < 0. && dy < 0.) {
    angle = CV_PI + std::atan(dy / dx);
  } else if (dx < 0. && dy > 0.) {
    angle = CV_PI / 2.0 + std::atan(-1 * dx / dy);
  }
  return angle;
}

// compute angle, return angle [-45, 45]
double
computeAngle(const std::vector<cv::Point2f>& poly)
{
  // if the bottom line is parallel to x-axis,
  //   then p0 must be the upper-left corner
  // updated by 2020.04.16, logic from lixin
  double angle = 0.;
  if (abs(poly[2].y - poly[3].y) < 1e-6 || abs(poly[3].y - poly[0].y) < 1e-6 ||
      abs(poly[0].y - poly[1].y) < 1e-6 || abs(poly[1].y - poly[2].y) < 1e-6) {
    return angle;
  }
  int p_bottom_idx = 0;
  int max_y = 0;
  for (unsigned int i = 0; i < poly.size(); ++i) {
    if (poly[i].y > max_y) {
      p_bottom_idx = i;
      max_y = poly[i].y;
    }
  }
  int p_bottom_right_idx = (p_bottom_idx - 1 + 4) % 4;
  angle = std::atan(
      -(poly[p_bottom_idx].y - poly[p_bottom_right_idx].y) /
      (poly[p_bottom_idx].x - poly[p_bottom_right_idx].x));
  if (angle <= 0) {
    return angle;
  }
  if ((angle / CV_PI * 180) > 45) {
    // this point is p2
    return -(CV_PI / 2 - angle);
  } else {
    // this point is p3
    return angle;
  }
}

// rotate image of mask rcnn
void
rotateOp2(const cv::Mat& src, cv::Mat& dst, float angle)
{
  int h = src.rows, w = src.cols;
  float rangle = angle * CV_PI / 180.;
  float nh = std::abs(std::cos(rangle) * h) + std::abs(std::sin(rangle) * w);
  float nw = std::abs(std::sin(rangle) * h) + std::abs(std::cos(rangle) * w);
  cv::Point2f center(nw * 0.5, nh * 0.5);
  auto rot_mat = cv::getRotationMatrix2D(center, angle, 1.);
  cv::Mat m = cv::Mat::zeros(3, 1, CV_64FC1);
  m.at<double>(0, 0) = (nw - w) * 0.5;
  m.at<double>(1, 0) = (nh - h) * 0.5;
  cv::Mat rot_move = rot_mat * m;
  rot_mat.at<double>(0, 2) += rot_move.at<double>(0, 0);
  rot_mat.at<double>(1, 2) += rot_move.at<double>(1, 0);
  // note: original py impl is int(std::ceil(nw))
  cv::warpAffine(
      src, dst, rot_mat, {(int)std::round(nw), (int)std::round(nh)},
      cv::INTER_LANCZOS4);
}

// rotate image of bboxes
void
readjust_bb(
    const cv::Point2f& old_center, const cv::Point2f& new_center,
    const double& theta, std::vector<std::vector<cv::Point2f>>& src)
{
  // theta is angle degree
  double radian = theta * CV_PI / 180.;
  float cosv = std::cos(radian), sinv = std::sin(radian);
  for (auto& v : src) {
    for (int i = 0; i < 4; i++) {
      float x_ = v[i].x - new_center.x, y_ = new_center.y - v[i].y;
      v[i].x = old_center.x + x_ * cosv - y_ * sinv;
      v[i].y = old_center.y - x_ * sinv - y_ * cosv;
    }
    // update by gulixin(210325): v5: points is reordered by cos and sin. no
    // need to reorder_quadrangle_points angin. reorder_quadrangle_points2(v);
  }
}

// shrink batch mat by widhts
void
shrink_batch_mat(const cv::Mat& src, cv::Mat& dst, int new_w)
{
  int n = src.size[0], h = src.size[1], w = src.size[2], c = src.size[3];
  vector<int> new_size = {n, h, new_w, c};
  dst = cv::Mat(src.dims, new_size.data(), CV_32F);
  auto type = c == 3 ? CV_32FC3 : CV_32F;
  for (int i = 0; i < n; i++) {
    cv::Mat src_block(h, w, type, src.data + src.step[0] * i);
    cv::Mat dst_block(h, new_w, type, dst.data + dst.step[0] * i);
    src_block(cv::Rect(0, 0, new_w, h)).copyTo(dst_block);
  }
}

void
crop_rrect(
    const cv::Mat& src, const std::vector<double> points,
    std::vector<cv::Mat> dsts)
{
  int n = points.size() / 8;
  for (int i = 0; i < n; i++) {
    std::vector<cv::Point2f> v;
    for (int j = 0; j < 3; j++) {
      int k1 = i * 8 + j * 2, k2 = i * 8 + j * 2 + 1;
      v.emplace_back((float)points[k1], (float)points[k2]);
    }
    float w = round(l2_norm(v[0], v[1]));
    float h = round(l2_norm(v[1], v[2]));
    vector<cv::Point2f> src_3points{v[0], v[1], v[2]};
    vector<cv::Point2f> dest_3points{{0, 0}, {w, 0}, {w, h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    cv::Mat m;
    cv::warpAffine(src, m, warp_mat, {int(w), int(h)}, cv::INTER_LINEAR);
    dsts.emplace_back(std::move(m));
  }
}


void
reorder_start_point(std::vector<cv::Point2f>& points, float cos, float sin)
{
  auto calc_angle = [](double cos_value, double sin_value) {
    auto norm = sqrt(pow(cos_value, 2) + pow(sin_value, 2));
    auto cos_value_norm = cos_value / norm;
    auto sin_value_norm = sin_value / norm;
    auto cos_angle = acos(cos_value_norm) * 180 / CV_PI;
    auto sin_angle = asin(sin_value_norm) * 180 / CV_PI;
    double angle = 0;
    if (cos_angle <= 90 && sin_angle <= 0) {
      angle = 360 + sin_angle;
    } else if (cos_angle <= 90 && sin_angle > 0) {
      angle = sin_angle;
    } else if (cos_angle > 90 && sin_angle > 0) {
      angle = cos_angle;
    } else if (cos_angle > 90 && sin_angle <= 0) {
      angle = 360 - cos_angle;
    }
    return angle;
  };
  auto norm_angle = [](double v) { return (v < 360 ? v : v - 360); };

  auto cos_value = 2 * cos - 1;
  auto sin_value = 2 * sin - 1;
  auto angle = calc_angle(cos_value, sin_value);
  auto p0 = points[0], p1 = points[1];
  auto vx = p1.x - p0.x;
  auto vy = p1.y - p0.y;
  auto box_angle = calc_angle(vx, vy);
  cv::Mat box_angles =
      (cv::Mat_<double>(1, 4) << box_angle, norm_angle(box_angle + 90),
       norm_angle(box_angle + 180), norm_angle(box_angle + 270));
  cv::Mat diff = cv::abs(box_angles - angle);
  cv::Mat delta_angle;
  cv::hconcat(diff, 360.0 - diff, delta_angle);
  std::vector<double> v(delta_angle);

  // std::cout << "xxxx: " << box_angle <<  "," << angle << "\n";
  // for (auto& v_ : v) {std::cout << "xxxx: " << v_ << "\n";}

  int mini = (int)(std::min_element(v.begin(), v.end()) - v.begin());
  int start = mini % 4;
  points.assign({points[start], points[(start + 1) % 4],
                 points[(start + 2) % 4], points[(start + 3) % 4]});
}

void
refine_box_orientation(std::vector<Point2fList>& points_vec)
{
  refine_box_orientation(points_vec, false);
}

void
refine_box_orientation(std::vector<Point2fList>& points_vec, bool ignore_thres)
{
  // biases: new points order start after lixin's logic corresponding to the
  // reordered points
  const float thres = 0.5;
  std::map<int, int> counter;
  std::vector<int> biases;
  for (auto& points : points_vec) {
    auto idxs = reorder_quadrangle_points(points);
    for (size_t i = 0; i < idxs.size(); i++) {
      if (points[0] == points[idxs[i]]) {
        biases.push_back(i);
        counter[i] = counter.find(i) == counter.end() ? 1 : counter[i] + 1;
        break;
      }
    }
  }

  int main_bias = -1;
  int max_count = 0;
  for (const auto kv : counter) {
    if (kv.second > max_count) {
      max_count = kv.second;
      main_bias = kv.first;
    }
  }

  for (size_t i = 0; i < points_vec.size(); i++) {
    auto& v = points_vec[i];
    if (biases[i] != main_bias &&
        (ignore_thres || l2_norm(v[0], v[1]) / l2_norm(v[1], v[2]) < thres)) {
      auto new_bias = main_bias - biases[i];
      v.assign({v[(new_bias + 4) % 4], v[(new_bias + 5) % 4],
                v[(new_bias + 6) % 4], v[(new_bias + 7) % 4]});
    }
  }
}

// face utils
void
preprocess_image(const cv::Mat& src, cv::Mat& dst, float scale)
{
  // todo: add process logit
  int h = src.rows, w = src.cols;
  int new_h = int(h * scale), new_w = int(w * scale);
  cv::Mat tmp;
  resizeOp(src, tmp, new_w, new_h);
  tmp.convertTo(dst, CV_32F);
  dst = (dst - 127.5) / 128.;
}

void
generate_bbox(
    const cv::Mat& cls_cls_map, const cv::Mat& reg,
    std::vector<std::vector<float>>& bboxes, float current_scale,
    float pnet_threshold)
{
  float stride = 2.;
  float cellsize = 12.;

  // boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
  for (int i = 0; i < cls_cls_map.size[0]; i++) {
    for (int j = 0; j < cls_cls_map.size[1]; j++) {
      if (cls_cls_map.at<float>(i, j, 1) > pnet_threshold) {
        std::vector<float> bbox;
        bbox.push_back(std::round(stride * j / current_scale));
        bbox.push_back(std::round(stride * i / current_scale));
        bbox.push_back(std::round((stride * j + cellsize) / current_scale));
        bbox.push_back(std::round((stride * i + cellsize) / current_scale));
        bbox.push_back(cls_cls_map.at<float>(i, j, 1));
        bbox.push_back(reg.at<float>(i, j, 0));
        bbox.push_back(reg.at<float>(i, j, 1));
        bbox.push_back(reg.at<float>(i, j, 2));
        bbox.push_back(reg.at<float>(i, j, 3));
        bboxes.push_back(bbox);
      }
    }
  }
}

void
convert_to_square(
    const std::vector<std::vector<float>>& src,
    std::vector<std::vector<float>>& dst)
{
  for (size_t i = 0; i < src.size(); i++) {
    std::vector<float> points;
    float h = src[i][3] - src[i][1] + 1;
    float w = src[i][2] - src[i][0] + 1;
    float max_side = std::max(h, w);
    float x1 = src[i][0] + w * 0.5 - max_side * 0.5;
    float y1 = src[i][1] + h * 0.5 - max_side * 0.5;
    points.push_back(float(std::round(x1)));
    points.push_back(float(std::round(y1)));
    points.push_back(float(std::round(x1 + max_side - 1)));
    points.push_back(float(std::round(y1 + max_side - 1)));
    points.push_back(src[i][4]);
    dst.push_back(points);
  }
}

void
rnet_pad(
    const std::vector<std::vector<float>>& src, float w, float h,
    std::vector<PadRnet>& dst)
{
  for (size_t i = 0; i < src.size(); i++) {
    float tmph = src[i][3] - src[i][1] + 1;
    float tmpw = src[i][2] - src[i][0] + 1;
    float dx = 0;
    float dy = 0;
    float edx = tmpw - 1;
    float edy = tmph - 1;
    float x = src[i][0];
    float y = src[i][1];
    float ex = src[i][2];
    float ey = src[i][3];
    if (ex > w - 1) {
      edx = tmpw + w - 2 - ex;
      ex = w - 1;
    }
    if (ey > h - 1) {
      edy = tmph + h - 2 - ey;
      ey = h - 1;
    }
    if (x < 0) {
      dx = 0 - x;
      x = 0;
    }
    if (y < 0) {
      dy = 0 - y;
      y = 0;
    }
    PadRnet pad;
    pad.dy = int(dy);
    pad.edy = int(edy);
    pad.dx = int(dx);
    pad.edx = int(edx);
    pad.y = int(y);
    pad.ey = int(ey);
    pad.x = int(x);
    pad.ex = int(ex);
    pad.tmpw = int(tmpw);
    pad.tmph = int(tmph);
    dst.push_back(pad);
  }
}

void
calibrate_box(
    const std::vector<std::vector<float>>& src_bbs,
    const std::vector<std::vector<float>>& regs,
    std::vector<std::vector<float>>& dst_bbs)
{
  for (size_t i = 0; i < src_bbs.size(); i++) {
    std::vector<float> bb;
    float h = src_bbs[i][3] - src_bbs[i][1] + 1;
    float w = src_bbs[i][2] - src_bbs[i][0] + 1;
    float x1 = src_bbs[i][0] + w * regs[i][0];
    float y1 = src_bbs[i][1] + h * regs[i][1];
    float x2 = src_bbs[i][2] + w * regs[i][2];
    float y2 = src_bbs[i][3] + h * regs[i][3];
    bb.push_back(x1);
    bb.push_back(y1);
    bb.push_back(x2);
    bb.push_back(y2);
    bb.push_back(src_bbs[i][4]);
    dst_bbs.push_back(bb);
  }
}

void
gather_all_faces_into_batch(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst)
{
  int bbs_nums = bboxes.size();
  std::vector<int> size = {bbs_nums, 48, 48};
  auto mats = cv::Mat(size.size(), size.data(), CV_8UC3);
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }

  for (int i = 0; i < bbs_nums; i++) {
    float w = bboxes[i][3] - bboxes[i][1] + 1;
    float h = bboxes[i][2] - bboxes[i][0] + 1;
    float max_size = std::max(h, w);
    float square_box_xmin = bboxes[i][1] + 0.5 * w - 0.5 * max_size;
    float square_box_ymin = bboxes[i][0] + 0.5 * h - 0.5 * max_size;
    float square_box_xmax = square_box_xmin + max_size - 1;
    float square_box_ymax = square_box_ymin + max_size - 1;
    int x_w = int(square_box_xmax) + 1 - int(square_box_xmin);
    int y_h = int(square_box_ymax) + 1 - int(square_box_ymin);
    // std::cout << "img.rows: " << img.rows << ", img.cols: " << img.cols <<
    // "\n"; std::cout << int(square_box_xmin) << ", " << int(square_box_ymin)
    // << ", "
    //           << x_w << ", " << y_h << std::endl;
    auto rect =
        img(cv::Rect(int(square_box_xmin), int(square_box_ymin), x_w, y_h));
    cv::Mat rect_resize;
    // resizeOp(rect, rect_resize, 48, 48);
    cv::resize(rect, rect_resize, {48, 48}, 0, 0, cv::INTER_AREA);
    cv::Mat rect_resize_(size[1], size[2], CV_8UC3, mats.data + step0 * i * 3);
    rect_resize.copyTo(rect_resize_);
  }
  cv::Mat mats_;
  mats.convertTo(mats_, CV_32F);
  mats_.copyTo(dst);
}

void
convert_to_square_feature(
    const std::vector<std::vector<float>>& src,
    std::vector<std::vector<float>>& dst)
{
  for (size_t i = 0; i < src.size(); i++) {
    std::vector<float> points;
    float w = src[i][3] - src[i][1] + 1;
    float h = src[i][2] - src[i][0] + 1;
    float max_side = std::max(h, w);
    float x1 = src[i][0] + h * 0.5 - max_side * 0.5;
    float y1 = src[i][1] + w * 0.5 - max_side * 0.5;
    points.push_back(float(std::round(x1)));
    points.push_back(float(std::round(y1)));
    points.push_back(float(std::round(x1 + max_side - 1)));
    points.push_back(float(std::round(y1 + max_side - 1)));
    points.push_back(src[i][4]);
    dst.push_back(points);
  }
}

void
pad_feature(
    const std::vector<std::vector<float>>& src, float w, float h,
    std::vector<PadRnet>& dst)
{
  for (size_t i = 0; i < src.size(); i++) {
    float tmph = src[i][2] - src[i][0] + 1;
    float tmpw = src[i][3] - src[i][1] + 1;
    float dx = 0;
    float dy = 0;
    float edx = tmpw - 1;
    float edy = tmph - 1;
    float x = src[i][1];
    float y = src[i][0];
    float ex = src[i][3];
    float ey = src[i][2];
    if (ex > w - 1) {
      edx = tmpw + w - 2 - ex;
      ex = w - 1;
    }
    if (ey > h - 1) {
      edy = tmph + h - 2 - ey;
      ey = h - 1;
    }
    if (x < 0) {
      dx = 0 - x;
      x = 0;
    }
    if (y < 0) {
      dy = 0 - y;
      y = 0;
    }
    PadRnet pad;
    pad.dy = int(dy);
    pad.edy = int(edy);
    pad.dx = int(dx);
    pad.edx = int(edx);
    pad.y = int(y);
    pad.ey = int(ey);
    pad.x = int(x);
    pad.ex = int(ex);
    pad.tmpw = int(tmpw);
    pad.tmph = int(tmph);
    dst.push_back(pad);
  }
}

void
gather_all_faces_into_batch_v2(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst)
{
  float h = img.rows, w = img.cols;
  std::vector<std::vector<float>> dets;
  convert_to_square_feature(bboxes, dets);
  std::vector<PadRnet> pads;
  pad_feature(dets, w, h, pads);
  int bbs_nums = dets.size();
  std::vector<int> size = {bbs_nums, 48, 48};
  auto mats = cv::Mat(size.size(), size.data(), CV_32FC3, cv::Scalar(0));
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }
  for (int i = 0; i < bbs_nums; i++) {
    cv::Mat mat_tmp(pads[i].tmph, pads[i].tmpw, CV_8UC3);
    for (int idx_r = pads[i].dy, idx_r_ = pads[i].y; idx_r < pads[i].edy + 1;
         idx_r++, idx_r_++) {
      for (int idx_c = pads[i].dx, idx_c_ = pads[i].x; idx_c < pads[i].edx + 1;
           idx_c++, idx_c_++) {
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[0] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[0];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[1] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[1];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[2] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[2];
      }
    }
    cv::Mat mat_resize0, mat_resize;
    resizeOp(mat_tmp, mat_resize0, 48, 48);
    mat_resize0.convertTo(mat_resize, CV_32F);

    // mat_resize = (mat_resize - 127.5) / 128.;
    cv::Mat mat_resize_(
        size[1], size[2], CV_32FC3, mats.data + step0 * i * 4 * 3);
    mat_resize.copyTo(mat_resize_);
  }
  mats.copyTo(dst);
}

void
gather_all_faces_into_batch_v3(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst, cv::Mat& max_face)
{
  float h = img.rows, w = img.cols;
  std::vector<std::vector<float>> dets;
  convert_to_square_feature(bboxes, dets);
  std::vector<PadRnet> pads;
  pad_feature(dets, w, h, pads);
  int bbs_nums = dets.size();
  std::vector<int> size = {bbs_nums, 48, 48};
  auto mats = cv::Mat(size.size(), size.data(), CV_32FC3, cv::Scalar(0));
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }
  for (int i = 0; i < bbs_nums; i++) {
    cv::Mat mat_tmp(pads[i].tmph, pads[i].tmpw, CV_8UC3);
    for (int idx_r = pads[i].dy, idx_r_ = pads[i].y; idx_r < pads[i].edy + 1;
         idx_r++, idx_r_++) {
      for (int idx_c = pads[i].dx, idx_c_ = pads[i].x; idx_c < pads[i].edx + 1;
           idx_c++, idx_c_++) {
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[0] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[0];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[1] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[1];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[2] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[2];
      }
    }
    cv::Mat mat_resize0, mat_resize;
    resizeOp(mat_tmp, mat_resize0, 48, 48);
    mat_resize0.convertTo(mat_resize, CV_32F);

    // mat_resize = (mat_resize - 127.5) / 128.;
    cv::Mat mat_resize_(
        size[1], size[2], CV_32FC3, mats.data + step0 * i * 4 * 3);
    mat_resize.copyTo(mat_resize_);
    if (i == 0) {
      mat_tmp.copyTo(max_face);
    }
  }
  mats.copyTo(dst);
}

void
gather_all_faces_into_batch_v4(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst, std::vector<cv::Mat>& mat_list)
{
  float h = img.rows, w = img.cols;
  std::vector<std::vector<float>> dets;
  convert_to_square_feature(bboxes, dets);
  std::vector<PadRnet> pads;
  pad_feature(dets, w, h, pads);
  int bbs_nums = dets.size();
  std::vector<int> size = {bbs_nums, 48, 48};
  auto mats = cv::Mat(size.size(), size.data(), CV_32FC3, cv::Scalar(0));
  int step0 = 1;
  for (unsigned int j = 1; j < size.size(); j++) {
    step0 *= size[j];
  }
  for (int i = 0; i < bbs_nums; i++) {
    cv::Mat mat_tmp(pads[i].tmph, pads[i].tmpw, CV_8UC3);
    for (int idx_r = pads[i].dy, idx_r_ = pads[i].y; idx_r < pads[i].edy + 1;
         idx_r++, idx_r_++) {
      for (int idx_c = pads[i].dx, idx_c_ = pads[i].x; idx_c < pads[i].edx + 1;
           idx_c++, idx_c_++) {
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[0] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[0];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[1] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[1];
        mat_tmp.at<cv::Vec3b>(idx_r, idx_c)[2] =
            img.at<cv::Vec3b>(idx_r_, idx_c_)[2];
      }
    }
    cv::Mat mat_resize0, mat_resize;
    resizeOp(mat_tmp, mat_resize0, 48, 48);
    mat_resize0.convertTo(mat_resize, CV_32F);

    // mat_resize = (mat_resize - 127.5) / 128.;
    cv::Mat mat_resize_(
        size[1], size[2], CV_32FC3, mats.data + step0 * i * 4 * 3);
    mat_resize.copyTo(mat_resize_);
    mat_list.emplace_back(mat_tmp);
  }
  mats.copyTo(dst);
}

void
transform_keypoints2image_coor(
    const std::vector<std::vector<float>>& bboxes,
    std::vector<std::vector<float>>& landmark_list)
{
  int bbs_nums = bboxes.size();
  for (int i = 0; i < bbs_nums; i++) {
    float w_tmp = bboxes[i][3] - bboxes[i][1] + 1;
    float h_tmp = bboxes[i][2] - bboxes[i][0] + 1;
    landmark_list[i][0] = w_tmp * landmark_list[i][0] + bboxes[i][1] - 1;
    landmark_list[i][2] = w_tmp * landmark_list[i][2] + bboxes[i][1] - 1;
    landmark_list[i][4] = w_tmp * landmark_list[i][4] + bboxes[i][1] - 1;
    landmark_list[i][6] = w_tmp * landmark_list[i][6] + bboxes[i][1] - 1;
    landmark_list[i][8] = w_tmp * landmark_list[i][8] + bboxes[i][1] - 1;
    landmark_list[i][1] = h_tmp * landmark_list[i][1] + bboxes[i][0] - 1;
    landmark_list[i][3] = h_tmp * landmark_list[i][3] + bboxes[i][0] - 1;
    landmark_list[i][5] = h_tmp * landmark_list[i][5] + bboxes[i][0] - 1;
    landmark_list[i][7] = h_tmp * landmark_list[i][7] + bboxes[i][0] - 1;
    landmark_list[i][9] = h_tmp * landmark_list[i][9] + bboxes[i][0] - 1;
  }
}

cv::Mat
meanAxis0(const cv::Mat& src)
{
  int num = src.rows;
  int dim = src.cols;
  // x1 y1
  // x2 y2
  cv::Mat output(1, dim, CV_32F);
  for (int i = 0; i < dim; i++) {
    float sum = 0;
    for (int j = 0; j < num; j++) {
      sum += src.at<float>(j, i);
    }
    output.at<float>(0, i) = sum / num;
  }
  return output;
}

cv::Mat
elementwiseMinus(const cv::Mat& A, const cv::Mat& B)
{
  cv::Mat output(A.rows, A.cols, A.type());
  assert(B.cols == A.cols);
  if (B.cols == A.cols) {
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
      }
    }
  }
  return output;
}

cv::Mat
varAxis0(const cv::Mat& src)
{
  cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
  cv::multiply(temp_, temp_, temp_);
  return meanAxis0(temp_);
}

int
MatrixRank(cv::Mat M)
{
  cv::Mat w, u, vt;
  cv::SVD::compute(M, w, u, vt);
  cv::Mat1b nonZeroSingularValues = w > 0.0001;
  int rank = countNonZero(nonZeroSingularValues);
  return rank;
}

cv::Mat
similarTransform(cv::Mat src, cv::Mat dst)
{
  int num = src.rows;
  int dim = src.cols;
  cv::Mat src_mean = meanAxis0(src);
  cv::Mat dst_mean = meanAxis0(dst);
  cv::Mat src_demean = elementwiseMinus(src, src_mean);
  cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
  cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
  cv::Mat d(dim, 1, CV_32F);
  d.setTo(1.0f);
  if (cv::determinant(A) < 0) {
    d.at<float>(dim - 1, 0) = -1;
  }
  cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
  cv::Mat U, S, V;
  cv::SVD::compute(A, S, U, V);

  // the SVD function in opencv differ from scipy .
  int rank = MatrixRank(A);
  if (rank == 0) {
    assert(rank == 0);
  } else if (rank == dim - 1) {
    if (cv::determinant(U) * cv::determinant(V) > 0) {
      T.rowRange(0, dim).colRange(0, dim) = U * V;
    } else {
      //   s = d[dim - 1]
      //   d[dim - 1] = -1
      //   T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
      //   d[dim - 1] = s
      int s = d.at<float>(dim - 1, 0) = -1;
      d.at<float>(dim - 1, 0) = -1;

      T.rowRange(0, dim).colRange(0, dim) = U * V;
      cv::Mat diag_ = cv::Mat::diag(d);
      cv::Mat twp = diag_ * V;  // np.dot(np.diag(d), V.T)
      cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
      cv::Mat C = B.diag(0);
      T.rowRange(0, dim).colRange(0, dim) = U * twp;
      d.at<float>(dim - 1, 0) = s;
    }
  } else {
    cv::Mat diag_ = cv::Mat::diag(d);
    cv::Mat twp = diag_ * V.t();  // np.dot(np.diag(d), V.T)
    cv::Mat res = U * twp;        // U
    T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
  }
  cv::Mat var_ = varAxis0(src_demean);
  float val = cv::sum(var_).val[0];
  cv::Mat res;
  cv::multiply(d, S, res);
  float scale = 1.0 / val * cv::sum(res).val[0];
  T.rowRange(0, dim).colRange(0, dim) =
      -T.rowRange(0, dim).colRange(0, dim).t();
  cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim);  // T[:dim, :dim]
  cv::Mat temp2 = src_mean.t();                         // src_mean.T
  cv::Mat temp3 = temp1 * temp2;  // np.dot(T[:dim, :dim], src_mean.T)
  cv::Mat temp4 = scale * temp3;
  T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
  T.rowRange(0, dim).colRange(0, dim) *= scale;
  return T;
}

void
regcong_preprocess(
    const cv::Mat& img, const std::vector<float>& bbox,
    const std::vector<float>& landmark, cv::Mat& warped)
{
  std::vector<int> image_size = {112, 112};
  if (landmark.size() != 0) {
    float src_value[5][2] = {{30.2946f + 8.0f, 51.6963f},
                             {65.5318f + 8.0f, 51.5014f},
                             {48.0252f + 8.0f, 71.7366f},
                             {33.5493f + 8.0f, 92.3655f},
                             {62.7299f + 8.0f, 92.2041f}};  // +8.0f for 112*112
    cv::Mat src(5, 2, CV_32FC1, src_value);
    float detect[5][2] = {{landmark[0], landmark[1]},
                          {landmark[2], landmark[3]},
                          {landmark[4], landmark[5]},
                          {landmark[6], landmark[7]},
                          {landmark[8], landmark[9]}};
    cv::Mat dst(5, 2, CV_32FC1, detect);

    cv::Mat M =
        similarTransform(dst, src);  // skimage.transform.SimilarityTransform
    auto M_new = M(cv::Rect(0, 0, 3, 2));
    for (int s = 0; s < M_new.rows; s++) {
      for (int k = 0; k < M_new.cols; k++) {
        // 对M进行截断取小数点后两位
        M_new.at<float>(s, k) = int(M_new.at<float>(s, k) * 100) / 100.f;
      }
    }
    cv::warpAffine(img, warped, M_new, cv::Size(112, 112));
  }
}

void
recong_normalize(const cv::Mat& src, cv::Mat& dst)
{
  // int src_rows = src.rows;
  std::vector<float> rows_sums;
  for (int i = 0; i < src.rows; i++) {
    float sum = 0;
    for (int j = 0; j < src.cols; j++) {
      sum += src.at<float>(i, j) * src.at<float>(i, j);
    }
    rows_sums.push_back(std::sqrt(sum));
  }
  cv::Mat m(src.rows, src.cols, CV_32FC1);
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      m.at<float>(i, j) = src.at<float>(i, j) / rows_sums[i];
    }
  }
  m.copyTo(dst);
}

std::string
dec2hex(const uint64_t dec_num, const int width)
{
  std::stringstream ioss;       //定义字符串流
  std::string s_temp;           //存放转化后字符
  ioss << std::hex << dec_num;  //以十六制形式输出
  ioss >> s_temp;
  if ((size_t)width > s_temp.size()) {
    std::string s_0(width - s_temp.size(), '0');  //位数不够则补0
    s_temp = s_0 + s_temp;                        //合并
  }

  std::string s =
      s_temp.substr(s_temp.length() - width, s_temp.length());  //取右width位
  return s;
}

std::string
ImageDHash(const cv::Mat image)
{
  uint64_t hashRow = 0;  // 用于保存hash值
  uint64_t hashCol = 0;  //
  cv::Mat imageGray;     // 转换后的灰度图像
  cv::Mat imageFinger;   // 缩放后的8x8的指纹图像
  int fingerSize = 8;    // 指纹图像的大小

  // bgr -> gray
  if (3 == image.channels()) {
    cv::cvtColor(image, imageGray, CV_BGR2GRAY);
  } else {
    imageGray = image.clone();
  }

  cv::resize(
      imageGray, imageFinger,
      cv::Size(fingerSize + 1, fingerSize + 1));  // 图像缩放
  for (int i = 0; i < fingerSize; i++) {
    for (int j = 0; j < fingerSize; j++) {
      if (imageFinger.at<unsigned char>(i, j) <
          imageFinger.at<unsigned char>(i, j + 1)) {
        hashRow = (hashRow << 1) + 1;
      } else {
        hashRow = hashRow << 1;
      }
      if (imageFinger.at<unsigned char>(i, j) <
          imageFinger.at<unsigned char>(i + 1, j)) {
        hashCol = (hashCol << 1) + 1;
      } else {
        hashCol = hashCol << 1;
      }
    }
  }
  auto hexRow = dec2hex(hashRow, 16);
  auto hexCol = dec2hex(hashCol, 16);

  return hexRow + hexCol;
}

void
calculate_mean_angle(
    const cv::Mat& cos_map, const cv::Mat& sin_map, const cv::Mat& yx_text,
    const std::vector<cv::Point2f> points_tmp, float& sin, float& cos)
{
  std::vector<float> ploy_coses, ploy_sines;
  int n = 0;
  for (int i = 0; i < cos_map.size[0]; i++) {
    float cos_value = cos_map.at<float>(i, 0);
    float sin_value = sin_map.at<float>(i, 0);
    int px = yx_text.at<int>(i, 1), py = yx_text.at<int>(i, 0);
    float a = (points_tmp[1].x - points_tmp[0].x) * (py - points_tmp[0].y) -
              (points_tmp[1].y - points_tmp[0].y) * (px - points_tmp[0].x);
    float b = (points_tmp[2].x - points_tmp[1].x) * (py - points_tmp[1].y) -
              (points_tmp[2].y - points_tmp[1].y) * (px - points_tmp[1].x);
    float c = (points_tmp[3].x - points_tmp[2].x) * (py - points_tmp[2].y) -
              (points_tmp[3].y - points_tmp[2].y) * (px - points_tmp[2].x);
    float d = (points_tmp[0].x - points_tmp[3].x) * (py - points_tmp[3].y) -
              (points_tmp[0].y - points_tmp[3].y) * (px - points_tmp[3].x);
    if ((a > 0 && b > 0 && c > 0 && d > 0) ||
        (a < 0 && b < 0 && c < 0 && d < 0)) {
      ploy_coses.emplace_back(std::move(cos_value));
      ploy_sines.emplace_back(std::move(sin_value));
    }
  }
  n = ploy_sines.size();
  sin = std::accumulate(ploy_sines.begin(), ploy_sines.end(), 0.) / n;
  cos = std::accumulate(ploy_coses.begin(), ploy_coses.end(), 0.) / n;
}

void
restore_rectangle(
    const cv::Mat& xy_test, const cv::Mat& geometry, cv::Mat& decoded_bboxes)
{
  int n = xy_test.size[0];
  auto func = [&xy_test, &geometry, &decoded_bboxes](int i) {
    float angle = geometry.at<float>(i, 4);
    float height = geometry.at<float>(i, 0) + geometry.at<float>(i, 2);
    float width = geometry.at<float>(i, 1) + geometry.at<float>(i, 3);
    std::vector<std::vector<float>> src;
    cv::Mat dst(5, 2, CV_32F);
    if (angle >= 0) {
      src = {{0, -height},
             {width, -height},
             {width, 0},
             {0, 0},
             {geometry.at<float>(i, 3), -geometry.at<float>(i, 2)}};
      for (unsigned j = 0; j < src.size(); ++j) {
        dst.at<float>(j, 0) =
            src[j][0] * (+cos(angle)) + src[j][1] * sin(angle);
        dst.at<float>(j, 1) =
            src[j][0] * (-sin(angle)) + src[j][1] * cos(angle);
      }
    } else {
      src = {{-width, -height},
             {0, -height},
             {0, 0},
             {-width, 0},
             {-geometry.at<float>(i, 1), -geometry.at<float>(i, 2)}};
      for (unsigned j = 0; j < src.size(); ++j) {
        dst.at<float>(j, 0) =
            src[j][0] * cos(-angle) + src[j][1] * (-sin(-angle));
        dst.at<float>(j, 1) =
            src[j][0] * sin(-angle) + src[j][1] * (+cos(-angle));
      }
    }
    for (int j = 0; j < dst.size[0] - 1; ++j) {
      dst.at<float>(j, 0) += xy_test.at<int>(i, 1) * 4 - dst.at<float>(4, 0);
      dst.at<float>(j, 1) += xy_test.at<int>(i, 0) * 4 - dst.at<float>(4, 1);
    }
    cv::Mat dstMat = dst(cv::Rect(0, 0, 2, 4));
    auto* ptr = decoded_bboxes.ptr<float>(i);
    cv::Mat tmp(4, 2, CV_32F, ptr);
    dstMat.copyTo(tmp);
  };

  // ThreadPool& tp = nn_thread_pool();
  int batch_num = ceil(float(n) / 20);
  int threads = ceil(n / float(batch_num));
  // std::vector<BoolFuture> rets(threads);
  for (int i = 0; i < threads; ++i) {
    int start = i * batch_num, end = std::min(int((i + 1) * batch_num), n) - 1;

    [&func, &xy_test, &geometry, &decoded_bboxes, start, end]() {
      for (int j = start; j <= end; j++) {
        func(j);
      }
      return true;
    }();
    // rets[i] =
    //     tp.enqueue([&func, &xy_test, &geometry, &decoded_bboxes, start,
    //     end]() {
    //       for (int j = start; j <= end; j++) {
    //         func(j);
    //       }
    //       return true;
    //     });
  }
  // for (int i = 0; i < threads; i++) {
  //   rets[i].get();
  // }
  decoded_bboxes = decoded_bboxes.reshape(1, {n, 4, 2});
}

// 20201027 hjt modification
void
getRRectRoisWithPaddingOp5(
    const cv::Mat& src, const cv::Mat& bbs, std::vector<cv::Mat>& rois)
{
  // src: cv::Matcomplete image
  // bbs: cv::Mat bboxes
  // rois : vector to save cropped subimages

  int n = bbs.size[0];
  cv::Mat tmp_bbs(n, 4, CV_32FC2, bbs.data);
  cv::Mat_<cv::Point2f> bbs_(tmp_bbs);

  for (int i = 0; i < n; i++) {
    std::vector<cv::Point2f> v;
    for (int j = 0; j < 4; j++) {
      v.emplace_back(bbs_(i, j));
    }
    // v <cv::Point2f> [pt0, pt1, pt2, pt3]
    float w = round(l2_norm(v[0], v[1]));
    float h = round(l2_norm(v[1], v[2]));

    std::vector<cv::Point2f> src_3points{v[0], v[1], v[2]};
    std::vector<cv::Point2f> dest_3points{{0, 0}, {w, 0}, {w, h}};
    cv::Mat warp_mat = cv::getAffineTransform(src_3points, dest_3points);
    cv::Mat m;
    cv::warpAffine(src, m, warp_mat, {int(w), int(h)}, cv::INTER_LINEAR);
    rois.emplace_back(std::move(m));
  }
}

// added by hf, 2023.03.19
TRITONSERVER_Error*
DecodeImgFromB64(
    absl::string_view data, cv::Mat& img, int imread_flag = cv::IMREAD_COLOR)
{
  std::string image_raw_bytes;
  if (!absl::Base64Unescape(data, &image_raw_bytes)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "failed to b64decode");
  }
  int n = image_raw_bytes.length();
  auto bytes_mat =
      cv::Mat(n, 1, CV_8U, const_cast<char*>(image_raw_bytes.data()));

  try {
    img = cv::imdecode(bytes_mat, imread_flag);
  }
  catch (cv::Exception& e) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.err.c_str());
  }
  // Check images
  if (img.cols <= 0 || img.rows <= 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "image shape is abnormal");
  }

  // Check images
  if (img.cols >= 0x7fff || img.rows >= 0x7fff) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "image scale is large than 0x7fff");
  }

  return nullptr;
}

}}  // namespace dataelem::alg
