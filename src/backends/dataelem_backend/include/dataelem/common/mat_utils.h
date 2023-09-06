#ifndef DATAELEM_COMMON_MAT_UTILS_H_
#define DATAELEM_COMMON_MAT_UTILS_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "dataelem/framework/types.h"

namespace dataelem { namespace alg {

using namespace std;
constexpr int THREAD_NUM_FOR_OPENCV = 20;

// opencv type traits
template <class T>
struct RawTypeToCvType {
};

template <int VALUE>
struct CvTypeToRawType {
};

#define CV_MATCH_TYPE_AND_ENUM(TYPE, ENUM) \
  template <>                              \
  struct RawTypeToCvType<TYPE> {           \
    static int v() { return ENUM; }        \
    static constexpr int value = ENUM;     \
  };                                       \
  template <>                              \
  struct CvTypeToRawType<ENUM> {           \
    typedef TYPE Type;                     \
  }

// utility functions
inline float
f_top_norm(float v)
{
  return max(floor(v) - 1.0f, 0.0f);
}

inline float
f_bottom_norm(float v, float z)
{
  return min(floor(v) + 1.0f, z);
}

inline float
CvRound(float v)
{
  float fv = round(v);
  return abs(fv - v) < 1e-4 ? fv : floor(v) + 1.0f;
}

inline int
CvRoundI(float v)
{
  float fv = round(v);
  return abs(fv - v) < 1e-4 ? int(fv) : int(floor(v)) + 1;
}

inline float
l2_distance(const cv::Point2f& a1, const cv::Point2f& a2)
{
  return pow(a1.x - a2.x, 2) + pow(a1.y - a2.y, 2);
}

inline float
l2_norm(const cv::Point2f& a1, const cv::Point2f& a2)
{
  return sqrt(pow(a1.x - a2.x, 2) + pow(a1.y - a2.y, 2));
}

inline void
imwriteOp(const cv::Mat& src, const string& filename)
{
  // by default write in png format
  // https://docs.opencv.org/2.4/modules/highgui/doc/
  //         reading_and_writing_images_and_video.html
  vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(3);
  cv::imwrite(filename, src, compression_params);
}

inline cv::Mat
imreadOp(const string& filename, int flags)
{
  return cv::imread(filename, flags);
}

// decode image from binary string op
inline cv::Mat
imdecodeOp(const string& s, int flags)
{
  // flags: IMREAD_UNCHANGED:-1, IMREAD_GRAYSCALE:0, IMREAD_COLOR:1
  //        IMREAD_ANYDEPTH:2, IMREAD_ANYCOLOR: 4
  vector<unsigned char> vdata(s.begin(), s.end());
  return cv::imdecode(vdata, flags);
}

inline string
imencodeOp(const cv::Mat& src, const string& fmt = ".png")
{
  vector<unsigned char> data;
  cv::imencode(fmt, src, data);
  return string(data.begin(), data.end());
}

// image color format transform
inline void
bgr2grayOp(const cv::Mat& src, cv::Mat& dest)
{
  cv::cvtColor(src, dest, CV_BGR2GRAY);
}

inline void
rgb2grayOp(const cv::Mat& src, cv::Mat& dest)
{
  cv::cvtColor(src, dest, CV_RGB2GRAY);
}

inline void
rgb2bgrOp(const cv::Mat& src, cv::Mat& dest)
{
  cv::cvtColor(src, dest, CV_RGB2BGR);
}

inline void
bgr2rgbOp(const cv::Mat& src, cv::Mat& dest)
{
  cv::cvtColor(src, dest, CV_BGR2RGB);
}

inline cv::Mat
BGR2RGB(cv::Mat& img)
{
  cv::Mat image(img.rows, img.cols, img.type());
  for (int i = 0; i < img.rows; ++i) {
    cv::Vec3b* p1 = img.ptr<cv::Vec3b>(i);
    cv::Vec3b* p2 = image.ptr<cv::Vec3b>(i);
    for (int j = 0; j < img.cols; ++j) {
      p2[j][2] = p1[j][0];
      p2[j][1] = p1[j][1];
      p2[j][0] = p1[j][2];
    }
  }
  return image;
}

// resize image
inline void
resizeOp(const cv::Mat& src, cv::Mat& dest, const int w, const int h)
{
  cv::resize(src, dest, {w, h}, 0, 0, cv::INTER_LINEAR);
}

// scale image with same coef
inline void
scaleOp(const cv::Mat& src, cv::Mat& dst, const float coef)
{
  cv::resize(src, dst, cv::Size(), coef, coef, cv::INTER_LINEAR);
}

inline float
calcCoefOp(const cv::Mat& src, int scale_max, int scale_min)
{
  float coef = std::min(
      static_cast<float>(scale_max) / std::max(src.rows, src.cols),
      static_cast<float>(scale_min) / std::min(src.rows, src.cols));
  return coef;
}

inline float
calcCoefOp(const cv::Mat& src, int scale_max)
{
  return static_cast<float>(scale_max) / std::max(src.rows, src.cols);
}

// rotate 90 degree
inline void
rotate90Op(const cv::Mat& src, cv::Mat& dest, int rotflag)
{
  // 0:ROTATE_90_CLOCKWISE, 1:ROTATE_180, 2:ROTATE_90_COUNTERCLOCKWISE
  if (rotflag == 0) {
    cv::transpose(src, dest);
    cv::flip(dest, dest, 1);
  } else if (rotflag == 1) {
    cv::flip(src, dest, -1);
  } else if (rotflag == 2) {
    cv::transpose(src, dest);
    cv::flip(dest, dest, 0);
  }
}

inline void
get180RotatedBoundingBox(cv::Rect& r, int w, int h)
{
  int x1 = r.x + r.width, y1 = r.y + r.height;
  r = {w - x1, h - y1, r.width, r.height};
}

// create shape(w,h) mat
inline cv::Mat
create_random_mat(int w, int h, int type, float low, float high)
{
  cv::Mat m = cv::Mat(h, w, type);
  cv::randu(m, cv::Scalar(low), cv::Scalar(high));
  return m;
}

inline float
point_norm(const cv::Point2f& point_xy)
{
  return sqrt(pow(point_xy.x, 2) + pow(point_xy.y, 2));
}

// normalize image
void imNormOp(const cv::Mat& src, cv::Mat& dest, cv::Vec3f mean, cv::Vec3f std);

// rotate image
void rotateOp(const cv::Mat& src, cv::Mat& dest, float angle);

// normalize positions of points
vector<size_t> reorder_and_nique_points(const vector<cv::Point2f>& v);

// normalize point and store the ordered axis value
vector<size_t> reorder_quadrangle_points(const vector<cv::Point2f>& v);
void nique_points(
    const vector<cv::Point2f>& v, vector<size_t>& idx, float thres = 0.4);

template <typename T>
void
reorder_quadrangle_points2(vector<T>& v)
{
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
  // make sure p0.y is small then p3.y or (p0.x==p3.y && p0.x < p3.x)
  if (v[p0].y > v[p3].y) {
    swap(p0, p3);
  }
  // make sure p1.y is small then p2.y
  if (v[p1].y > v[p2].y) {
    swap(p1, p2);
  }
  v.assign({v[p0], v[p1], v[p2], v[p3]});
}

// get rotated rectangle roi
void getRRectRoiOp(
    const cv::Mat& src, cv::Mat& dest, const vector<cv::Point2f>& src_points,
    int fixed_h);

// get rotated rectangle roi
void getRRectRoiOp(
    const cv::Mat& src, cv::Mat& dst, const vector<cv::Point2f>& points,
    int new_w, int new_h);

// get rotated rectangle rois
void getRRectRoisWithPaddingOp(
    const cv::Mat& src, vector<cv::Mat>& dest, vector<float>& dest_widths,
    const vector<vector<cv::Point2f>>& points_vec, int fixed_h);

void getRRectRoisWithPaddingOp2(
    const cv::Mat& src, cv::Mat& dst, cv::Mat& widths,
    const vector<vector<cv::Point2f>>& points_vec, int fixed_h,
    vector<float>& bbs, float nique_threshold);

void getRRectRoisWithPaddingOp3(
    const cv::Mat& src, const cv::Mat& bbs, int batch_size, int fixed_h,
    float nique_threshold, int output_channels, bool use_min_width_limit,
    int device_count, PairMatList& dst, std::vector<int>& dst_indexes);

void getRRectRoisWithPaddingOp4(
    const cv::Mat& src, const cv::Mat& bbs, int fixed_h, float nique_threshold,
    int output_channels, vector<cv::Mat>& rois);

// get rectangle roi, approx value comes from net inference, not defined by user
void getRectRoiOp(
    const cv::Mat& src, cv::Mat& dest, const vector<cv::Point2f> points,
    int fixed_h);

// get rectangle rois, the op is used for ctc model provided by liuqinjie@
void getRectRoisWithPaddingOp(
    const cv::Mat& src, const cv::Mat& rects, cv::Mat& rois, cv::Mat& widths,
    int fixed_h, bool align_4factor = true);

// get rectangle rois for transoformer model provided by chenfeng@
void getRectRoisWithPaddingOp2(
    const cv::Mat& src, const cv::Mat& rects, cv::Mat& rois, cv::Mat& widths,
    int fixed_h);

void transformerPreprocessOp(
    const std::vector<cv::Mat>& mat_list, cv::Mat& rois, cv::Mat& widths,
    int fixed_h);

void transformerCTCPreprocessOp(
    const std::vector<cv::Mat>& srcs, int batch_size, int fixed_h,
    int output_channels, bool use_min_width_limit, int device_count,
    PairMatList& dst, std::vector<int>& dst_indexes);

void enLargeRRectOp(
    const vector<cv::Point2f>& src_points, vector<cv::Point2f>& dest_points,
    float deltaW, float deltaH);

void ctcPreprocessOp(
    const std::vector<cv::Mat>& mat_list, int fixed_h, cv::Mat& dst,
    cv::Mat& width_mat);

void mergeBatchMatOp(const std::vector<cv::Mat>& mat_list, cv::Mat& dst);

void transposeHW(const cv::Mat& src, cv::Mat& dst);

void drawRectangle(const cv::Mat& src, const cv::Mat& r, cv::Mat& dst);

void compute_centerness_targets(const cv::Mat& geometry, cv::Mat& centerness);

void compute_sideness_targets(
    const cv::Mat& geometry, const std::string& lr_or_tb, cv::Mat& sideness);

void point_inside_of_box(
    const std::vector<cv::Point2f>& xy_text,
    const std::vector<cv::Point2f>& box, std::vector<int>& idex);

bool check_angle_length(const cv::Point2f& point1, const cv::Point2f& point2);

void refine_box(
    std::vector<cv::Point2f>& box, const std::vector<cv::Point2f>& xy_text,
    const cv::Mat& geometry, const std::vector<std::vector<cv::Point2f>>& boxes,
    float& ratio_h, float& ratio_w);

// mask related functions
typedef std::vector<std::vector<cv::Point>> Contours;
inline void
findContoursOp(const cv::Mat& mask, Contours& contours)
{
  cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}

// fixed bug @2019.09.20, update h -> (h - 1)
//  review the logic again, update h - 1 -> h @2019.09.26
inline void
clip_boxes(cv::Mat& bboxes, const float orig_h, const float orig_w)
{
  cv::Mat_<float> bb(bboxes);
  for (int i = 0; i < bboxes.rows; ++i) {
    // bb(i, 0) = std::min(std::max(bb(i, 0), 0.f), orig_w);
    // bb(i, 1) = std::min(std::max(bb(i, 1), 0.f), orig_h);
    // bb(i, 2) = std::min(std::max(bb(i, 2), 0.f), orig_w);
    // bb(i, 3) = std::min(std::max(bb(i, 3), 0.f), orig_h);
    bb(i, 0) = std::max(bb(i, 0), 0.f);
    bb(i, 1) = std::max(bb(i, 1), 0.f);
    bb(i, 2) = std::min(bb(i, 2), orig_w);
    bb(i, 3) = std::min(bb(i, 3), orig_h);
  }
}

// functions provided by @sangkun
// compute angle
double computeAngle(const std::vector<cv::Point2f>& v);
double computeAngle(
    const std::vector<cv::Point2f>& v,
    bool normalize_image_based_on_text_orientation);

// rotate image of mask rcnn
void rotateOp2(const cv::Mat& src, cv::Mat& dest, float angle);

// rotate image of bboxes
void readjust_bb(
    const cv::Point2f& old_center, const cv::Point2f& new_center,
    const double& theta, std::vector<std::vector<cv::Point2f>>& src);

inline int
calc_prop_scale(const cv::Mat& src, bool east_or_mask = false)
{
  std::vector<int> scale_list1 = {200, 400, 600, 800, 1000, 1200, 1600};
  std::vector<int> scale_list2 = {200, 400, 600, 800, 1056};
  auto& scale_list = east_or_mask ? scale_list2 : scale_list1;
  int max_side = std::max(src.rows, src.cols);
  int min_diff = 1e6;
  int scale = 0;
  for (const auto& s : scale_list) {
    if (std::abs((max_side - s)) < min_diff) {
      scale = s;
      min_diff = std::abs(max_side - s);
    }
  }
  return scale;
}

inline int
calc_prop_scale(const cv::Mat& src, const std::vector<int>& scale_list)
{
  int max_side = std::max(src.rows, src.cols);
  int min_diff = 1e6;
  int scale = 0;
  for (const auto& s : scale_list) {
    if (std::abs((max_side - s)) < min_diff) {
      scale = s;
      min_diff = std::abs(max_side - s);
    }
  }
  return scale;
}

inline std::pair<int, std::string>
calc_prop_model(
    const int max_side, const std::unordered_map<int, std::string>& scale_map)
{
  int min_diff = 1e6;
  std::pair<int, std::string> prop_model;
  for (const auto& s : scale_map) {
    if (std::abs((max_side - s.first)) < min_diff) {
      min_diff = std::abs(max_side - s.first);
      prop_model = s;
    }
  }
  return prop_model;
}

// shrink batch mat by widhts
void shrink_batch_mat(const cv::Mat& src, cv::Mat& dst, int width);

// flatter vector of vector
inline void
flat_points(
    const std::vector<std::vector<cv::Point2f>>& points_vec,
    std::vector<double>& flat_values)
{
  for (const auto& points : points_vec) {
    for (const auto& point : points) {
      flat_values.push_back((double)point.x);
      flat_values.push_back((double)point.y);
    }
  }
}

void crop_rrect(
    const cv::Mat& src, const std::vector<double> points,
    std::vector<cv::Mat> dsts);

template <typename _Tp>
cv::Mat
vec2mat(std::vector<_Tp>& v, int channels, int rows)
{
  cv::Mat mat = cv::Mat(v);
  cv::Mat dest = mat.reshape(channels, rows).clone();
  return dest;
}

inline cv::Mat
create_random_ndarray(int dims, int* sizes, int type, float low, float high)
{
  cv::Mat m = cv::Mat(dims, sizes, type);
  cv::randu(m, cv::Scalar(low), cv::Scalar(high));
  return m;
}

// create shape(1,1) mat
template <typename T>
cv::Mat
create_scalar_mat(T v)
{
  cv::Mat m = cv::Mat(1, 1, RawTypeToCvType<T>::value);
  m.at<T>(0, 0) = v;
  return m;
}

// debug functions
inline void
print_mat_info(const cv::Mat& m)
{
  std::cout << "mat::dims=[";
  int i = 0;
  for (; i < m.dims - 1; i++) {
    std::cout << m.size[i] << ",";
  }
  std::cout << m.size[i];
  if (m.dims == 2 && m.channels() > 1) {
    std::cout << "," << m.channels();
  }
  std::cout << "]\n";

  std::cout << "mat::type=" << CV_MAT_DEPTH(m.type()) << "\n";
}


template <typename F>
cv::Scalar
_tensor_sum(const cv::Mat& m)
{
  // for channel size great 4, only calc sum for first 4 channels
  cv::Scalar r = {0, 0, 0, 0};
  if (m.empty()) {
    return r;
  }
  std::vector<cv::Mat> channels;
  cv::split(m, channels);
  int n =
      std::accumulate(m.size.p, m.size.p + m.dims, 1, std::multiplies<int>());
  for (int i = 0; i < min(m.channels(), 4); i++) {
    auto* ptr = reinterpret_cast<F*>(channels[i].data);
    double s = 0;
    for (int j = 0; j < n; j++) {
      s += double(*(ptr + j));
    }
    r[i] = (double)s;
  }
  return r;
}

template <typename F>
void
print_mat(
    const cv::Mat& m, const std::string& name = "", bool with_content = false)
{
  std::cout << std::setprecision(10);
  std::cout << ">>>>>>>>>>>>>>>>>>>>\n";
  std::cout << "mat.name:" << name << "\n";
  std::cout << "mat::type=" << m.type() << "\n";
  std::cout << "mat::sum=" << _tensor_sum<F>(m) << "\n";

  std::cout << "mat::dims=[";
  int i = 0;
  for (; i < m.dims - 1; i++) {
    std::cout << m.size[i] << ",";
  }
  std::cout << m.size[i];
  if (m.dims == 2 && m.channels() > 1) {
    std::cout << "," << m.channels();
  }
  std::cout << "]\n";

  std::cout << "mat.isContinuous=" << m.isContinuous() << "\n";
  std::cout << "mat.rows(h)=" << m.rows << "\n";
  std::cout << "mat.cols(w)=" << m.cols << "\n";
  std::cout << "mat.channels(c)=" << m.channels() << "\n";

  auto func1 = [](F f) { std::cout << (int)f << ","; };
  auto func2 = [](F f) { std::cout << f << ","; };
  auto func = m.depth() <= 1 ? func1 : func2;

  if (m.empty())
    return;

  if (with_content) {
    // print normal mat
    if (m.dims == 2) {
      std::cout << "mat::data:\n";
      for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
          auto* ptr = m.ptr<F>(i, j);
          for (int k = 0; k < m.channels(); k++) {
            func(*(ptr + k));
          }
        }
        std::cout << "\n";
      }
    } else {
      std::cout << "not supported for high dimention mat\n";
    }
  } else {
    int c = m.channels();
    int size = (c >= 1 ? c : 1);
    for (int i = 0; i < m.dims; i++) {
      size *= m.size[i];
    }
    int n = std::min(16, size);
    auto* ptr = reinterpret_cast<F*>(m.data);
    std::cout << "mat::data[0-15]=[";
    for (int i = 0; i < n; i++) {
      func(*(ptr + i));
    }
    std::cout << "]\n";
  }
}

inline int
calcDiffCount(const cv::Mat& a, const cv::Mat& b, float eps = 1e-2)
{
  if (a.empty() && b.empty()) {
    return 0;
  }
  if (a.cols != b.cols || a.rows != b.rows || a.channels() != b.channels() ||
      a.dims != b.dims || a.type() != b.type()) {
    return -1;
  }

  cv::Mat diff;
  cv::Mat a1 = a, b1 = b;
  if (a.channels() > 1) {
    a1 = a.reshape(1);
    b1 = b.reshape(1);
  }
  cv::Mat diff_abs = cv::abs(a1 - b1);
  cv::compare(diff_abs, eps, diff, cv::CMP_GE);
  return cv::countNonZero(diff);
}

void reorder_start_point(
    std::vector<cv::Point2f>& points, float cos, float sin);

// update by hanfeng @2020.06.15,
// https://gitlab.4pd.io/cvxy4pd/cvbe/nn-predictor-offline/issues/4 added by
// hanfeng at 2020.05.20,
// https://gitlab.4pd.io/cvxy4pd/cvbe/nn-predictor/issues/56
void refine_box_orientation(std::vector<Point2fList>& points_vec);
void refine_box_orientation(
    std::vector<Point2fList>& points_vec, bool ignore_thres);

// face utils added by sangkun at 2020.05.27
// face mtcnn(pnet/rnet/onet model) related functions
struct PadRnet {
  int dy;
  int edy;
  int dx;
  int edx;
  int y;
  int ey;
  int x;
  int ex;
  int tmpw;
  int tmph;
};

void preprocess_image(const cv::Mat& src, cv::Mat& dst, float scale);

void generate_bbox(
    const cv::Mat& cls_cls_map, const cv::Mat& reg,
    std::vector<std::vector<float>>& bboxes, float current_scale,
    float pnet_threshold);

void convert_to_square(
    const std::vector<std::vector<float>>& src,
    std::vector<std::vector<float>>& dst);

void rnet_pad(
    const std::vector<std::vector<float>>& src, float w, float h,
    std::vector<PadRnet>& dst);

void calibrate_box(
    const std::vector<std::vector<float>>& src_bbs,
    const std::vector<std::vector<float>>& regs,
    std::vector<std::vector<float>>& dst_bbs);

// face feature(detect/onet/recong model) related functions
void gather_all_faces_into_batch(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst);

void convert_to_square_feature(
    const std::vector<std::vector<float>>& src,
    std::vector<std::vector<float>>& dst);

void pad_feature(
    const std::vector<std::vector<float>>& src, float w, float h,
    std::vector<PadRnet>& dst);

void gather_all_faces_into_batch_v2(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst);

void gather_all_faces_into_batch_v3(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst, cv::Mat& max_face);

void gather_all_faces_into_batch_v4(
    const cv::Mat& img, const std::vector<std::vector<float>>& bboxes,
    cv::Mat& dst, std::vector<cv::Mat>& mat_list);

void transform_keypoints2image_coor(
    const std::vector<std::vector<float>>& bbs,
    std::vector<std::vector<float>>& landmark);

cv::Mat meanAxis0(const cv::Mat& src);

cv::Mat elementwiseMinus(const cv::Mat& A, const cv::Mat& B);

cv::Mat varAxis0(const cv::Mat& src);

int MatrixRank(cv::Mat M);

cv::Mat similarTransform(cv::Mat src, cv::Mat dst);

void regcong_preprocess(
    const cv::Mat& img, const std::vector<float>& bbox,
    const std::vector<float>& landmark, cv::Mat& warped);

void recong_normalize(const cv::Mat& src, cv::Mat& dst);

std::string dec2hex(const uint64_t dec_num, const int width);

std::string ImageDHash(const cv::Mat image);

void calculate_mean_angle(
    const cv::Mat& cos_map, const cv::Mat& sin_map, const cv::Mat& yx_text,
    const std::vector<cv::Point2f> points_tmp, float& sin, float& cos);

void restore_rectangle(
    const cv::Mat& xy_test, const cv::Mat& geometry, cv::Mat& decoded_bboxes);

// 20201027 hjt modification
void getRRectRoisWithPaddingOp5(
    const cv::Mat& src, const cv::Mat& bbs, std::vector<cv::Mat>& rois);

void transformerCTCPreprocessOp2(
    const std::vector<cv::Mat>& srcs, int batch_size, int fixed_h,
    int output_channels, int downsample_rate, int W_min, int W_max,
    int device_count, PairMatList& dst, std::vector<int>& dst_indexes);

void preprocess_recog_batch_v2(
    const std::vector<cv::Mat>& imgs, cv::Mat& rois, cv::Mat& shapes, int H,
    int W_min, int W_max, bool is_grayscale, int downsample_rate,
    int extra_padding_length);

inline cv::Mat
to_2dmat(const cv::Mat& m)
{
  int h = m.size[0];
  int w = m.size[1];
  int c = m.size[2];
  return cv::Mat(h, w, CV_MAKETYPE(m.depth(), c), m.data);
}

TRITONSERVER_Error* DecodeImgFromB64(absl::string_view, cv::Mat&, int);

}}  // namespace dataelem::alg

#endif  // DATAELEM_COMMON_MAT_UTILS_H_
