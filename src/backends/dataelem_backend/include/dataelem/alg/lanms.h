#ifndef DATAELEM_ALG_LANMS_H_
#define DATAELEM_ALG_LANMS_H_

#include <opencv2/opencv.hpp>
#include "ext/clipper/clipper.hpp"

// locality-aware NMS
namespace lanms {

namespace cl = ClipperLib;

struct Polygon {
  cl::Path poly;
  float score;
};

typedef std::vector<Polygon> PolygonList;
typedef std::vector<std::vector<Polygon>> PolygonListList;

// http://geomalgorithms.com/a01-_area.html#2D%20Polygons
float paths_area(const ClipperLib::Paths& ps);
float poly_iou(const Polygon& a, const Polygon& b);

inline bool
should_merge(const Polygon& a, const Polygon& b, float iou_thres)
{
  return poly_iou(a, b) > iou_thres;
}

// Incrementally merge polygons
class PolyMerger {
 public:
  PolyMerger() : score(0), nr_polys(0) { memset(data, 0, sizeof(data)); }

  void add(const Polygon& p_given);
  inline std::int64_t sqr(std::int64_t x) { return x * x; }
  Polygon normalize_poly(const Polygon& ref, const Polygon& p);
  Polygon get() const;

 private:
  std::int64_t data[8];
  float score;
  std::int32_t nr_polys;
};

std::vector<Polygon> standard_nms(std::vector<Polygon>&, float);

std::vector<Polygon> standard_nms_jinbiao(std::vector<Polygon>&, float);

std::vector<Polygon> merge_quadrangle_n9(
    const float* bbs, const float* scores, size_t n, float iou_threshold);

std::vector<size_t> merge_quadrangle_standard(
    const std::vector<std::vector<cv::Point2f>>& bbs,
    const std::vector<float>& scores, size_t n, float iou_threshold);

std::vector<size_t> merge_quadrangle_standard_parallel(
    const std::vector<std::vector<cv::Point2f>>& bbs,
    const std::vector<float>& scores, size_t n, float iou_threshold);

}  // namespace lanms

#endif  // DATAELEM_ALG_LANMS_H_
