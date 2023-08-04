#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/thread_pool.h"

namespace lanms {

// http://geomalgorithms.com/a01-_area.html#2D%20Polygons
float
paths_area(const ClipperLib::Paths& ps)
{
  float area = 0;
  for (auto&& p : ps) area += cl::Area(p);
  return area;
}

float
poly_iou(const Polygon& a, const Polygon& b)
{
  cl::Clipper clpr;
  clpr.AddPath(a.poly, cl::ptSubject, true);
  clpr.AddPath(b.poly, cl::ptClip, true);
  cl::Paths inter, uni;
  clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
  clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);
  auto inter_area = paths_area(inter), uni_area = paths_area(uni);
  return std::fabs(inter_area) / std::max(std::fabs(uni_area), 1.0f);
}

void
PolyMerger::add(const Polygon& p_given)
{
  Polygon p;
  if (nr_polys > 0) {
    // vertices of two polygons to merge may not in the same order;
    // we match their vertices by choosing the ordering that
    // minimizes the total squared distance.
    // see function normalize_poly for details.
    p = normalize_poly(get(), p_given);
  } else {
    p = p_given;
  }
  assert(p.poly.size() == 4);
  auto& poly = p.poly;
  auto s = p.score;
  data[0] += poly[0].X * s;
  data[1] += poly[0].Y * s;

  data[2] += poly[1].X * s;
  data[3] += poly[1].Y * s;

  data[4] += poly[2].X * s;
  data[5] += poly[2].Y * s;

  data[6] += poly[3].X * s;
  data[7] += poly[3].Y * s;

  score += p.score;

  nr_polys += 1;
}

Polygon
PolyMerger::normalize_poly(const Polygon& ref, const Polygon& p)
{
  std::int64_t min_d = std::numeric_limits<std::int64_t>::max();
  size_t best_start = 0, best_order = 0;

  for (size_t start = 0; start < 4; start++) {
    size_t j = start;
    std::int64_t d =
        (sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 0) % 4].X) +
         sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 0) % 4].Y) +
         sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 1) % 4].X) +
         sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 1) % 4].Y) +
         sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 2) % 4].X) +
         sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 2) % 4].Y) +
         sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 3) % 4].X) +
         sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 3) % 4].Y));
    if (d < min_d) {
      min_d = d;
      best_start = start;
      best_order = 0;
    }

    d =
        (sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 3) % 4].X) +
         sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 3) % 4].Y) +
         sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 2) % 4].X) +
         sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 2) % 4].Y) +
         sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 1) % 4].X) +
         sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 1) % 4].Y) +
         sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 0) % 4].X) +
         sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 0) % 4].Y));
    if (d < min_d) {
      min_d = d;
      best_start = start;
      best_order = 1;
    }
  }

  Polygon r;
  r.poly.resize(4);
  auto j = best_start;
  if (best_order == 0) {
    for (size_t i = 0; i < 4; i++) r.poly[i] = p.poly[(j + i) % 4];
  } else {
    for (size_t i = 0; i < 4; i++) r.poly[i] = p.poly[(j + 4 - i - 1) % 4];
  }
  r.score = p.score;
  return r;
}

Polygon
PolyMerger::get() const
{
  Polygon p;

  auto& poly = p.poly;
  poly.resize(4);
  auto score_inv = 1.0f / std::max(1e-8f, score);
  poly[0].X = data[0] * score_inv;
  poly[0].Y = data[1] * score_inv;
  poly[1].X = data[2] * score_inv;
  poly[1].Y = data[3] * score_inv;
  poly[2].X = data[4] * score_inv;
  poly[2].Y = data[5] * score_inv;
  poly[3].X = data[6] * score_inv;
  poly[3].Y = data[7] * score_inv;

  assert(score > 0);
  p.score = score;

  return p;
}


// The standard NMS algorithm.
std::vector<Polygon>
standard_nms(std::vector<Polygon>& polys, float iou_threshold)
{
  size_t n = polys.size();
  if (n == 0) {
    return {};
  }

  std::vector<size_t> indices(n);
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) {
    return polys[i].score > polys[j].score;
  });

  std::vector<size_t> keep;
  while (indices.size()) {
    size_t p = 0, cur = indices[0];
    keep.emplace_back(cur);
    for (size_t i = 1; i < indices.size(); i++) {
      if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
        indices[p++] = indices[i];
      }
    }
    indices.resize(p);
  }

  std::vector<Polygon> ret;
  for (auto&& i : keep) {
    ret.emplace_back(polys[i]);
  }
  return ret;
}

std::vector<Polygon>
standard_nms_jianbiao(std::vector<Polygon>& polys, float iou_threshold)
{
  size_t n = polys.size();
  if (n == 0) {
    return {};
  }
  std::sort(std::begin(polys), std::end(polys), [&](Polygon& a, Polygon& b) {
    return a.score > b.score;
  });

  std::vector<std::vector<bool>> mask(n, std::vector<bool>(n, 0));
  dataelem::alg::parallel_run_dynamic(
      n, [&mask, &polys, &iou_threshold, &n](size_t i) {
        for (unsigned int j = i + 1; j < n; j++) {
          mask[i][j] = should_merge(polys[i], polys[j], iou_threshold);
        }
      });

  std::vector<bool> flags(mask[0]);
  std::vector<Polygon> ret = {polys[0]};
  for (unsigned int i = 1; i < n; i++) {
    if (!flags[i]) {
      ret.emplace_back(polys[i]);
      for (unsigned int j = i + 1; j < n; j++) {
        if (mask[i][j]) {
          flags[j] = true;
        }
      }
    }
  }
  return ret;
}

std::vector<Polygon>
merge_quadrangle_n9(
    const float* bbs, const float* scores, size_t t, float iou_threshold)
{
  using cInt = cl::cInt;
  // first pass
  std::vector<Polygon> polys_bbs;
  for (size_t i = 0; i < t; i++) {
    auto* p = bbs + i * 8;
    Polygon poly{
        {
            {cInt(p[0]), cInt(p[1])},
            {cInt(p[2]), cInt(p[3])},
            {cInt(p[4]), cInt(p[5])},
            {cInt(p[6]), cInt(p[7])},
        },
        scores[i],
    };
    polys_bbs.emplace_back(poly);
  }
  int num = 10, threads = ceil(t / ceil(float(t) / num));
  std::vector<std::vector<Polygon>> polys_merge(threads);
  auto merge = [](const std::vector<Polygon>& polys_bbs, size_t start,
                  size_t end, const float& iou_threshold,
                  std::vector<std::vector<Polygon>>& polys_merge, int k) {
    size_t merger_num = 0;
    std::vector<Polygon> polys;
    for (size_t i = start; i <= end; ++i) {
      Polygon poly = polys_bbs[i];
      if (polys.size()) {
        // merge with the last one
        auto& bpoly = polys.back();
        if (should_merge(poly, bpoly, iou_threshold)) {
          PolyMerger merger;
          merger.add(bpoly);
          merger.add(poly);
          bpoly = merger.get();
          merger_num += 1;
          if (i == end) {
            bpoly.score /= merger_num;
          }
        } else {
          bpoly.score /= merger_num;
          merger_num = 1;
          polys.emplace_back(poly);
        }
      } else {
        polys.emplace_back(poly);
        merger_num = 1;
      }
    }
    polys_merge[k] = polys;
  };
  dataelem::alg::ThreadPool& tp = dataelem::alg::nn_thread_pool();
  std::vector<dataelem::alg::BoolFuture> rets(threads);
  for (int i = 0; i < threads; i++) {
    int start = i * ceil(float(t) / num),
        end = std::min(int((i + 1) * ceil(float(t) / num)), int(t)) - 1;
    rets[i] = tp.enqueue(
        [&merge, &polys_bbs, start, end, iou_threshold, &polys_merge, i]() {
          merge(polys_bbs, start, end, iou_threshold, polys_merge, i);
          return true;
        });
  }
  dataelem::alg::GetAsyncRets(rets);
  std::vector<Polygon> polys_merged;
  for (unsigned i = 0; i < polys_merge.size(); i++) {
    polys_merged.insert(
        polys_merged.end(), polys_merge[i].begin(), polys_merge[i].end());
  }
  int start = 0, end = polys_merged.size() - 1;
  merge(polys_merged, start, end, iou_threshold, polys_merge, 0);
  std::vector<Polygon> final;
  if (polys_merge[0].size() > 600) {
    final = standard_nms(polys_merge[0], iou_threshold);
  } else {
    final = standard_nms_jianbiao(polys_merge[0], iou_threshold);
  }
  return final;
}

std::vector<size_t>
merge_quadrangle_standard(
    const std::vector<std::vector<cv::Point2f>>& bbs,
    const std::vector<float>& scores, size_t n, float iou_threshold)
{
  using cInt = cl::cInt;
  constexpr float NMS_SCALE = 10000;
  std::vector<Polygon> polys;
  for (size_t i = 0; i < n; i++) {
    auto p = bbs[i];
    Polygon poly{
        {{cInt(p[0].x * NMS_SCALE), cInt(p[0].y * NMS_SCALE)},
         {cInt(p[1].x * NMS_SCALE), cInt(p[1].y * NMS_SCALE)},
         {cInt(p[2].x * NMS_SCALE), cInt(p[2].y * NMS_SCALE)},
         {cInt(p[3].x * NMS_SCALE), cInt(p[3].y * NMS_SCALE)}},
        scores[i],
    };
    polys.emplace_back(poly);
  }

  size_t cnt = polys.size();
  if (cnt == 0) {
    return {};
  }

  std::vector<size_t> indices(n);
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) {
    return polys[i].score > polys[j].score;
  });

  std::vector<size_t> keep;
  while (indices.size()) {
    size_t p = 0, cur = indices[0];
    keep.emplace_back(cur);
    for (size_t i = 1; i < indices.size(); i++) {
      if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
        indices[p++] = indices[i];
      }
    }
    indices.resize(p);
  }
  return keep;
}

std::vector<size_t>
merge_quadrangle_standard_parallel(
    const std::vector<std::vector<cv::Point2f>>& bbs,
    const std::vector<float>& scores, size_t n, float iou_threshold)
{
  using cInt = cl::cInt;
  constexpr float NMS_SCALE = 10000;
  std::vector<Polygon> polys;
  for (size_t i = 0; i < n; i++) {
    auto p = bbs[i];
    Polygon poly{
        {{cInt(p[0].x * NMS_SCALE), cInt(p[0].y * NMS_SCALE)},
         {cInt(p[1].x * NMS_SCALE), cInt(p[1].y * NMS_SCALE)},
         {cInt(p[2].x * NMS_SCALE), cInt(p[2].y * NMS_SCALE)},
         {cInt(p[3].x * NMS_SCALE), cInt(p[3].y * NMS_SCALE)}},
        scores[i],
    };
    polys.emplace_back(poly);
  }

  std::vector<size_t> indices(n);
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) {
    return polys[i].score > polys[j].score;
  });
  std::vector<std::vector<bool>> mask(n, std::vector<bool>(n, 0));
  dataelem::alg::parallel_run_dynamic(
      n, [&mask, &polys, &iou_threshold, &n, &indices](size_t i) {
        for (unsigned int j = i + 1; j < n; j++) {
          mask[i][j] =
              should_merge(polys[indices[i]], polys[indices[j]], iou_threshold);
        }
      });
  std::vector<size_t> keep = {indices[0]};
  std::vector<bool> flags(mask[0]);
  for (unsigned int i = 1; i < n; i++) {
    int idx = indices[i];
    if (!flags[i]) {
      keep.emplace_back(idx);
      for (unsigned int j = i + 1; j < n; j++) {
        if (mask[i][j]) {
          flags[j] = true;
        }
      }
    }
  }
  return keep;
}

}  // namespace lanms