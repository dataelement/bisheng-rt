#pragma once

#include "clipper/clipper.hpp"

// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		float score;
		float box_cos;
		float box_sin;
	};

	float paths_area(const ClipperLib::Paths &ps) {
		float area = 0;
		for (auto &&p: ps)
			area += cl::Area(p);
		return area;
	}

	float poly_iou(const Polygon &a, const Polygon &b) {
		cl::Clipper clpr;
		clpr.AddPath(a.poly, cl::ptSubject, true);
		clpr.AddPath(b.poly, cl::ptClip, true);

		cl::Paths inter, uni;
		clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
		clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

		auto inter_area = paths_area(inter),
			 uni_area = paths_area(uni);
		return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
	}

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold) {
		return poly_iou(a, b) > iou_threshold;
	}


	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold) {
		size_t n = polys.size();
		if (n == 0)
			return {};
		std::vector<size_t> indices(n);
		std::iota(std::begin(indices), std::end(indices), 0);
		std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

		std::vector<size_t> keep;
		while (indices.size()) {
			size_t p = 0, cur = indices[0];
			keep.emplace_back(cur);
			for (size_t i = 1; i < indices.size(); i ++) {
				if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
					indices[p ++] = indices[i];
				}
			}
			indices.resize(p);
		}

		std::vector<Polygon> ret;
		for (auto &&i: keep) {
			ret.emplace_back(polys[i]);
		}
		return ret;
	}

	std::vector<Polygon>
		merge_quadrangle_standard(const float *data, size_t n, float iou_threshold) {
			using cInt = cl::cInt;

			// first pass
			std::vector<Polygon> polys;
			for (size_t i = 0; i < n; i ++) {
				auto p = data + i * 11;
				Polygon poly{
					{
						{cInt(p[0]), cInt(p[1])},
						{cInt(p[2]), cInt(p[3])},
						{cInt(p[4]), cInt(p[5])},
						{cInt(p[6]), cInt(p[7])},
					},
					p[8],
					p[9],
					p[10],
				};
				polys.emplace_back(poly);
			}
			return standard_nms(polys, iou_threshold);
		}
}
