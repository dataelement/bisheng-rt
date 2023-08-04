#include <absl/strings/str_join.h>

#include "dataelem/alg/recog_helper.h"
#include "dataelem/common/mat_utils.h"

namespace dataelem { namespace alg {

void
LongImageSegment::segment(
    const MatList& patchs, MatList& new_patchs, IntegerList& groups)
{
  // input image is color, otherwise change the threholds in the construction
  for (size_t group_id = 0; group_id < patchs.size(); group_id++) {
    auto& m = patchs[group_id];
    int w = m.cols;
    int h = m.rows;
    if (w <= _unit_len) {
      new_patchs.emplace_back(m);
      groups.push_back(group_id);
      continue;
    }

    cv::Mat m_gray;
    if (m.channels() > 1) {
      cv::cvtColor(m, m_gray, CV_BGR2GRAY);
    } else {
      m_gray = m;
    }

    double min_v, max_v;
    cv::minMaxLoc(m_gray, &min_v, &max_v);
    double is_zero = max_v - min_v > 1e-6, r = max_v - min_v;
    double alpha = is_zero ? 1 / r : 1;
    double beta = is_zero > 1e-6 ? -min_v / r : -min_v;
    cv::Mat m_norm;
    m_gray.convertTo(m_norm, CV_64F, alpha, beta);

    int part_num = (int)(w * 1.0 / _seg_width);
    int start_point = 0, end_point = 0;
    bool end_flag = false;
    std::vector<int> point_list;

    for (int i = 0; i < part_num; i++) {
      start_point += _seg_width;
      end_point = std::min(start_point + _start_offset, w);
      for (int point = start_point; point < end_point - _interval;
           point += _interval) {
        if (!end_flag) {
          cv::Mat sub_patch =
              m_norm(cv::Rect(point - _interval, 0, _var_window_size, h));
          // print_mat<double>(sub_patch, "seg");
          cv::Scalar mean, stddev;
          cv::meanStdDev(sub_patch.reshape(1), mean, stddev);
          // std::cout << "var:" << stddev(0) << ",p:" << point << "\n";
          if (stddev(0) < _max_stddev_value ||
              (end_point - 2 * _interval <= point &&
               point < end_point - _interval)) {
            if (w - point < _min_end_marge) {
              point_list.push_back(w);
              end_flag = true;
            } else {
              point_list.push_back(point);
            }
            break;
          }
        }
      }  // for
    }    // for

    if (!end_flag) {
      point_list.push_back(w);
    }

    int last_point_ori = 0;
    for (auto& point : point_list) {
      new_patchs.emplace_back(
          m(cv::Rect(last_point_ori, 0, point - last_point_ori, h)));
      groups.push_back(group_id);
      last_point_ori = point;
    }
  }
}

void
LongImageSegment::merge(
    const SeqList& sequences, const IntegerList& groups,
    SeqList& merged_sequences, bool split_long_sentence_blank)
{
  if (sequences.texts.size() <= 1) {
    merged_sequences.texts = sequences.texts;
    merged_sequences.scores = sequences.scores;
    return;
  }

  auto curr_val = sequences.texts[0];
  auto curr_group = groups[0];
  for (size_t i = 1; i < sequences.texts.size(); i++) {
    if (groups[i] == curr_group) {
      if (split_long_sentence_blank) {
        curr_val += " ";
        curr_val += sequences.texts[i];
      } else {
        curr_val += sequences.texts[i];
      }
    } else {
      merged_sequences.texts.emplace_back(std::move(curr_val));
      curr_group = groups[i];
      curr_val = sequences.texts[i];
    }
  }
  merged_sequences.texts.emplace_back(std::move(curr_val));

  if (sequences.scores.size() > 1) {
    auto curr_score = sequences.scores[0];
    curr_group = groups[0];
    int num_in_grop = 1;
    for (size_t i = 1; i < sequences.scores.size(); i++) {
      if (groups[i] == curr_group) {
        curr_score += sequences.scores[i];
        num_in_grop++;
      } else {
        merged_sequences.scores.emplace_back(
            std::move(curr_score / num_in_grop));
        curr_group = groups[i];
        curr_score = sequences.scores[i];
        num_in_grop = 1;
      }
    }
    merged_sequences.scores.emplace_back(std::move(curr_score / num_in_grop));
  }
}

void
LongImageSegment::segment_v2(
    const std::vector<cv::Mat>& patchs, std::vector<cv::Mat>& new_patchs,
    std::vector<int>& groups)
{
  // input image is color, otherwise change the threholds in the construction
  for (size_t group_id = 0; group_id < patchs.size(); group_id++) {
    int H = 32;
    auto& m = patchs[group_id];
    int w = m.cols;
    int h = m.rows;
    int resized_w = int(w * H * 1.0 / h);

    // debug purpose
    // int _unit_len = 800;
    // int _interval = 1;
    // int _start_offset = 100;
    // int _seg_width = _unit_len - 100;
    // int _var_window_size = 2 * _interval + 1;
    // int _min_end_marge = 50;

    if (resized_w <= _unit_len) {
      new_patchs.emplace_back(m);
      groups.push_back(group_id);
      continue;
    }

    // resize and cvt to gray
    cv::Mat img, m_gray;
    resizeOp(m, img, resized_w, H);
    if (img.channels() > 1) {
      cv::cvtColor(img, m_gray, CV_BGR2GRAY);
    } else {
      m_gray = img;
    }

    cv::Scalar img_mean, img_std;
    cv::meanStdDev(m_gray.reshape(1), img_mean, img_std);
    // std::cout << "mean " << img_mean(0) << " std " << img_std(0) <<
    // std::endl;
    cv::Mat m_norm;
    double is_nonzero = img_std(0) > 1e-6;
    double alpha = is_nonzero ? 1 / img_std(0) : 1;
    double beta = is_nonzero ? -img_mean(0) / img_std(0) : -img_mean(0);
    m_gray.convertTo(m_norm, CV_64F, alpha, beta);

    int part_num = (int)(resized_w * 1.0 / _seg_width);
    int start_point = 0, end_point = 0, seg_point = 0;
    bool end_flag = false;
    std::vector<int> point_list;

    for (int i = 0; i < part_num; i++) {
      start_point += _seg_width;
      end_point = std::min(start_point + _start_offset, resized_w);
      if (end_point - _interval > start_point) {
        std::vector<float> vars;
        std::vector<int> offsets;
        for (int point = start_point; point < end_point - _interval;
             point += _interval) {
          cv::Mat seg =
              m_norm(cv::Rect(point - _interval, 0, _var_window_size, H));
          cv::Scalar mean, stddev;
          cv::meanStdDev(seg.reshape(1), mean, stddev);
          vars.emplace_back(stddev(0));
          offsets.emplace_back(point);
        }
        auto mini = std::min_element(vars.begin(), vars.end());
        seg_point = offsets[mini - vars.begin()];

        if (resized_w - seg_point < _min_end_marge) {
          point_list.push_back(resized_w);
          end_flag = true;
          break;
        } else {
          point_list.push_back(seg_point);
        }
      }
    }

    if (!end_flag) {
      point_list.push_back(resized_w);
    }

    int last_point_ori = 0;
    for (auto& point : point_list) {
      int point_ori = std::min(int(point * h * 1.0 / H), w);
      new_patchs.emplace_back(
          m(cv::Rect(last_point_ori, 0, point_ori - last_point_ori, h))
              .clone());
      groups.push_back(group_id);
      last_point_ori = point_ori;
    }
  }
}

void
LongImageSegment::merge_v2(
    const std::vector<absl::string_view>& texts,
    const std::vector<float>& scores, const IntegerList& groups,
    bool with_blank, std::vector<std::string>& output_texts,
    std::vector<float>& output_scores)
{
  if (texts.size() <= 1) {
    if (texts.size() == 1) {
      output_texts.emplace_back(std::string(texts[0]));
    }
    output_scores = scores;
    return;
  }

  std::string SEP = with_blank ? " " : "";
  std::string curr_val = std::string(texts[0]);
  auto curr_score = scores[0];
  auto curr_group = groups[0];
  int num_in_grop = 1;
  for (size_t i = 1; i < texts.size(); i++) {
    if (groups[i] == curr_group) {
      absl::StrAppend(&curr_val, SEP, texts.at(i));
      curr_score += scores[i];
      num_in_grop++;
    } else {
      output_texts.emplace_back(std::move(curr_val));
      output_scores.emplace_back(curr_score / num_in_grop);
      curr_group = groups[i];
      curr_val = std::string(texts[i]);
      num_in_grop = 1;
    }
  }
  output_texts.emplace_back(std::move(curr_val));
  output_scores.emplace_back(curr_score / num_in_grop);
}

}}  // namespace dataelem::alg
