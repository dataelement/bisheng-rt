#ifndef DATAELEM_ALG_RECOG_HELPER_H_
#define DATAELEM_ALG_RECOG_HELPER_H_

#include "dataelem/framework/types.h"

namespace dataelem { namespace alg {

class LongImageSegment {
 public:
  LongImageSegment()
  {
    _seg_width = _unit_len - 100;
    _var_window_size = 2 * _interval + 1;
  }

  ~LongImageSegment() {}

  void segment(const MatList& patchs, MatList& new_patchs, IntegerList& groups);
  void merge(
      const SeqList& sequences, const IntegerList& groups,
      SeqList& merged_sequences, bool split_long_sentence_blank = false);
  void segment_v2(
      const std::vector<cv::Mat>& patchs, std::vector<cv::Mat>& new_patchs,
      std::vector<int>& groups);  // 20201229 hjt modification

  void merge_v2(
      const std::vector<absl::string_view>& texts,
      const std::vector<float>& scores, const IntegerList& groups,
      bool with_blank, std::vector<std::string>& output_texts,
      std::vector<float>& output_scores);

 private:
  int _interval = 1;
  double _max_stddev_value = 0.03;
  int _unit_len = 800;
  int _start_offset = 100;
  int _min_end_marge = 50;
  int _seg_width;
  int _var_window_size;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_RECOG_HELPER_H_