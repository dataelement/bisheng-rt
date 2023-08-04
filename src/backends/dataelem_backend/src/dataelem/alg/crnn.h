#ifndef DATAELEM_ALG_CRNN_H_
#define DATAELEM_ALG_CRNN_V1_H_

#include "dataelem/alg/recog_helper.h"
#include "dataelem/common/apidata.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class CRNN : public Algorithmer {
 public:
  CRNN() = default;
  ~CRNN() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  TRITONSERVER_Error* GraphStep(
      AlgRunContext* context, const PairMatList& inputs,
      std::vector<std::string>& texts, std::vector<float>& scores);

  TRITONSERVER_Error* PreprocessStep(
      const APIData& params, const MatList& inputs, PairMatList& outputs,
      IntegerList& index, IntegerList& groups);

  TRITONSERVER_Error* PostprocessStep(
      const APIData& params, const IntegerList& index,
      const std::vector<std::string>& input_texts,
      const std::vector<float>& input_scores,
      std::vector<absl::string_view>& output_texts,
      std::vector<float>& output_scores);

 private:
  int _batch_size;
  int _fixed_height;
  int _input_channels;
  int _downsample_rate;
  int _W_min;
  int _W_max;
  bool _is_pd_model;
  bool _output_matrix;
  int _recog_ins_num;
  std::vector<std::string> _label_list;
  StepConfig graph_io_names_;

  // Treat image segmentor as part of recog, combine it in recog alg.
  // for recog alg that support long segmentation, use it.
  std::unique_ptr<LongImageSegment> _long_image_segmentor;

  std::unique_ptr<triton::backend::BackendMemory> input0_buffer_;
  std::unique_ptr<triton::backend::BackendMemory> input1_buffer_;
  cv::Mat input0_;
  cv::Mat input1_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_CRNN_V1_H_
