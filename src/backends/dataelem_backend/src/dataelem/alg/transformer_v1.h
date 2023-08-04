#ifndef DATAELEM_ALG_TRANSFORMER_V1_H_
#define DATAELEM_ALG_TRANSFORMER_V1_H_

#include "dataelem/alg/recog_helper.h"
#include "dataelem/common/apidata.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

// updated 2023.03.19, support trt and tf, by hf

class TransformerV1 : public Algorithmer {
 public:
  TransformerV1() = default;
  ~TransformerV1() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  TRITONSERVER_Error* GraphStep(
      AlgRunContext* context, const PairMatList& inputs,
      std::vector<absl::string_view>& texts, std::vector<float>& scores);

  TRITONSERVER_Error* PreprocessStep(
      const APIData& params, const MatList& inputs, PairMatList& outputs,
      IntegerList& index, IntegerList& groups);

  TRITONSERVER_Error* PostprocessStep(
      const APIData& params, const IntegerList& index,
      const std::vector<absl::string_view>& input_texts,
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
  bool _enable_trt;
  int _recog_ins_num;
  StepConfig graph_io_names_;
  StepConfig graph_post_io_names_;

  // Treat image segmentor as part of recog, combine it in recog alg.
  // for recog alg that support long segmentation, use it.
  std::unique_ptr<LongImageSegment> _long_image_segmentor;

  std::unique_ptr<triton::backend::BackendMemory> input0_buffer_;
  std::unique_ptr<triton::backend::BackendMemory> input1_buffer_;
  cv::Mat input0_;
  cv::Mat input1_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_TRANSFORMER_V1_H_
