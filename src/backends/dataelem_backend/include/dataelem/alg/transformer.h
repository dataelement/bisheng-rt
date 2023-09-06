#ifndef DATAELEM_ALG_TRANSFORMER_H_
#define DATAELEM_ALG_TRANSFORMER_H_

#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class TransformerBase : public Algorithmer {
 public:
  TransformerBase() = default;
  ~TransformerBase() = default;

  virtual TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

 protected:
  int _batch_size;
  int _fixed_height;
  int _input_channels;
  int _extra_padding_length = 108;
};


class TransformerAlg : public TransformerBase {
 public:
  TransformerAlg() = default;
  ~TransformerAlg() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  std::unique_ptr<triton::backend::BackendMemory> input0_buffer_;
  std::unique_ptr<triton::backend::BackendMemory> input1_buffer_;
  cv::Mat input0_;
  cv::Mat input1_;

  StepConfig graph_io_names_ = {
      {"image", "image_shape"},
      {"while/Exit_1", "Transformer/strided_slice_16"}};
};

class TransformerTrtAlg : public TransformerBase {
 public:
  TransformerTrtAlg() = default;
  ~TransformerTrtAlg() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  std::unique_ptr<triton::backend::BackendMemory> input0_buffer_;
  std::unique_ptr<triton::backend::BackendMemory> input1_buffer_;
  cv::Mat input0_;
  cv::Mat input1_;

  StepConfig graph_io_names_ = {{"image", "image_shape"}, {"texts"}};

  // StepConfig graph_io_names_ = {
  //     {"inputs", "inputs_shape"},
  //     {"output_ids", "parent_ids", "sequence_length"}};

  // StepConfig post_graph_io_names_ = {
  //     {"output_ids", "parent_ids", "sequence_length"}, {"while/Exit_1"}};
};


// class TransformerPost : public TransformerBase {
//  public:
//   TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
//   TRITONSERVER_Error* Execute(AlgRunContext* context);

//  private:
// };

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_TRANSFORMER_H_