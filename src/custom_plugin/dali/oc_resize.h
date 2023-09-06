#ifndef CUSTOM_PLUGIN_DALI_RESIZE_H_
#define CUSTOM_PLUGIN_DALI_RESIZE_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace other_ns {

template <typename Backend>
class OCResize : public ::dali::Operator<Backend> {
 public:
  inline explicit OCResize(const ::dali::OpSpec& spec)
      : ::dali::Operator<Backend>(spec)
  {
    resize_x_ = spec.GetArgument<int>("resize_x");
    resize_y_ = spec.GetArgument<int>("resize_y");
  }

  virtual inline ~OCResize() = default;

  OCResize(const OCResize&) = delete;
  OCResize& operator=(const OCResize&) = delete;
  OCResize(OCResize&&) = delete;
  OCResize& operator=(OCResize&&) = delete;

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(
      std::vector<::dali::OutputDesc>& output_desc,
      const ::dali::workspace_t<Backend>& ws) override
  {
    const auto& input = ws.template Input<Backend>(0);
    output_desc.resize(1);

    auto out_shape = input.shape();
    int N = out_shape.num_samples();
    for (int i = 0; i < N; i++) {
      auto out_sample_shape = out_shape.tensor_shape_span(i);
      out_sample_shape[0] = int64_t(resize_y_);
      out_sample_shape[1] = int64_t(resize_x_);
    }

    output_desc[0] = {out_shape, input.type()};
    return true;
  }

  void RunImpl(::dali::workspace_t<Backend>& ws) override;

 private:
  int resize_x_;
  int resize_y_;
};

}  // namespace other_ns

#endif  // CUSTOM_PLUGIN_DALI_RESIZE_H_
