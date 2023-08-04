#include "oc_resize.h"
#include "opencv2/opencv.hpp"

namespace other_ns {

template <>
void
OCResize<::dali::CPUBackend>::RunImpl(::dali::HostWorkspace& ws)
{
  const auto& input = ws.Input<::dali::CPUBackend>(0);
  auto& output = ws.Output<::dali::CPUBackend>(0);

  ::dali::TypeInfo type = input.type_info();
  // auto& tp = ws.GetThreadPool();
  const auto& in_shape = input.shape();
  const auto& out_shape = output.shape();

  // do computing
  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    void* input_buffer = const_cast<void*>(input.raw_tensor(sample_id));
    auto in_sample_shape = in_shape.tensor_shape_span(sample_id);

    void* output_buffer = output.raw_mutable_tensor(sample_id);
    cv::Mat dst(resize_y_, resize_x_, CV_8UC3, output_buffer);

    int64_t h = in_sample_shape[0];
    int64_t w = in_sample_shape[1];
    int64_t c = in_sample_shape[2];
    cv::Mat src(h, w, CV_8UC3, input_buffer);
    cv::resize(src, dst, {resize_y_, resize_x_}, 0, 0, cv::INTER_LINEAR);
  }
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(
    CustomOCResize, ::other_ns::OCResize<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(CustomOCResize)
    .DocStr(R"code(OCResize images.)code")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("resize_x", "resize x", 0, true)
    .AddOptionalArg("resize_y", "resize y", 0, true);