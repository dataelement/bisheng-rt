#ifndef DATAELEM_ALG_OCR_APP_H_
#define DATAELEM_ALG_OCR_APP_H_

#include "dataelem/alg/recog_helper.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

class OcrIntermediate : public Algorithmer {
 public:
  OcrIntermediate() = default;
  ~OcrIntermediate() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  std::unique_ptr<LongImageSegment> _long_image_segmentor;

  int _fixed_height;
  int _input_channels;
  int _downsample_rate;
  int _W_min;
  int _W_max;
  int _version;
  float _hw_thrd;

  int max_cache_patchs_;
  std::unique_ptr<triton::backend::BackendMemory> input0_buffer_;
  cv::Mat input0_;
};


class OcrPost : public Algorithmer {
 public:
  OcrPost() = default;
  ~OcrPost() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  int _version;
};

class AdjustBboxFromAngle : public Algorithmer {
 public:
  AdjustBboxFromAngle() = default;
  ~AdjustBboxFromAngle() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  float _thrd;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_OCR_APP_H_