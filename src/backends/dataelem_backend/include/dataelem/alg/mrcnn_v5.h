#ifndef DATAELEM_ALG_MRCNN_V5_H_
#define DATAELEM_ALG_MRCNN_V5_H_

#include "dataelem/common/thread_pool.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

// updated 2022.10.19, reconstruct for pipeline
// updated 2020.04.12
// add support for v5

class MaskRCNNV5Base : public Algorithmer {
 public:
  MaskRCNNV5Base() = default;
  ~MaskRCNNV5Base() = default;

  virtual TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

 protected:
  float _nms_threshold;
  bool _use_text_direction;
  bool _padding;
  std::string _version;
  std::vector<int> _scale_list;
};

class MaskRCNNV5Prep : public MaskRCNNV5Base {
 public:
  MaskRCNNV5Prep() = default;
  ~MaskRCNNV5Prep() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 protected:
  std::unique_ptr<triton::backend::BackendMemory> input_buffer_;
  bool _base64dec = true;
};

class MaskRCNNV5Post : public MaskRCNNV5Base {
 public:
  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);
  TRITONSERVER_Error* Execute(AlgRunContext* context);

 private:
  void mask_to_bb(
      const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
      const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& orig,
      const cv::Mat& scale, const bool& enable_huarong_box_adjust,
      const bool& unify_text_direction, std::vector<Point2fList>& points_vec,
      std::vector<float>& scores_vec);

 protected:
  std::unique_ptr<ThreadPool> _pool;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_MRCNN_V5_H_
