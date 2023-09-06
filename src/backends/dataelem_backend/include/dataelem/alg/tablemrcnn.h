#ifndef DATAELEM_ALG_TABLEMRCNN_H_
#define DATAELEM_ALG_TABLEMRCNN_H_

#include "dataelem/common/apidata.h"
#include "dataelem/framework/alg.h"

namespace dataelem { namespace alg {

// updated 2020.04.12
// add support for v5

class TableMRCNN : public Algorithmer {
 public:
  TableMRCNN() = default;
  ~TableMRCNN() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

  TRITONSERVER_Error* Execute(AlgRunContext* context);

  TRITONSERVER_Error* GraphStep(
      AlgRunContext* context, const MatList& inputs, MatList& outputs);

 private:
  TRITONSERVER_Error* PreprocessStep(
      const APIData& params, const MatList& inputs, MatList& outputs);

  TRITONSERVER_Error* PostprocessStep(
      const APIData& params, const MatList& inputs, MatList& outputs);

  void mask_to_bb(
      const cv::Mat& scores, const cv::Mat& masks, const cv::Mat& bboxes,
      const cv::Mat& boxes_cos, const cv::Mat& boxes_sin, const cv::Mat& labels,
      const cv::Mat& orig, const cv::Mat& scale,
      std::vector<Point2fList>& points_vec, std::vector<float>& scores_vec,
      std::vector<int>& labels_vec);

 private:
  float _nms_threshold;
  bool _use_text_direction;
  bool _padding;
  std::string _version;
  std::vector<int> _scale_list;
  StepConfig graph_io_names_;

  std::unique_ptr<triton::backend::BackendMemory> input_buffer_mem_;
  char* input_buffer_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_ALG_MRCNN_H_