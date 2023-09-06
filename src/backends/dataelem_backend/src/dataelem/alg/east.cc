#include "dataelem/alg/east.h"

#include <array>

#include "absl/strings/escaping.h"
#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/common/thread_pool.h"
#include "dataelem/framework/alg_factory.h"
#include "nlohmann/json.hpp"


namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(EastV4);


TRITONSERVER_Error*
EastV4::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "east_v4";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "nms_threshold", &_nms_threshold, 0.2f);
  SafeParseParameter(params, "score_threshold", &_score_threshold, 0.8f);

  std::string scale_list_str;
  SafeParseParameter(params, "scale_list", &scale_list_str, "");
  if (!scale_list_str.empty()) {
    ParseArrayFromString(scale_list_str, _scale_list);
  } else {
    _scale_list.assign({200, 400, 600, 800, 1056});
  }
  _use_text_direction = true;

  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

  return nullptr;
}


TRITONSERVER_Error*
EastV4::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);

  // Run input initialize
  // parse input0
  cv::Mat src;
  nlohmann::json param;
  {
    std::string image_raw_bytes;
    auto* tensor1 = &inputs[0];
    if (!absl::Base64Unescape(tensor1->GetString(0), &image_raw_bytes)) {
      TRITONJSON_STATUSRETURN("base64 decode failed in alg:" + alg_name_);
    }
    int n = image_raw_bytes.length();
    auto bytes_mat =
        cv::Mat(n, 1, CV_8U, const_cast<char*>(image_raw_bytes.data()));
    try {
      src = cv::imdecode(bytes_mat, 1);
    }
    catch (cv::Exception& e) {
      TRITONJSON_STATUSRETURN(e.err);
    }

    if (optional_inputs_.size() > 0) {
      // parse params
      try {
        OCTensor* params_tensor;
        if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
          auto content1 = params_tensor->GetString(0);
          param = nlohmann::json::parse(
              content1.data(), content1.data() + content1.length());
        }
      }
      catch (nlohmann::json::parse_error& e) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
      }
    }
  }

  // Run Prep
  cv::Mat out, hw;
  {
    int longer_edge_size;
    if (param.contains("longer_edge_size")) {
      longer_edge_size = param["longer_edge_size"].get<int>();
      if (longer_edge_size <= 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "longer_edge_size must be greater than zero");
      }
    } else {
      longer_edge_size = calc_prop_scale(src, _scale_list);
    }

    cv::Mat resize;
    float coef = calcCoefOp(src, longer_edge_size);
    int new_w = (int)src.cols * coef;
    int new_h = (int)src.rows * coef;
    new_w = new_w % 32 == 0 ? new_w : (new_w / 32 + 1) * 32;
    new_h = new_h % 32 == 0 ? new_h : (new_h / 32 + 1) * 32;
    cv::Mat mat1, dst;
    resizeOp(src, resize, new_w, new_h);
    if (param.contains("padding") && param["padding"].get<bool>()) {
      cv::copyMakeBorder(
          resize, dst, 0, longer_edge_size - resize.rows, 0,
          longer_edge_size - resize.cols, cv::BORDER_CONSTANT, {0, 0, 0});
    } else {
      dst = resize.clone();
    }

    mat1 = BGR2RGB(dst);
    // matutils::bgr2rgbOp(dst, mat1);
    mat1.convertTo(mat1, CV_32F);
    if (param.contains("padding") && param["padding"].get<bool>()) {
      out = mat1.reshape(3, {1, longer_edge_size, longer_edge_size});
    } else {
      out = mat1.reshape(3, {1, new_h, new_w});
    }
    float ratio_h = (float)new_h / (float)src.rows;
    float ratio_w = (float)new_w / (float)src.cols;
    hw = (cv::Mat_<float>(1, 2) << ratio_h, ratio_w);
  }

  // Run Infer
  OCTensorList outputs;
  {
    TRITONSERVER_Error* err = nullptr;
    OCTensorList inputs = {OCTensor(std::move(out))};
    TRITONSERVER_InferenceRequest* graph_request = nullptr;
    err = CreateServerRequestWithTensors(
        context->GetBackendRequestInfo(), graph_executor_->GetServer(),
        (graph_names_[0]).c_str(), &inputs, graph_io_names_.input_names,
        graph_io_names_.output_names, &graph_request);
    if (err != nullptr) {
      TRITONSERVER_InferenceRequestDelete(graph_request);
      RETURN_IF_ERROR(err);
    }

    std::future<TRITONSERVER_InferenceResponse*> future;
    err = graph_executor_->AsyncExecute(graph_request, context, &future);
    if (err != nullptr) {
      TRITONSERVER_InferenceRequestDelete(graph_request);
      RETURN_IF_ERROR(err);
    }

    auto* graph_response = future.get();
    err = ParseTensorsFromServerResponse(
        graph_response, graph_io_names_.output_names, &outputs);
    GraphInferResponseDelete(graph_response);
    RETURN_IF_ERROR(err);
  }

  // Run Post
  int rows = outputs[0].m().size[0], cols = outputs[0].m().size[1];
  if (rows == 0 || cols == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "graph infer error");
  }

  const cv::Mat& scores_ori = outputs[0].m().reshape(1, {rows, cols});
  const cv::Mat& geometry_ori = outputs[1].m().reshape(1, {rows, cols, 5});
  const cv::Mat& cos_map_ori = outputs[2].m().reshape(1, {rows, cols});
  const cv::Mat& sin_map_ori = outputs[3].m().reshape(1, {rows, cols});
  float ratio_h = hw.at<float>(0, 0);
  float ratio_w = hw.at<float>(0, 1);

  int num = cv::countNonZero(scores_ori > _score_threshold);
  cv::Mat scores(num, 1, CV_32F), decoded_bboxes(num, 4, CV_32FC2),
      yx_text(num, 2, CV_32S);
  cv::Mat geometry(num, 5, CV_32F), cos_map(num, 1, CV_32F),
      sin_map(num, 1, CV_32F);
  int count = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      auto* ptr_score = scores_ori.ptr<float>(i, j);
      auto* ptr_cosmap = cos_map_ori.ptr<float>(i, j);
      auto* ptr_sinmap = sin_map_ori.ptr<float>(i, j);
      if (*ptr_score > _score_threshold) {
        *scores.ptr<float>(count, 0) = *ptr_score;
        *yx_text.ptr<int>(count, 0) = i;
        *yx_text.ptr<int>(count, 1) = j;
        *cos_map.ptr<float>(count, 0) = *ptr_cosmap;
        *sin_map.ptr<float>(count, 0) = *ptr_sinmap;
        *geometry.ptr<float>(count, 0) = *geometry_ori.ptr<float>(i, j, 0);
        *geometry.ptr<float>(count, 1) = *geometry_ori.ptr<float>(i, j, 1);
        *geometry.ptr<float>(count, 2) = *geometry_ori.ptr<float>(i, j, 2);
        *geometry.ptr<float>(count, 3) = *geometry_ori.ptr<float>(i, j, 3);
        *geometry.ptr<float>(count, 4) = *geometry_ori.ptr<float>(i, j, 4);
        count++;
      }
    }
  }
  // restore_rectangle
  restore_rectangle(yx_text, geometry, decoded_bboxes);
  if (scores.dims == 0) {
    context->SetTensor(
        io_names_.output_names,
        {OCTensor({0, 4, 2}, CV_32F), OCTensor({0}, CV_32F)});
    return nullptr;
  }

  // compute_centerness_targets
  cv::Mat centerness;
  compute_centerness_targets(geometry, centerness);
  cv::Mat scores_ = centerness.mul(scores);
  std::vector<cv::Point2f> xy_text;
  std::vector<std::vector<cv::Point2f>> boxes;
  for (int i = 0; i < decoded_bboxes.size[0]; i++) {
    xy_text.emplace_back(yx_text.at<int>(i, 1) * 4, yx_text.at<int>(i, 0) * 4);
    std::vector<cv::Point2f> box;
    for (int j = 0; j < 4; j++) {
      box.emplace_back(
          decoded_bboxes.at<float>(i, j, 0), decoded_bboxes.at<float>(i, j, 1));
    }
    boxes.emplace_back(std::move(box));
  }

  int n = decoded_bboxes.size[0];
  const float factor = 10000.0;
  cv::Mat decoded_bboxes_ = decoded_bboxes * factor;
  auto* bbs_ptr = reinterpret_cast<float*>(decoded_bboxes_.data);
  auto* scores_ptr = reinterpret_cast<float*>(scores_.data);
  auto polys =
      lanms::merge_quadrangle_n9(bbs_ptr, scores_ptr, n, _nms_threshold);
  std::vector<cv::Point2f> points_vec;
  int ploy_size = (int)polys.size();
  points_vec.reserve(ploy_size * 4);
  ThreadPool& tp = nn_thread_pool();
  std::mutex mutex;
  int batch_num = ceil(float(ploy_size) / 20),
      threads = ceil(ploy_size / float(batch_num));
  std::vector<BoolFuture> rets(threads);
  for (int t = 0; t < threads; ++t) {
    int start = t * batch_num,
        end = std::min(int((t + 1) * batch_num), ploy_size) - 1;
    rets[t] = tp.enqueue(
        [&mutex, &cos_map, &sin_map, &geometry, &yx_text, &xy_text, &boxes,
         &ratio_h, &ratio_w, &polys, &points_vec, &factor, start,
         end]() -> bool {
          for (int j = start; j <= end; j++) {
            const auto& p = polys[j];
            std::vector<cv::Point2f> points, points_tmp;
            for (int i = 0; i < 4; i++) {
              points.emplace_back(p.poly[i].X / factor, p.poly[i].Y / factor);
              points_tmp.emplace_back(
                  p.poly[i].X / factor / 4, p.poly[i].Y / factor / 4);
            }
            // oder point
            float sin, cos;
            calculate_mean_angle(
                cos_map, sin_map, yx_text, points_tmp, sin, cos);
            reorder_start_point(points, cos, sin);
            refine_box(points, xy_text, geometry, boxes, ratio_h, ratio_w);
            {
              std::lock_guard<std::mutex> guard(mutex);
              points_vec.insert(points_vec.end(), points.begin(), points.end());
            }
          }
          return true;
        });
  }
  for (int t = 0; t < threads; t++) {
    rets[t].get();
  }

  cv::Mat output0 = vec2mat(points_vec, 2, ploy_size);
  cv::Mat scores_mat(ploy_size, 1, CV_32F, Scalarf(0.0));
  auto scores_tensor = OCTensor(std::move(scores_mat));
  scores_tensor.set_shape({ploy_size});
  context->SetTensor(
      io_names_.output_names, {std::move(output0), std::move(scores_tensor)});

  return nullptr;
}

}}  // namespace dataelem::alg
