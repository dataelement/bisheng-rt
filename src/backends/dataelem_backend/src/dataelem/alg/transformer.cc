#include "dataelem/alg/transformer.h"

#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"


namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(TransformerAlg);
REGISTER_ALG_CLASS(TransformerTrtAlg);

TRITONSERVER_Error*
TransformerBase::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "transformer";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "fixed_height", &_fixed_height, 32);
  SafeParseParameter(params, "batch_size", &_batch_size, 32);
  SafeParseParameter(params, "input_channels", &_input_channels, 1);
  return nullptr;
}


TRITONSERVER_Error*
TransformerAlg::init(triton::backend::BackendModel* model_state)
{
  TransformerBase::init(model_state);
  alg_name_ = "transformer_alg";
  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  std::string dep_model_name;
  SafeParseParameter(params, "dep_model_name", &dep_model_name, "");
  graph_names_.emplace_back(dep_model_name);

  std::string input_names_str;
  std::vector<std::string> input_names_vec;
  SafeParseParameter(params, "graph_input_name", &input_names_str, "");
  ParseArrayFromString(input_names_str, input_names_vec);
  if (input_names_vec.size() > 0 && input_names_vec[0].size() > 0) {
    graph_io_names_.input_names = input_names_vec;
  }

  std::string output_names_str;
  std::vector<std::string> output_names_vec;
  SafeParseParameter(params, "graph_output_name", &output_names_str, "");
  ParseArrayFromString(output_names_str, output_names_vec);
  if (output_names_vec.size() > 0 && output_names_vec[0].size() > 0) {
    graph_io_names_.output_names = output_names_vec;
  }

  // create input memory buffer
  int cache_n = 2;
  int c = _input_channels == 1 ? 1 : 3;
  int input0_buffer_size = cache_n * _batch_size * _fixed_height * 1000 * c * 4;
  std::vector<int> input0_shape({cache_n, _batch_size, _fixed_height, 1000, c});
  triton::backend::BackendMemory* input0_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
      input0_buffer_size, &input0_memory));

  input0_buffer_.reset(input0_memory);
  input0_ = cv::Mat(input0_shape, CV_32F, input0_buffer_->MemoryPtr());

  int input1_buffer_size = cache_n * _batch_size * 2 * 4;
  std::vector<int> input1_shape({cache_n, _batch_size, 2});
  triton::backend::BackendMemory* input1_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
      input1_buffer_size, &input1_memory));

  input1_buffer_.reset(input1_memory);
  input1_ = cv::Mat(input1_shape, CV_32F, input1_buffer_->MemoryPtr());

  return nullptr;
}

TRITONSERVER_Error*
TransformerAlg::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  // images: (n, h, w_max, 3), images_shape: (n, 2)
  auto& images = inputs[0].m();
  auto& images_shape = inputs[1].m();

  // split into batches and call infer
  int n = images.size[0];
  int h = images.size[1];
  int w_max = images.size[2];
  int c = images.size[3];
  auto src_type = c == 3 ? CV_8UC3 : CV_8UC1;
  auto dst_type = c == 3 ? CV_32FC3 : CV_32FC1;

  std::vector<std::pair<int, int>> temp_widths;
  for (int i = 0; i < n; i++) {
    temp_widths.emplace_back(images_shape.at<int>(i, 1), i);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first > p2.first;
      });

  int batchs = std::ceil(n * 1.0 / _batch_size);
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(batchs);

  int pinned_cnt = input0_.size[0];
  // pinned_cnt = -1;
  std::vector<cv::Mat> rois;
  std::vector<cv::Mat> widths;
  // transctc输入batchsize=1时seq_len不是vector
  int flag = 0;
  for (int k = 0; k < batchs; k++) {
    int s = k * _batch_size;
    int e = k == (batchs - 1) ? n : (k + 1) * _batch_size;
    int sn = e - s;
    if (sn == 1) {
      flag = 1;
    }
    int padded_width = temp_widths[s].first + _extra_padding_length;

    std::vector<int> roi_shape = {sn + flag, h, padded_width, c};
    cv::Mat roi, width;
    if (k < pinned_cnt) {
      roi = cv::Mat(roi_shape, dst_type, input0_.data + input0_.step[0] * k);
      roi.setTo(0.0f);
      width = cv::Mat(sn + flag, 2, CV_32S, input1_.data + input1_.step[0] * k);
    } else {
      roi = cv::Mat(roi_shape, dst_type, Scalarf(0.0));
      width = cv::Mat(sn + flag, 2, CV_32S);
    }

    // create batch
    for (int i = 0; i < sn; ++i) {
      int j = temp_widths[s + i].second;
      int w = images_shape.at<int>(j, 1);
      cv::Mat img(h, w_max, src_type, images.data + images.step[0] * j);
      cv::Mat src = img(cv::Rect(0, 0, w, h));
      cv::Mat dst(h, padded_width, dst_type, roi.data + roi.step[0] * i);
      src.convertTo(dst(cv::Rect(0, 0, w, h)), CV_32F, 1.0 / 255.0);
      // dst(cv::Rect(w, 0, padded_width - w, h)) = Scalarf(0.0);
      width.at<int>(i, 0) = h;
      width.at<int>(i, 1) = w;
    }

    rois.emplace_back(std::move(roi));
    widths.emplace_back(std::move(width));
    OCTensorList inputs = {rois[k], widths[k]};
    if (k < pinned_cnt) {
      inputs[0].set_pinned();
      inputs[1].set_pinned();
    }

    // create request
    TRITONSERVER_InferenceRequest* req = nullptr;
    RETURN_IF_ERROR(CreateServerRequestWithTensors(
        context->GetBackendRequestInfo(), graph_executor_->GetServer(),
        graph_names_[0].c_str(), &inputs, graph_io_names_.input_names,
        graph_io_names_.output_names, &req));
    RETURN_IF_ERROR(graph_executor_->AsyncExecute(req, context, &futures[k]));
  }

  // parse result
  std::vector<absl::string_view> tmp_texts;
  std::vector<float> scores;
  std::vector<TRITONSERVER_InferenceResponse*> resps;
  for (int k = 0; k < batchs; k++) {
    auto* resp = futures[k].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, graph_io_names_.output_names, &outputs));
    for (int64_t i = 0; i < outputs[0].shape()[0]; i++) {
      tmp_texts.push_back(outputs[0].GetString(i));
    }

    auto tmp_scores = std::vector<float>(outputs[1].m());
    scores.insert(scores.end(), tmp_scores.begin(), tmp_scores.end());
    resps.push_back(resp);
  }

  if (flag) {
    tmp_texts.pop_back();
    scores.pop_back();
  }

  // reorder texts
  std::vector<absl::string_view> texts(n);
  for (int i = 0; i < n; i++) {
    texts[temp_widths[i].second] = tmp_texts[i];
  }

  auto scores_mat =
      (n > 0 ? cv::Mat(scores).reshape(1, n).clone()
             : cv::Mat(n, 1, CV_32F, Scalarf(0.0)));

  auto scores_tensor = OCTensor(std::move(scores_mat));
  scores_tensor.set_shape({n});
  context->SetTensor(
      io_names_.output_names,
      {std::move(OCTensor(texts, {n})), std::move(scores_tensor)});

  for (int k = 0; k < batchs; k++) {
    GraphInferResponseDelete(resps[k]);
  }
  return nullptr;
}


TRITONSERVER_Error*
TransformerTrtAlg::init(triton::backend::BackendModel* model_state)
{
  TransformerBase::init(model_state);
  alg_name_ = "transformer_trt_alg";
  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  std::string dep_model_name;
  SafeParseParameter(params, "dep_model_name", &dep_model_name, "");
  ParseArrayFromString(dep_model_name, graph_names_);
  // graph_names_.emplace_back(dep_model_name);

  // create input memory buffer
  int cache_n = 2;
  int c = _input_channels == 1 ? 1 : 3;
  int input0_buffer_size = cache_n * _batch_size * _fixed_height * 1000 * c * 4;
  std::vector<int> input0_shape({cache_n, _batch_size, _fixed_height, 1000, c});
  triton::backend::BackendMemory* input0_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
      input0_buffer_size, &input0_memory));

  input0_buffer_.reset(input0_memory);
  input0_ = cv::Mat(input0_shape, CV_32F, input0_buffer_->MemoryPtr());

  int input1_buffer_size = cache_n * _batch_size * 2 * 4;
  std::vector<int> input1_shape({cache_n, _batch_size, 2});
  triton::backend::BackendMemory* input1_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
      input1_buffer_size, &input1_memory));

  input1_buffer_.reset(input1_memory);
  input1_ = cv::Mat(input1_shape, CV_32F, input1_buffer_->MemoryPtr());

  return nullptr;
}

TRITONSERVER_Error*
TransformerTrtAlg::Execute(AlgRunContext* context)
{
  OCTensorList inputs;
  context->GetTensor(io_names_.input_names, inputs);
  // images: (n, h, w_max, 3), images_shape: (n, 2)
  auto& images = inputs[0].m();
  auto& images_shape = inputs[1].m();

  // split into batches and call infer
  int n = images.size[0];
  int h = images.size[1];
  int w_max = images.size[2];
  int c = images.size[3];
  auto src_type = c == 3 ? CV_8UC3 : CV_8UC1;
  auto dst_type = c == 3 ? CV_32FC3 : CV_32FC1;

  std::vector<std::pair<int, int>> temp_widths;
  for (int i = 0; i < n; i++) {
    temp_widths.emplace_back(images_shape.at<int>(i, 1), i);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first > p2.first;
      });

  int batchs = std::ceil(n * 1.0 / _batch_size);

  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(batchs);
  int pinned_cnt = input0_.size[0];
  std::vector<cv::Mat> rois;
  std::vector<cv::Mat> widths;

  for (int k = 0; k < batchs; k++) {
    int s = k * _batch_size;
    int e = k == (batchs - 1) ? n : (k + 1) * _batch_size;
    int sn = e - s;
    int padded_width = temp_widths[s].first + _extra_padding_length;

    std::vector<int> roi_shape = {sn, h, padded_width, c};
    cv::Mat roi, width;
    if (k < pinned_cnt) {
      roi = cv::Mat(roi_shape, dst_type, input0_.data + input0_.step[0] * k);
      roi.setTo(0.0f);
      width = cv::Mat(sn, 2, CV_32S, input1_.data + input1_.step[0] * k);
    } else {
      roi = cv::Mat(roi_shape, dst_type, Scalarf(0.0));
      width = cv::Mat(sn, 2, CV_32S);
    }

    // create batch
    for (int i = 0; i < sn; ++i) {
      int j = temp_widths[s + i].second;
      int w = images_shape.at<int>(j, 1);
      cv::Mat img(h, w_max, src_type, images.data + images.step[0] * j);
      cv::Mat src = img(cv::Rect(0, 0, w, h));
      cv::Mat dst(h, padded_width, dst_type, roi.data + roi.step[0] * i);
      src.convertTo(dst(cv::Rect(0, 0, w, h)), CV_32F, 1.0 / 255.0);
      // dst(cv::Rect(w, 0, padded_width - w, h)) = Scalarf(0.0);
      width.at<int>(i, 0) = h;
      width.at<int>(i, 1) = w;
    }

    rois.emplace_back(std::move(roi));
    widths.emplace_back(std::move(width));
    OCTensorList inputs = {rois[k], widths[k]};
    if (k < pinned_cnt) {
      inputs[0].set_pinned();
      inputs[1].set_pinned();
    }

    // create graph request
    TRITONSERVER_InferenceRequest* req = nullptr;
    RETURN_IF_ERROR(CreateServerRequestWithTensors(
        context->GetBackendRequestInfo(), graph_executor_->GetServer(),
        graph_names_[0].c_str(), &inputs, graph_io_names_.input_names,
        graph_io_names_.output_names, &req));
    RETURN_IF_ERROR(graph_executor_->AsyncExecute(req, context, &futures[k]));
  }

  // parse result
  std::vector<absl::string_view> tmp_texts;
  std::vector<float> scores;

  // std::vector<std::future<TRITONSERVER_InferenceResponse*>> post_futures(
  //     batchs);

  // reverse get, short batch in large index
  // // for (int k = 0; k < batchs; k++) {
  // for (int k = batchs - 1; k >= 0; k--) {
  //   auto* resp = futures[k].get();
  //   OCTensorList outputs;
  //   RETURN_IF_ERROR(ParseTensorsFromServerResponse(
  //       resp, graph_io_names_.output_names, &outputs));

  //   // create post request
  //   TRITONSERVER_InferenceRequest* req = nullptr;
  //   RETURN_IF_ERROR(CreateServerRequestWithTensors(
  //       context->GetBackendRequestInfo(), graph_executor_->GetServer(),
  //       graph_names_[1].c_str(), &outputs, post_graph_io_names_.input_names,
  //       post_graph_io_names_.output_names, &req));
  //   RETURN_IF_ERROR(
  //       graph_executor_->AsyncExecute(req, context, &post_futures[k]));

  //   GraphInferResponseDelete(resp);
  // }

  // for (int k = 0; k < batchs; k++) {
  //   auto* resp = post_futures[k].get();
  //   OCTensorList outputs;
  //   RETURN_IF_ERROR(ParseTensorsFromServerResponse(
  //       resp, post_graph_io_names_.output_names, &outputs));

  //   for (int64_t i = 0; i < outputs[0].shape()[0]; i++) {
  //     tmp_texts.push_back(outputs[0].GetString(i));
  //   }
  //   GraphInferResponseDelete(resp);
  // }

  for (int k = 0; k < batchs; k++) {
    auto* resp = futures[k].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, graph_io_names_.output_names, &outputs));

    for (int64_t i = 0; i < outputs[0].shape()[0]; i++) {
      tmp_texts.push_back(outputs[0].GetString(i));
    }
    GraphInferResponseDelete(resp);
  }

  // reorder texts
  std::vector<absl::string_view> texts(n);
  for (int i = 0; i < n; i++) {
    texts[temp_widths[i].second] = tmp_texts[i];
  }

  // auto scores_mat = cv::Mat(scores).reshape(1, n).clone();
  cv::Mat scores_mat(n, 1, CV_32F, Scalarf(0.0));
  auto scores_tensor = OCTensor(std::move(scores_mat));
  scores_tensor.set_shape({n});

  context->SetTensor(
      io_names_.output_names,
      {std::move(OCTensor(texts, {n})), std::move(scores_tensor)});

  return nullptr;
}
}}  // namespace dataelem::alg
