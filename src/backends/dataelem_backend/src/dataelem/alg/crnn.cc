#include "dataelem/alg/crnn.h"

#include <array>

#include "dataelem/alg/lanms.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/alg_factory.h"

namespace dataelem { namespace alg {

REGISTER_ALG_CLASS(CRNN);

int
resize_image(
    const cv::Mat& img1, cv::Mat& img5, int H, int W_min, int W_max,
    bool is_grayscale, int downsample_rate, int extra_padding_length, bool is_pd_model)
{
  if (W_min % downsample_rate != 0 || W_max % downsample_rate != 0) {
    return -1;
  }
  cv::Mat img2;
  // very slow here
  if (is_grayscale) {
    bgr2grayOp(img1, img2);
  } else {
    bgr2rgbOp(img1, img2);
  }

  int h = img2.rows;
  int w = img2.cols;

  int w2 = std::max((H * w) / h, 1);
  int W = std::ceil(w2 * 1.0 / downsample_rate) * downsample_rate;

  cv::Mat img3;
  if (W <= W_max && W >= W_min) {
    resizeOp(img2, img3, W, H);  // img3.shape [H, W, 1]

  } else if (W < W_min) {
    cv::Mat img2prime;
    resizeOp(img2, img2prime, W, H);
    cv::copyMakeBorder(
        img2prime, img3, 0, 0, 0, W_min - W, cv::BORDER_CONSTANT, 0);
  } else {
    W = W_max;
    cv::Mat img2prime;
    int h2 = std::max((W * h) / w, 1);
    resizeOp(img2, img2prime, W, h2);
    int margin = (H - h2) / 2;
    int remainder = (H - h2) % 2;
    int bottom = margin + remainder;
    int top = margin;
    cv::copyMakeBorder(
        img2prime, img3, top, bottom, 0, 0, cv::BORDER_CONSTANT, 0);
  }

  cv::Mat img4;
  cv::copyMakeBorder(
      img3, img4, 0, 0, 0, extra_padding_length, cv::BORDER_CONSTANT, 0);
  
  if (is_pd_model){
    img4.convertTo(img5, CV_32FC3, 2.0 / 255.0, -1.0);
  }else{
    img4.convertTo(img5, CV_32FC3, 1.0 / 255.0, 0.0);
  }
  return 0;
}

void
preprocess_recog_batch(
    const std::vector<cv::Mat>& imgs, cv::Mat& rois, cv::Mat& shapes, int H,
    int W_min, int W_max, bool is_grayscale, int downsample_rate,
    int extra_padding_length, bool is_pd_model, int pinned_cnt, int k, cv::Mat& buff0,
    cv::Mat& buff1)
{
  int channels = is_grayscale ? 1 : 3;
  std::vector<cv::Mat> imgs_preprocessed;
  std::vector<int> shapes_preprocessed;

  int shape_max = 0;
  for (auto img0 : imgs) {
    cv::Mat img5;
    resize_image(
        img0, img5, H, W_min, W_max, is_grayscale, downsample_rate,
        extra_padding_length, is_pd_model);

    shape_max = std::max(shape_max, img5.cols);
    shapes_preprocessed.emplace_back(img5.rows);
    shapes_preprocessed.emplace_back(img5.cols - extra_padding_length);
    imgs_preprocessed.emplace_back(std::move(img5));
  }

  int batchsize = imgs_preprocessed.size();
  std::vector<int> size = {batchsize, channels, H, shape_max};
  if (k < pinned_cnt) {
    rois = cv::Mat(size, CV_32F, buff0.data + buff0.step[0] * k);
    rois.setTo(0.0f);
    shapes = cv::Mat(size[0], 2, CV_32S, buff1.data + buff1.step[0] * k);
  } else {
    rois = cv::Mat(size.size(), size.data(), CV_32F, Scalarf(0.0));
    shapes = cv::Mat(size[0], 2, CV_32S);
  }

  int volumn_bytes = H * shape_max * channels * 4;
  for (int i = 0; i < batchsize; i++) {
    cv::Mat img_dst = cv::Mat(channels, H * shape_max, CV_32FC1, rois.data + volumn_bytes * i);
    cv::Mat img_src0 = cv::Mat(H, shape_max, CV_32FC3, cv::Scalar(0.0f));
    imgs_preprocessed[i].copyTo(img_src0(
        cv::Rect(0, 0, imgs_preprocessed[i].cols, imgs_preprocessed[i].rows)));
    cv::Mat img_src = cv::Mat(H * shape_max, channels, CV_32FC1, img_src0.data);
    cv::transpose(img_src, img_dst);
  }

  cv::Mat(shapes_preprocessed).reshape(0, batchsize).copyTo(shapes);
}


// 20201027 hjt modification #3 and #4
// 2023.03.25 hf modification, support pinned memory buffer
void
CRNNPreprocessOp3(
    const std::vector<cv::Mat>& srcs, int batch_size, int fixed_h,
    int output_channels, int downsample_rate, int W_min, int W_max, 
    bool is_pd_model, int device_count, PairMatList& dst, 
    std::vector<int>& dst_indexes, cv::Mat& buff0, cv::Mat& buff1)
{
  // #3 stable sort
  int n = srcs.size();
  // const int MIN_WIDTH = 40;
  std::vector<std::pair<int, int>> temp_widths;
  for (int i = 0; i < n; i++) {
    const auto& mat = srcs[i];
    int new_w = int(fixed_h * 1.0 / mat.rows * mat.cols);
    temp_widths.emplace_back(new_w, i);
  }

  std::stable_sort(
      temp_widths.begin(), temp_widths.end(),
      [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return p1.first > p2.first;
      });
  for (const auto& v : temp_widths) {
    dst_indexes.push_back(v.second);
  }

  // this logic will changed to dynamic config
  int selected_batchs, selected_batch_size;
  if (device_count <= 1) {
    // cpu mode: device_count = 0; gpu mode, only one device
    selected_batch_size = batch_size;
    selected_batchs = std::ceil(n * 1.0 / batch_size);
  } else {
    // gpu mode
    if (n <= batch_size * device_count) {
      selected_batch_size = std::ceil(n * 1.0 / device_count);
      selected_batchs = n <= device_count ? n : device_count;
    } else {
      selected_batch_size = batch_size;
      selected_batchs = std::ceil(n * 1.0 / batch_size);
    }
  }

  int pinned_cnt = buff0.size[0];

  // #4 new preprocessing function
  for (int k = 0; k < selected_batchs; k++) {
    int s = k * selected_batch_size;
    int e = k == (selected_batchs - 1) ? n : (k + 1) * selected_batch_size;
    int sn = e - s;
    cv::Mat rois;
    cv::Mat widths;
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < sn; ++i) {
      imgs.emplace_back(srcs[temp_widths[s + i].second]);
    }
    int extra_padding_length = 0;
    bool is_grayscale = output_channels == 1 ? true : false;

    preprocess_recog_batch(
        imgs, rois, widths, fixed_h, W_min, W_max, is_grayscale,
        downsample_rate, extra_padding_length, is_pd_model, pinned_cnt, k, buff0, buff1);
    dst.emplace_back(rois, widths);
  }
}

std::vector<std::string>
ReadDict(const std::string& path)
{
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

TRITONSERVER_Error*
CRNN::init(triton::backend::BackendModel* model_state)
{
  Algorithmer::init(model_state);
  alg_name_ = "crnn";

  auto& model_config = model_state->ModelConfig();
  JValue params;
  model_config.Find("parameters", &params);
  SafeParseParameter(params, "fixed_height", &_fixed_height, 32);
  SafeParseParameter(params, "batch_size", &_batch_size, 32);
  SafeParseParameter(params, "input_channels", &_input_channels, 3);
  SafeParseParameter(params, "downsample_rate", &_downsample_rate, 8);
  SafeParseParameter(params, "W_min", &_W_min, 40);
  SafeParseParameter(params, "W_max", &_W_max, 800);

  SafeParseParameter(params, "output_matrix", &_output_matrix, false);
  SafeParseParameter(params, "is_pd_model", &_is_pd_model, false);
  
  std::string model_name_str;
  SafeParseParameter(params, "dep_model_name", &model_name_str, "");
  ParseArrayFromString(model_name_str, graph_names_);

  if (_output_matrix) {
    graph_io_names_ = StepConfig({"x"}, {"softmax_5.tmp_0"});
  } else {
    graph_io_names_ = StepConfig({"x"}, {"feat_ind", "feat_prob"});
  }

  _long_image_segmentor.reset(new LongImageSegment());

  _recog_ins_num = 1;

  std::string charset_name = "character_dict.txt";
  SafeParseParameter(params, "charset_name", &charset_name);
  auto repo_path = model_state->RepositoryPath();

  std::string label_path =
      triton::common::JoinPath({repo_path, "../../..", "charsets", charset_name});

  bool is_exist = false;
  triton::common::FileExists(label_path, &is_exist);
  if (!is_exist) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "character file not exists.");
  }

  _label_list = ReadDict(label_path);
  _label_list.insert(_label_list.begin(), "#");
  _label_list.push_back(" ");

  // create pinned input memory buffer
  int cache_n = 2;
  int c = _input_channels == 1 ? 1 : 3;
  int input0_buffer_size = cache_n * _batch_size * _fixed_height * 1000 * c * 4;
  std::vector<int> input0_shape({cache_n, _batch_size, _fixed_height, 1000, c});
  triton::backend::BackendMemory* input0_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
#ifdef TRITON_ENABLE_GPU
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
#else
      {triton::backend::BackendMemory::AllocationType::CPU}, 0,
#endif
      input0_buffer_size, &input0_memory));

  input0_buffer_.reset(input0_memory);
  input0_ = cv::Mat(input0_shape, CV_32F, input0_buffer_->MemoryPtr());

  int input1_buffer_size = cache_n * _batch_size * 2 * 4;
  std::vector<int> input1_shape({cache_n, _batch_size, 2});
  triton::backend::BackendMemory* input1_memory;
  RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
      model_state->TritonMemoryManager(),
#ifdef TRITON_ENABLE_GPU
      {triton::backend::BackendMemory::AllocationType::CPU_PINNED}, 0,
#else
      {triton::backend::BackendMemory::AllocationType::CPU}, 0,
#endif
      input1_buffer_size, &input1_memory));

  input1_buffer_.reset(input1_memory);
  input1_ = cv::Mat(input1_shape, CV_32F, input1_buffer_->MemoryPtr());
  return nullptr;
}

TRITONSERVER_Error*
CRNN::Execute(AlgRunContext* context)
{
  TRITONSERVER_Error* err = nullptr;
  OCTensorList input_tensors;
  context->GetTensor(io_names_.input_names, input_tensors);

  // parse optional params
  rapidjson::Document d;
  OCTensor* params_tensor;
  if (context->GetTensor(optional_inputs_[0], &params_tensor)) {
    auto buffer = params_tensor->GetString(0);
    d.Parse(buffer.data(), buffer.length());
    if (d.HasParseError()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "json parsing error on params");
    }
  }
  APIData params(d);
  // mode:1 image + bboxes, mode:0 patches
  params.add("mode", 1);

  int imread_flag = cv::IMREAD_COLOR;
  std::string data_format;
  get_ad_value<std::string>(params, "data_format", data_format);
  if (data_format.compare("gray") == 0) {
    imread_flag = cv::IMREAD_GRAYSCALE;
  }

  // parse optional patchs
  int bb_count = 0;
  MatList inputs;
  OCTensor* patchs_tensor;
  std::vector<absl::string_view> patchs_data;
  if (context->GetTensor(optional_inputs_[1], &patchs_tensor)) {
    patchs_tensor->GetStrings(patchs_data);
    inputs.resize(patchs_data.size());
    for (size_t i = 0; i < patchs_data.size(); i++) {
      err = DecodeImgFromB64(patchs_data[i], inputs[i], imread_flag);
      RETURN_IF_ERROR(err);
    }
    bb_count = patchs_data.size();
    params.add("mode", 0);
  } else {
    bb_count = input_tensors[1].shape_ptr()[0];
    inputs.emplace_back(input_tensors[0].GetImage());
    inputs.emplace_back(input_tensors[1].GetMat());
  }

  if (bb_count == 0) {
    OCTensor t0({0}, TRITONOPENCV_TYPE_STRING);
    OCTensor t1({0}, TRITONOPENCV_TYPE_FP32);
    context->SetTensor(io_names_.output_names, {std::move(t0), std::move(t1)});
    return nullptr;
  }

  std::vector<absl::string_view> post_texts;
  std::vector<float> post_scores;
  PairMatList prep_outputs;
  IntegerList index;
  IntegerList groups;
  std::vector<std::string> texts;
  std::vector<float> scores;
  RETURN_IF_ERROR(PreprocessStep(params, inputs, prep_outputs, index, groups));
  RETURN_IF_ERROR(GraphStep(context, prep_outputs, texts, scores));
  RETURN_IF_ERROR(
      PostprocessStep(params, index, texts, scores, post_texts, post_scores));

  // Support long image merge.
  bool support_long_image_segment = false;
  get_ad_value<bool>(
      params, "support_long_image_segment", support_long_image_segment);
  if (support_long_image_segment) {
    bool split_long_sentence_blank = false;
    get_ad_value<bool>(
        params, "split_long_sentence_blank", split_long_sentence_blank);

    std::vector<std::string> merged_texts;
    std::vector<float> merged_scores;
    _long_image_segmentor->merge_v2(
        post_texts, post_scores, groups, split_long_sentence_blank,
        merged_texts, merged_scores);

    int n = merged_texts.size();
    auto scores_mat =
        (n > 0 ? cv::Mat(merged_scores).reshape(1, n).clone()
               : cv::Mat(n, 1, CV_32F, Scalarf(0.0)));
    auto scores_tensor = OCTensor(std::move(scores_mat));
    scores_tensor.set_shape({n});
    context->SetTensor(
        io_names_.output_names,
        {std::move(OCTensor(merged_texts, {n})), std::move(scores_tensor)});
    return nullptr;
  }

  int n = post_texts.size();
  auto scores_mat =
      (n > 0 ? cv::Mat(post_scores).reshape(1, n).clone()
             : cv::Mat(n, 1, CV_32F, Scalarf(0.0)));
  auto scores_tensor = OCTensor(std::move(scores_mat));
  scores_tensor.set_shape({n});
  context->SetTensor(
      io_names_.output_names,
      {std::move(OCTensor(post_texts, {n})), std::move(scores_tensor)});

  return nullptr;
}

TRITONSERVER_Error*
CRNN::PreprocessStep(
    const APIData& params, const MatList& inputs, PairMatList& outputs,
    IntegerList& index, IntegerList& groups)
{
  const float DEFAULT_THRESHOLD = 0.4;
  float nique_threshold;
  int mode;
  get_ad_value<float>(
      params, "nique_threshold", nique_threshold, DEFAULT_THRESHOLD);
  get_ad_value<int>(params, "mode", mode, 1);

  // 20201027 hjt modification
  MatList rois;
  if (mode == 1) {
    getRRectRoisWithPaddingOp5(inputs[0], inputs[1], rois);
  } else {
    rois.insert(rois.end(), inputs.begin(), inputs.end());
  }

  // 2023.03.19, hf modified, support long image segmentation
  bool support_long_image_segment = false;
  get_ad_value<bool>(
      params, "support_long_image_segment", support_long_image_segment);
  if (support_long_image_segment) {
    MatList patchs;
    _long_image_segmentor->segment_v2(rois, patchs, groups);
    CRNNPreprocessOp3(
        patchs, _batch_size, _fixed_height, _input_channels, _downsample_rate,
        _W_min, _W_max, _is_pd_model, _recog_ins_num, outputs, index, input0_, input1_);
    return nullptr;
  }

  CRNNPreprocessOp3(
      rois, _batch_size, _fixed_height, _input_channels, _downsample_rate,
      _W_min, _W_max, _is_pd_model, _recog_ins_num, outputs, index, input0_, input1_);

  return nullptr;
}

TRITONSERVER_Error*
CRNN::GraphStep(
    AlgRunContext* context, const PairMatList& inputs,
    std::vector<std::string>& texts, std::vector<float>& scores)
{
  TRITONSERVER_Error* err = nullptr;
  // send request
  size_t n = inputs.size();
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(n);
  for (size_t i = 0; i < n; i++) {
    OCTensorList input_tensors = {inputs[i].first};
    if (n < (size_t)input0_.size[0]) {
      input_tensors[0].set_pinned();
    }

    err = GraphExecuate(
        graph_names_[0], context, graph_io_names_.input_names,
        graph_io_names_.output_names, input_tensors, &futures[i]);
    RETURN_IF_ERROR(err);
  }

  // parse result
  for (size_t k = 0; k < n; k++) {
    auto* resp = futures[k].get();
    OCTensorList outputs;
    RETURN_IF_ERROR(ParseTensorsFromServerResponse(
        resp, graph_io_names_.output_names, &outputs));

    cv::Mat feat_ind, feat_prob;
    if (_output_matrix) {
      auto featmap = outputs[0].m().clone();
      float* featmap_data = (float*)featmap.data;
      feat_ind =
          cv::Mat(featmap.size[0], featmap.size[1], CV_32S, cv::Scalar(0));
      feat_prob =
          cv::Mat(featmap.size[0], featmap.size[1], CV_32F, cv::Scalar(0.0f));
      int step0 = featmap.size[1] * featmap.size[2];
      int step1 = featmap.size[2];
      for (int i = 0; i < featmap.size[0]; i++) {
        float* data0 = featmap_data + i * step0;
        for (int j = 0; j < featmap.size[1]; j++) {
          float* data1 = data0 + j * step1;
          double min_val, max_val;
          cv::Point2i min_loc;
          cv::Point2i max_loc;
          cv::Mat m = cv::Mat(1, (int)featmap.size[2], CV_32FC1, data1);
          cv::minMaxLoc(m, &min_val, &max_val, &min_loc, &max_loc);
          feat_ind.at<int>(i, j) = max_loc.x;
          feat_prob.at<float>(i, j) = (float)max_val;
        }
      }
    } else {
      feat_ind = outputs[0].m().clone();
      feat_prob = outputs[1].m().clone();
    }

    for (int64_t j = 0; j < feat_ind.size[0]; j++) {
      float prob = 0.0f;
      int cnt = 0;
      std::string text;
      for (int64_t i = 0; i < feat_ind.size[1]; i++) {
        int ind = feat_ind.at<int>(j, i);
        if (ind == 0 || (i > 0 && ind == feat_ind.at<int>(j, i - 1))) {
          continue;
        }

        text += _label_list[ind];
        prob += feat_prob.at<float>(j, i);
        cnt++;
      }

      texts.push_back(text);
      if (cnt == 0) {
        scores.push_back(0.0);
      } else {
        scores.push_back(prob / cnt);
      }
    }

    GraphInferResponseDelete(resp);
  }

  return nullptr;
}

TRITONSERVER_Error*
CRNN::PostprocessStep(
    const APIData& params, const IntegerList& index,
    const std::vector<std::string>& input_texts,
    const std::vector<float>& input_scores,
    std::vector<absl::string_view>& output_texts,
    std::vector<float>& output_scores)
{
  output_texts.resize(input_texts.size());
  output_scores.resize(input_scores.size());
  for (size_t j = 0; j < input_texts.size(); j++) {
    output_texts[index[j]] = input_texts[j];
  }

  for (size_t j = 0; j < input_scores.size(); j++) {
    output_scores[index[j]] = input_scores[j];
  }

  return nullptr;
}


}}  // namespace dataelem::alg