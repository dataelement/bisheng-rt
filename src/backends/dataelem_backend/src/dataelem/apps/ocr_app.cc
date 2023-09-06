#include "dataelem/apps/ocr_app.h"

#include "absl/strings/escaping.h"
#include "dataelem/common/apidata.h"
#include "dataelem/common/mat_utils.h"
#include "dataelem/framework/app_factory.h"

namespace dataelem { namespace alg {

REGISTER_APP_CLASS(OCRApp);


TRITONSERVER_Error*
OCRApp::init(triton::backend::BackendModel* model_state)
{
  Application::init(model_state);
  app_name_ = "OCRApp";
  return nullptr;
}


TRITONSERVER_Error*
OCRApp::Execute(AlgRunContext* context, std::string* resp)
{
  try {
    TRITONSERVER_Error* err = nullptr;
    AppRequestInfo request_info;
    auto timer = triton::common::Timer();

    // Deserialize the input
    OCTensor* input = nullptr;
    context->GetTensor("INPUT", &input);
    auto data_ptr = const_cast<char*>(input->data_ptr());
    rapidjson::Document d;
    d.ParseInsitu(data_ptr);
    if (d.HasParseError()) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "json parsing error on reqest body");
    }

    APIData ad(d);
    if (err == nullptr && !ad.has("param")) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "must contain param in request");
    }

    if (err != nullptr) {
      *resp = WriteErrorResponse(err);
      return nullptr;
    }

    APIData param = ad.getobj("param");
    std::string det_name = "", recog_name = "", table_name = "", cls_name = "";
    std::string checkbox_name = "";
    std::string uie_name = "", uie_schema = "";
    std::vector<std::string> checkbox_names;
    std::vector<std::string> auto_recog_names;
    bool enable_format = true;
    bool image_fix, compress_image, enable_gray, enhance_constract_clarity;
    bool perspective_correct, erase_seal, erase_watermark, detect_ps, remove_moire;
    std::string binarize_method, cut_border;
    std::string seal_name = "";
    try {
      get_ad_value<std::string>(param, "det", det_name);
      get_ad_value<std::string>(param, "recog", recog_name);
      get_ad_value<std::string>(param, "table", table_name);
      get_ad_value<std::string>(param, "cls", cls_name);
      get_ad_value<std::string>(param, "ellm", uie_name);
      get_ad_value<std::string>(param, "ellm_schema", uie_schema);
      get_ad_value<std::vector<std::string>>(param, "checkbox", checkbox_names);
      if (checkbox_names.size() > 0) {
        checkbox_name = checkbox_names[0];
      }

      get_ad_value<std::vector<std::string>>(
          param, "auto_hand_print_params", auto_recog_names);

      get_ad_value<bool>(param, "enable_format", enable_format);
      get_ad_value<bool>(param, "image_fix", image_fix, false);
      get_ad_value<bool>(param, "compress_image", compress_image, false);
      get_ad_value<bool>(param, "enable_gray", enable_gray, false);
      get_ad_value<bool>(param, "enhance_constract_clarity", enhance_constract_clarity, false);
      get_ad_value<bool>(param, "perspective_correct", perspective_correct, false);
      get_ad_value<bool>(param, "erase_seal", erase_seal, false);
      get_ad_value<bool>(param, "erase_watermark", erase_watermark, false);
      get_ad_value<bool>(param, "detect_ps", detect_ps, false);
      get_ad_value<bool>(param, "remove_moire", remove_moire, false);
      get_ad_value<std::string>(param, "binarize_method", binarize_method, "");
      get_ad_value<std::string>(param, "cut_border", cut_border, "");
      get_ad_value<std::string>(param, "seal", seal_name);
    }
    catch(const std::exception& ex){
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "params error, please have a check!");
      *resp = WriteErrorResponse(err);
      return nullptr;
    }

    bool general_prep = false;
    if(image_fix || compress_image || enable_gray || enhance_constract_clarity || perspective_correct 
    || erase_seal || erase_watermark || detect_ps || remove_moire || binarize_method.size() > 0 || cut_border.size() > 0){
      general_prep = true;
    }

    std::string format_model_name = "ocr_format_elem";
    std::string checkbox_post_model_name = "checkbox_post";
    std::string general_prep_model_name = "general_prep";

    if (param.has("request_id")) {
      request_info.request_id = param.get("request_id").get<int64_t>();
    } else {
      SET_MILLI_TIMESTAMP(request_info.request_id);
    }
    // set backend request id
    std::string backend_reqid = std::to_string(request_info.request_id);
    context->GetBackendRequestInfo()->request_id = backend_reqid.c_str();
    bool ellm_only = false;
    if (uie_name.length() > 0 && ad.has("ocr")) {
      ellm_only = true;
    }
    if (det_name.length() == 0 && recog_name.length() == 0 && !ellm_only && !general_prep) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "must specify at least det, recog, ellm or general_prep");
      *resp = WriteErrorResponse(err, &request_info);
      return nullptr;
    }

    if (det_name.length() == 0 && recog_name.length() == 0 && !ellm_only && general_prep) {
      enable_format = false;
    }

    // Step 0. parse the input data
    int imread_flag = cv::IMREAD_COLOR;
    std::string data_format;
    get_ad_value<std::string>(ad, "data_format", data_format);
    if (data_format.compare("gray") == 0) {
      imread_flag = cv::IMREAD_GRAYSCALE;
    }

    bool recog_only = det_name.empty() && !recog_name.empty() && !general_prep;
    if (!ad.has("data")) {
      err =
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "data is required");
      *resp = WriteErrorResponse(err);
      return nullptr;
    }
    auto data_vec = ad.get("data").get<std::vector<std::string>>();
    if (data_vec.size() == 0) {
      err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "data empty");
    }

    bool enable_ocr_res = false;
    std::vector<std::string> ocr_results;
    if (ad.has("ocr")) {
      enable_ocr_res = true;
      ocr_results = ad.get("ocr").get<std::vector<std::string>>();
      if (ocr_results.size() == 0) {
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "ocr empty");
      }
    }

    MatList imgs(data_vec.size());
    if (!recog_only && !general_prep) {
      for (size_t i = 0; i < data_vec.size(); i++) {
        err = DecodeImgFromB64(data_vec[i], imgs[i], imread_flag);
        CHECK_ERROR_WITH_BREAK(err);
      }
    }

    if (err != nullptr) {
      *resp = WriteErrorResponse(err, &request_info);
      return nullptr;
    }

    // Step 1. Do Det and Recog
    std::string param_str = param.to_str();
    OCTensorList inter_tensors;
    StringList inter_nodes;
    OCTensorList result_tensors;
    StringList result_nodes;
    OCTensor checkbox_tensor;
    bool has_ocr_result = false;
    bool has_checkbox = false;
    bool has_seal = false;

    OCTensor param_tensor = OCTensor(param_str, true);
    OCTensor img_tensor =
        (imgs.size() > 0 ? OCTensor(imgs[0])
                        : OCTensor({0, 1, 3}, TRITONOPENCV_TYPE_UINT8));

    // Phrase 1. Get OCR Results
    do {
      if (enable_ocr_res) {
        int64_t size = ocr_results.size();
        OCTensor ocr_res_ot(ocr_results, {size});
        result_tensors.emplace_back(std::move(ocr_res_ot));
        result_nodes.emplace_back("ocr_result");
        has_ocr_result = true;
        break;
      }

      OCTensorList prep_outputs;
      if (general_prep){
        StringList input_names = {"bin_images", "params"};
        StringList output_names = {"prep_image", "prep_params"};
        OCTensorList prep_inputs = {OCTensor(data_vec[0]), param_tensor};
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            general_prep_model_name, context, input_names, output_names, prep_inputs,
            &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* r = future.get();
        err = ParseTensorsFromServerResponse(r, output_names, &prep_outputs);
        GraphInferResponseDelete(r);
        CHECK_ERROR_WITH_BREAK(err);
        if (!det_name.empty() || !recog_name.empty() || uie_name.length() > 0){
          cv::Mat img = prep_outputs[0].GetMat();
          int h = img.size[0];
          int w = img.size[1];
          cv::Mat img0;
          if(img.dims == 2){
            cv::Mat img_color;
            img0 = cv::Mat(h, w, CV_8UC1, img.data);
            cvtColor(img, img_color, 8);
            img_tensor = OCTensor(img_color);
          }else{
            img0 = cv::Mat(h, w, CV_8UC3, img.data);
            img_tensor = OCTensor(img0);
          }
        }else{
          result_tensors.emplace_back(prep_outputs[0]);
          result_nodes.emplace_back("general_prep_image");
          result_tensors.emplace_back(prep_outputs[1]);
          result_nodes.emplace_back("general_prep_params");
          break;
        }
      }

      OCTensorList det_inputs = {img_tensor, param_tensor};
      // Do checkbox
      std::future<TRITONSERVER_InferenceResponse*> checkbox_future;
      if (!checkbox_name.empty()) {
        StringList input_names = {"image", "params"};
        StringList output_names = {"boxes"};
        err = GraphExecuate(
            checkbox_name, context, input_names, output_names, det_inputs,
            &checkbox_future);
        has_checkbox = true;
        CHECK_ERROR_WITH_BREAK(err);
      }

      // Do Seal Recog
      std::future<TRITONSERVER_InferenceResponse*> seal_future;
      if (!seal_name.empty()) {
        StringList input_names = {"image", "params"};
        StringList output_names = {"seal_result"};
        err = GraphExecuate(
            seal_name, context, input_names, output_names, det_inputs,
            &seal_future);
        has_seal = true;
        CHECK_ERROR_WITH_BREAK(err);
      }

      // Do Det
      if (det_name.length() > 0) {
        OCTensorList inputs = {img_tensor, param_tensor};
        StringList input_names = {"image", "params"};
        StringList output_names = {"boxes", "boxes_score", "src_scale"};
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            det_name, context, input_names, output_names, inputs, &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &inter_tensors);
        GraphInferResponseDelete(resp);
        inter_nodes.insert(
            inter_nodes.end(), output_names.begin(), output_names.end());
        CHECK_ERROR_WITH_BREAK(err);

        if (!checkbox_name.empty()) {
          // Split bboxes by checkbox
          auto* r1 = checkbox_future.get();
          OCTensorList tensors;
          err = ParseTensorsFromServerResponse(r1, {"boxes"}, &tensors);
          GraphInferResponseDelete(r1);
          CHECK_ERROR_WITH_BREAK(err);
          checkbox_tensor = tensors[0];

          // Split text bboxes by checkboxes
          OCTensorList post_tensors;
          std::future<TRITONSERVER_InferenceResponse*> future;
          OCTensorList inputs = {
              inter_tensors[0], checkbox_tensor, inter_tensors[1], param_tensor};
          StringList input_names = {
              "boxes", "checkbox_boxes", "scores", "params"};
          StringList output_names = {"text_boxes", "text_scores"};
          err = GraphExecuate(
              checkbox_post_model_name, context, input_names, output_names,
              inputs, &future);
          CHECK_ERROR_WITH_BREAK(err);
          auto* r2 = future.get();
          err = ParseTensorsFromServerResponse(r2, output_names, &post_tensors);
          GraphInferResponseDelete(r2);
          CHECK_ERROR_WITH_BREAK(err);
          inter_tensors[0] = post_tensors[0];
          inter_tensors[1] = post_tensors[1];
        }
      }

      // Do recog
      if (auto_recog_names.size() > 0) {
        std::string auto_alg_name = "auto_hand_print";
        StringList input_names = {"image", "boxes", "params"};
        if (det_name.length() == 0) {
          input_names.push_back("patchs");
        }
        StringList output_names = {"texts", "texts_score"};
        auto bboxes_tensor = inter_tensors[0];
        OCTensorList inputs = {img_tensor, bboxes_tensor, param_tensor};
        if (det_name.length() == 0) {
          int64_t patchs_n = data_vec.size();
          OCTensor patchs_tensor(data_vec, {patchs_n});
          inputs.push_back(patchs_tensor);
        }
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            auto_alg_name, context, input_names, output_names, inputs, &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &inter_tensors);
        GraphInferResponseDelete(resp);
        inter_nodes.insert(
            inter_nodes.end(), output_names.begin(), output_names.end());
        CHECK_ERROR_WITH_BREAK(err);
      } else {
        if (recog_name.length() > 0 && det_name.length() > 0) {
          StringList input_names = {"image", "boxes", "params"};
          StringList output_names = {"texts", "texts_score"};
          auto bboxes_tensor = inter_tensors[0];
          OCTensorList inputs = {img_tensor, bboxes_tensor, param_tensor};
          std::future<TRITONSERVER_InferenceResponse*> future;
          err = GraphExecuate(
              recog_name, context, input_names, output_names, inputs, &future);
          CHECK_ERROR_WITH_BREAK(err);
          auto* resp = future.get();
          err =
              ParseTensorsFromServerResponse(resp, output_names, &inter_tensors);
          GraphInferResponseDelete(resp);
          inter_nodes.insert(
              inter_nodes.end(), output_names.begin(), output_names.end());
          CHECK_ERROR_WITH_BREAK(err);
        }

        // Do Recog only
        if (recog_only) {
          StringList input_names = {"image", "boxes", "params", "patchs"};
          StringList output_names = {"texts", "texts_score"};
          OCTensor dummy_image({0, 1, 3}, TRITONOPENCV_TYPE_UINT8);
          OCTensor dummy_boxes({0, 4, 2}, TRITONOPENCV_TYPE_FP32);
          int64_t patchs_n = data_vec.size();
          OCTensor patchs_tensor(data_vec, {patchs_n});
          OCTensorList inputs = {
              dummy_image, dummy_boxes, param_tensor, patchs_tensor};

          std::future<TRITONSERVER_InferenceResponse*> future;
          err = GraphExecuate(
              recog_name, context, input_names, output_names, inputs, &future);
          CHECK_ERROR_WITH_BREAK(err);
          auto* resp = future.get();
          err =
              ParseTensorsFromServerResponse(resp, output_names, &inter_tensors);
          GraphInferResponseDelete(resp);
          inter_nodes.insert(
              inter_nodes.end(), output_names.begin(), output_names.end());
          CHECK_ERROR_WITH_BREAK(err);
        }
      }

      // Merge checkbox with normal ocr result
      if (has_checkbox) {
        // -1, 4, 2
        auto m1 = inter_tensors[0].GetMatrix();
        auto m2 = checkbox_tensor.GetMatrix();
        cv::Mat m3;
        cv::vconcat(m1, m2, m3);
        int n = m3.rows;
        inter_tensors[0] = OCTensor(std::move(m3));
        inter_tensors[0].set_shape({n, 4, 2});
        auto text_tensors = inter_tensors[3];
        std::vector<absl::string_view> texts0;
        std::vector<std::string> texts;
        text_tensors.GetStrings(texts0);
        for (size_t i = 0; i < texts0.size(); i++) {
          texts.emplace_back(texts0[i].data(), texts0[i].length());
        }

        unsigned char bytes[4] = {0xf0, 0x9f, 0x99, 0x8c};
        std::string str(reinterpret_cast<char*>(bytes), sizeof(bytes));
        std::string text_checkbox = str + checkbox_name;
        for (int i = 0; i < m2.rows; i++) {
          texts.emplace_back(text_checkbox);
        }

        inter_tensors[3] = std::move(OCTensor(texts, {n}));

        std::vector<int> s0 = {m2.size[0]};
        cv::Mat m4 = inter_tensors[1].GetMatrix();
        cv::Mat m5 = cv::Mat(s0, CV_32FC1, cv::Scalar(-1.0f));
        cv::Mat m6;
        cv::vconcat(m4, m5, m6);
        inter_tensors[1] = OCTensor(std::move(m6));
        inter_tensors[1].set_shape({n});

        cv::Mat m7 = inter_tensors[4].GetMatrix();
        cv::Mat m8;
        cv::vconcat(m7, m5, m8);
        inter_tensors[4] = OCTensor(std::move(m8));
        inter_tensors[4].set_shape({n});
      }

      if (has_seal) {
        auto* r1 = seal_future.get();
        OCTensorList tensors;
        err = ParseTensorsFromServerResponse(r1, {"seal_result"}, &tensors);
        GraphInferResponseDelete(r1);
        CHECK_ERROR_WITH_BREAK(err);

        // todo: merge other elem
        inter_tensors.emplace_back(tensors[0]);
        inter_nodes.emplace_back("other_elems");
        result_tensors.emplace_back(tensors[0]);
        result_nodes.emplace_back("seal_result");
      }

      // Do OCRFormat
      if (enable_format && inter_tensors.size() >= 5) {
        StringList input_names = inter_nodes;
        input_names.emplace_back("params");
        OCTensorList inputs = inter_tensors;
        inputs.emplace_back(param_tensor);
        StringList output_names = {"RESULTS"};

        std::vector<absl::string_view> texts0;
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            format_model_name, context, input_names, output_names, inputs,
            &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &result_tensors);
        result_nodes.emplace_back("ocr_result");
        GraphInferResponseDelete(resp);
        CHECK_ERROR_WITH_BREAK(err);
        has_ocr_result = true;
      }
    } while (0);

    if (err != nullptr) {
      *resp = WriteErrorResponse(err, &request_info);
      return nullptr;
    }

    // Phrase 2. Get Information Extraction Results
    do {
      if (has_ocr_result && uie_name.length() > 0) {
        StringList input_names = {"image", "ocr_results", "ellm_schema"};
        StringList output_names = {"ellm_result"};

        OCTensor schema_tensor =
            std::move(OCTensor({absl::string_view(uie_schema)}, {1}));
        int idx_ocr_res = result_tensors.size() - 1;
        OCTensorList inputs = {
            img_tensor, result_tensors[idx_ocr_res], schema_tensor};
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            uie_name, context, input_names, output_names, inputs, &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &result_tensors);
        result_nodes.emplace_back("ellm_result");
        GraphInferResponseDelete(resp);
        CHECK_ERROR_WITH_BREAK(err);
      }

      // Do Table (table depend ocr result)
      if (table_name.length() > 0 && has_ocr_result) {
        StringList input_names = {"image", "ocr_result", "params"};
        int idx_ocr_res = result_tensors.size() - 1;
        OCTensorList inputs = {
            img_tensor, result_tensors[idx_ocr_res], param_tensor};
        StringList output_names = {"table_result"};
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            table_name, context, input_names, output_names, inputs, &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &result_tensors);
        result_nodes.emplace_back("table_result");
        GraphInferResponseDelete(resp);
        CHECK_ERROR_WITH_BREAK(err);
      }

      // Do Cls (cls depend ocr result)
      if (cls_name.length() > 0 && has_ocr_result) {
        StringList input_names = {"ocr_result", "params"};
        int idx_ocr_res = result_tensors.size() - 1;
        OCTensorList inputs = {result_tensors[idx_ocr_res], param_tensor};
        StringList output_names = {"cls_result"};
        std::future<TRITONSERVER_InferenceResponse*> future;
        err = GraphExecuate(
            cls_name, context, input_names, output_names, inputs, &future);
        CHECK_ERROR_WITH_BREAK(err);
        auto* resp = future.get();
        err = ParseTensorsFromServerResponse(resp, output_names, &result_tensors);
        result_nodes.emplace_back("cls_result");
        GraphInferResponseDelete(resp);
        CHECK_ERROR_WITH_BREAK(err);
      }

    } while (0);

    if (err != nullptr) {
      *resp = WriteErrorResponse(err, &request_info);
      return nullptr;
    }

    request_info.elapse = timer.toc();
    if (result_tensors.size() == 0) {
      // mode: Det only
      rapidjson::StringBuffer buffer;
      WriteOKResponse(&buffer, inter_tensors, inter_nodes, &request_info);
      *resp = buffer.GetString();
    } else if (result_tensors.size() >= 1) {
      // mode: Det + Recog + Table
      rapidjson::StringBuffer buffer;
      for (size_t i = 0; i < result_tensors.size(); i++) {
        result_tensors[i].set_jsonstr();
      }
      WriteOKResponse(&buffer, result_tensors, result_nodes, &request_info);
      *resp = buffer.GetString();
    }
  }
  catch (const std::runtime_error& re) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, re.what());
  }
  catch(const std::exception& ex){
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, ex.what());
  }
  catch(...){
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "unknow error!");
  }

  return nullptr;
}


}}  // namespace dataelem::alg
