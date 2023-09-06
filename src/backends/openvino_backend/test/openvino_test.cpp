#include "openvino/openvino.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void
openvino_infer_det(cv::Mat img, std::string device_name = "CPU")
{
  std::string model_path =
      "/home/public/models/openvino_2022_2/ch_PP-OCRv3_det_infer/model.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);

  // const ov::Layout tensor_layout{"NCHW"};
  ov::preprocess::PrePostProcessor ppp =
      ov::preprocess::PrePostProcessor(model);
  // ov::preprocess::InputInfo& input_info = ppp.input();
  // input_info.tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
  // ppp.input().preprocess().convert_element_type(ov::element::f32).convert_layout("NCHW");
  // ppp.input().preprocess();
  // input_info.model().set_layout("NCHW");
  // ov::preprocess::OutputInfo& output_info = ppp.output();
  // output_info.tensor().set_element_type(ov::element::f32);
  // model = ppp.build();
  ov::CompiledModel compiled_model = core.compile_model(model, device_name);
  ov::InferRequest infer_request = compiled_model.create_infer_request();

  int l = int(1 * 3 * 960 * 608);
  float s0 = 0.0f;
  float* d = (float*)img.data;
  for (int i = 0; i < l; i++) {
    s0 += d[i];
  }
  std::cout << "input_tensor sum:" << s0 << std::endl;
  ov::Tensor input_tensor(ov::element::f32, {1, 3, 960, 608}, (float*)img.data);
  // infer_request.set_input_tensor(input_tensor);
  infer_request.set_tensor("x", input_tensor);

  infer_request.infer();
  ov::Tensor output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  // ov::Shape output_shape = {1, 1, 960, 608};
  const float* featmap = output_tensor.data<const float>();
  std::cout << "out shape:" << output_shape[0] << " " << output_shape[1] << " "
            << output_shape[2] << " " << output_shape[3] << std::endl;
  int size =
      output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
  float s = 0.0f;
  for (int i = 0; i < size; i++) {
    s += featmap[i];
  }
  std::cout << "sum:" << s << std::endl;
  std::cout << "save result" << std::endl;
  std::vector<int> feat_shape = {1, 1, 960, 608};
  cv::Mat feat_map = cv::Mat(feat_shape, CV_32F, (uint8_t*)featmap);
  cv::FileStorage file("det_feat.ext", cv::FileStorage::WRITE);
  file << "feat_map" << feat_map;
  file.release();
}

void
openvino_infer_cls(cv::Mat img, std::string device_name = "CPU")
{
  std::string model_path =
      "/home/public/models/openvino_2022_2/ch_ppocr_mobile_v2.0_cls_infer/"
      "model.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);

  ov::CompiledModel compiled_model = core.compile_model(model, device_name);
  ov::InferRequest infer_request = compiled_model.create_infer_request();

  ov::Tensor input_tensor(ov::element::f32, {2, 3, 48, 192}, (float*)img.data);
  infer_request.set_tensor("x", input_tensor);

  infer_request.infer();
  std::string name = "save_infer_model/scale_0.tmp_1";
  ov::Tensor output_tensor = infer_request.get_tensor(name);
  // ov::Tensor output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  const float* featmap = output_tensor.data<const float>();
  std::cout << "out shape:" << output_shape[0] << " " << output_shape[1]
            << std::endl;
  int size = output_shape[0] * output_shape[1];
  std::cout << "featmap:";
  for (int i = 0; i < size; i++) {
    std::cout << featmap[i] << " ";
  }
  std::cout << std::endl;
}

void
openvino_infer_rec(cv::Mat img, std::string device_name = "CPU")
{
  std::string model_path =
      "/home/public/models/openvino_2022_2/"
      "ch_PP-OCRv3_rec_infer_fp16_matrix/model.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);

  ov::CompiledModel compiled_model = core.compile_model(model, device_name);
  ov::InferRequest infer_request = compiled_model.create_infer_request();

  ov::Tensor input_tensor(ov::element::f32, {2, 3, 48, 711}, (float*)img.data);
  infer_request.set_tensor("x", input_tensor);

  infer_request.infer();
  std::string name = "softmax_5.tmp_0";
  ov::Tensor output_tensor = infer_request.get_tensor(name);
  ov::Shape output_shape = output_tensor.get_shape();
  const float* featmap = output_tensor.data<const float>();
  std::cout << "out shape:" << output_shape[0] << " " << output_shape[1] << " "
            << output_shape[2] << std::endl;
  int size = output_shape[0] * output_shape[1] * output_shape[2];
  std::cout << "sum:";
  float s = 0.0f;
  for (int i = 0; i < size; i++) {
    s += featmap[i];
  }
  std::cout << s << std::endl;

  int step0 = output_shape[1] * output_shape[2];
  int step1 = output_shape[2];
  std::cout << "out:";
  for (int i = 0; i < output_shape[0]; i++) {
    const float* outmap = featmap + i * step0;
    for (int j = 0; j < output_shape[1]; j++) {
      const float* data = outmap + j * step1;
      double min_val, max_val;
      cv::Point2i min_loc;
      cv::Point2i max_loc;
      cv::Mat m = cv::Mat(1, (int)output_shape[2], CV_32FC1, (float*)data);
      cv::minMaxLoc(m, &min_val, &max_val, &min_loc, &max_loc);
      int loc = max_loc.x;
      std::cout << loc << " " << max_val << ";";
    }
    std::cout << std::endl;
  }
}

void
det()
{
  cv::FileStorage fs("det_im.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["img"] >> m;
  fs.release();
  openvino_infer_det(m);
}

void
cls()
{
  cv::FileStorage fs("cls_in_img.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["img"] >> m;
  fs.release();
  openvino_infer_cls(m);
}

void
rec()
{
  cv::FileStorage fs("rec_in_img.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["img"] >> m;
  fs.release();
  openvino_infer_rec(m);
}

int
main(int argc, char** argv)
{
  rec();
  return 0;
}