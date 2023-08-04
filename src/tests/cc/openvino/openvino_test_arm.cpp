#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include "openvino/openvino.hpp"

void openvino_infer_det(cv::Mat img, std::string device_name="CPU"){
  //std::string model_path = "/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_det_infer/model.xml";
  //std::string model_path = "/root/workspace/models/det_r34_db_1_opset16.xml";
  std::string model_path = "/root/workspace/model_repo/ocr_lite_openvino/graphs/det_db_graph/1/model.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);
  std::cout<<"core.read_model success"<<std::endl;
  //ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
  //std::cout<<"PrePostProcessor success"<<std::endl;
  //ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::num_streams(ov::streams::NUMA), ov::inference_num_threads(200));
  ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::num_streams(32), ov::inference_num_threads(200), ov::affinity(ov::Affinity::NUMA), ov::enable_profiling(false));
  auto nthreads = compiled_model.get_property(ov::inference_num_threads);
  std::cout<<"core.compile_model success, nthreads:"<<nthreads<<std::endl;
  ov::InferRequest infer_request = compiled_model.create_infer_request();
  std::cout<<"compiled_model.create_infer_request success"<<std::endl;

  int l = int(1*3*960*960);
  float s0 = 0.0f;
  float* d = (float*)img.data;
  for(int i=0; i<l; i++){
    s0 += d[i];
  }
  std::cout<<"input_tensor sum:"<<s0<<std::endl;
  ov::Tensor input_tensor(ov::element::f32, {1, 3, 960, 960}, (float*)img.data);
  //infer_request.set_input_tensor(input_tensor);
  infer_request.set_tensor("x", input_tensor);
  std::cout<<"start infer"<<std::endl;
  auto start = std::chrono::system_clock::now();
  infer_request.infer();
  auto end = std::chrono::system_clock::now();
  auto duration = end - start;
  std::chrono::duration<double> diff = duration;
  std::cout<<"end infer:"<<diff.count()<<" s"<<std::endl;
  ov::Tensor output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  //ov::Shape output_shape = {1, 1, 960, 960};
  const float* featmap = output_tensor.data<const float>();
  std::cout<<"out shape:"<<output_shape[0]<<" "<<output_shape[1]<<" "<<output_shape[2]<<" "<<output_shape[3]<<std::endl;
  int size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
  float s = 0.0f;
  for (int i = 0; i < size; i++) {
    s += featmap[i];
  }
  std::cout<<"sum:"<<s<<std::endl;
  std::cout<<"save result"<<std::endl;
  std::vector<int> feat_shape = {1, 1, 960, 960};
  cv::Mat feat_map = cv::Mat(feat_shape, CV_32F, (uint8_t*)featmap);
  cv::FileStorage file("det_feat.ext", cv::FileStorage::WRITE);
  file << "feat_map" << feat_map;
  file.release();
}

void openvino_infer_cls(cv::Mat img, std::string device_name="CPU"){
  //std::string model_path = "/home/liuqingjie/models/openvino_2022_2/ch_ppocr_mobile_v2.0_cls_infer/model.xml";
  std::string model_path = "/root/workspace/model_repo/ocr_lite_openvino/graphs/cls_angle_graph/1/model.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);

  ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::num_streams(ov::streams::NUMA), ov::inference_num_threads(200));
  ov::InferRequest infer_request = compiled_model.create_infer_request();

  ov::Tensor input_tensor(ov::element::f32, {32, 3, 48, 192}, (float*)img.data);
  infer_request.set_tensor("x", input_tensor);
  std::cout<<"start infer"<<std::endl;
  auto start = std::chrono::system_clock::now();
  infer_request.infer();
  auto end = std::chrono::system_clock::now();
  auto duration = end - start;
  std::chrono::duration<double> diff = duration;
  std::cout<<"end infer:"<<diff.count()<<" s"<<std::endl;
  std::string name = "save_infer_model/scale_0.tmp_1";
  ov::Tensor output_tensor = infer_request.get_tensor(name);
  //ov::Tensor output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  const float* featmap = output_tensor.data<const float>();
  std::cout<<"out shape:"<<output_shape[0]<<" "<<output_shape[1]<<std::endl;
  int size = output_shape[0] * output_shape[1];
  std::cout<<"featmap:";
  for (int i = 0; i < size; i++) {
    std::cout<<featmap[i]<<" ";
  }
  std::cout<<std::endl;
}

void openvino_infer_rec(cv::Mat img, std::string device_name="CPU"){
  //std::string model_path = "/home/liuqingjie/models/openvino_2022_2/ch_PP-OCRv3_rec_infer_fp16_matrix/model.xml";
  //std::string model_path = "/root/workspace/model_repo/ocr_lite_openvino/graphs/rec_ch_graph/1/model.xml";
  std::string model_path = "/root/workspace/models/rec_res34_bilstm_opset7.xml";
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);

  ov::CompiledModel compiled_model = core.compile_model(model, device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), ov::num_streams(ov::streams::NUMA), ov::inference_num_threads(200));
  ov::InferRequest infer_request = compiled_model.create_infer_request();

  ov::Tensor input_tensor(ov::element::f32, {6, 3, 32, 600}, (float*)img.data);
  infer_request.set_tensor("x", input_tensor);
  std::cout<<"start infer"<<std::endl;
  auto start = std::chrono::system_clock::now();
  infer_request.infer();
  auto end = std::chrono::system_clock::now();
  auto duration = end - start;
  std::chrono::duration<double> diff = duration;
  std::cout<<"end infer:"<<diff.count()<<" s"<<std::endl;
  std::string name = "softmax_0.tmp_0";
  ov::Tensor output_tensor = infer_request.get_tensor(name);
  ov::Shape output_shape = output_tensor.get_shape();
  const float* featmap = output_tensor.data<const float>();
  std::cout<<"out shape:"<<output_shape[0]<<" "<<output_shape[1]<<" "<<output_shape[2]<<std::endl;
  int size = output_shape[0] * output_shape[1] * output_shape[2];
  std::cout<<"sum:";
  float s = 0.0f;
  for (int i = 0; i < size; i++) {
    s += featmap[i];
  }
  std::cout<<s<<std::endl;

  int step0 = output_shape[1] * output_shape[2];
  int step1 = output_shape[2];
  std::cout<<"out:";
  for(int i=0; i<output_shape[0]; i++){
    const float* outmap = featmap + i * step0;
    for(int j=0; j<output_shape[1]; j++){
      const float* data = outmap + j * step1;
      double min_val, max_val;
      cv::Point2i min_loc;
      cv::Point2i max_loc;
      cv::Mat m = cv::Mat(1, (int)output_shape[2], CV_32FC1, (float*)data);
      cv::minMaxLoc(m, &min_val, &max_val, &min_loc, &max_loc);
      int loc = max_loc.x;
      std::cout<<loc<<" "<<max_val<<";";
    }
    std::cout<<std::endl;
  }
}

void det(){
  cv::FileStorage fs("det_batched_imgs.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["batched_imgs"] >> m;
  fs.release();
  openvino_infer_det(m);
}

void cls(){
  cv::FileStorage fs("cls_batched_imgs.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["batched_imgs"] >> m;
  fs.release();
  openvino_infer_cls(m);
}

void rec(){
  cv::FileStorage fs("rec_batched_imgs.ext", cv::FileStorage::READ);
  cv::Mat m;
  fs["batched_imgs"] >> m;
  fs.release();
  openvino_infer_rec(m);
}

int main(int argc, char** argv){
  det();
  //cls();
  //rec();
  return 0;
}