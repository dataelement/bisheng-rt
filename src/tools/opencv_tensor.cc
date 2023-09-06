//Loading Opencv fIles for processing

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include <fstream>

DEFINE_bool(run_demo, false, "run demo");
DEFINE_bool(show_cvinfo, false, "show cv info");

void showGraphInfo(tensorflow::GraphDef & graph_def) {
  int i;
  for (i = 0; i < graph_def.node_size(); i++) {
    graph_def.node(i).PrintDebugString();
  }
}

tensorflow::Session* load_graph(const tensorflow::SessionOptions& opt,
                                int device_id, std::string graph_pb) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status status = ReadBinaryProto(
                                tensorflow::Env::Default(), graph_pb, &graph_def);
  if (!status.ok()) {
    std::cout << "read pb failed\n";
    return nullptr;
  }
  // modify node device information
  std::string device("/gpu:" + std::to_string(device_id));
  for (int i = 0; i < graph_def.node_size(); ++i) {
    graph_def.mutable_node(i)->set_device(device);
  }
  auto session = (tensorflow::NewSession(opt));
  //showGraphInfo(graph_def);
  session->Create(graph_def);
  return session;
}


void show_cv_info() {
#ifdef HAVE_TBB
  std::cout << "Running with TBB" << "\n";
#else
#ifdef _OPENMP
  std::cout << "Running with OpenMP" << "\n";
#else
  std::cout << "Running without OpenMP and without TBB" << "\n";
#endif
#endif

#ifdef HAVE_IPP
  std::cout << "Running with IPP" << "\n";
#else
  std::cout << "Running without IPP" << "\n";
#endif

  std::cout << cv::getBuildInformation() << std::endl;
}


void run_demo() {
  std::string path = "../test_data/regress_data/cat.jpg";
  std::string InputName = "InputImage";
  std::string OutputName = "InceptionV1/Logits/Predictions/Reshape_1";
  std::string graphFile = "../test_data/models/inception_v1.pb";
  int height = 224;
  int width = 224;
  int mean = 128;
  int std = 128;

  // create input tensor
  cv::Size s(height, width);
  cv::Mat readImage = cv::imread(path);
  cv::Mat Image;
  cv::resize(readImage, Image, s, 0, 0, cv::INTER_CUBIC);
  int depth = Image.channels();
  // creating a Tensor for storing the data
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({1, height, width, depth}));
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();
  cv::Mat Image2;
  Image.convertTo(Image2, CV_32FC1);
  Image = Image2;
  Image = Image - mean;
  Image = Image / std;
  const float * source_data = (float*) Image.data;
  for (int y = 0; y < height; ++y) {
    const float* source_row = source_data + (y * width * depth);
    for (int x = 0; x < width; ++x) {
      const float* source_pixel = source_row + (x * depth);
      for (int c = 0; c < depth; ++c) {
        const float* source_value = source_pixel + c;
        input_tensor_mapped(0, y, x, c) = *source_value;
      }
    }
  }

  // create session
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto &config = options.config;
  // TODO: RUN THE OPTIMIZE ON THE CUDA10
  config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(
    tensorflow::OptimizerOptions::ON_1);
  config.mutable_gpu_options()->set_visible_device_list("0");
  config.mutable_gpu_options()->set_allow_growth(false);

  // config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.8);
  // auto vd1 = config.mutable_gpu_options()->mutable_experimental()->add_virtual_devices();
  // vd1->add_memory_limit_mb(3000.0);
  // vd1->add_memory_limit_mb(6000.0);
  // auto vd2 = config.mutable_gpu_options()->mutable_experimental()->add_virtual_devices();
  // vd2->add_memory_limit_mb(2000.0);
  // vd2->add_memory_limit_mb(5000.0);

  std::unique_ptr<tensorflow::Session> session(load_graph(options, 0, graphFile));

  // show device
  std::vector<tensorflow::DeviceAttributes> dev_attr_vec;
  session->ListDevices(&dev_attr_vec);
  int j = 0;
  std::cout << "session bind device infos.\n";
  for (auto dev : dev_attr_vec) {
    std::cout << "device index:" << j << "\n";
    std::cout << dev.physical_device_desc() << ","
              << dev.name() << "," << dev.device_type() << ","
              << dev.memory_limit() << "\n";
    j++;
  }

  // run graph
  std::vector<tensorflow::Tensor> finalOutput;
  auto run_status = session->Run(
  {{InputName, input_tensor}}, {OutputName}, {}, &finalOutput);
  tensorflow::Tensor output = std::move(finalOutput.at(0));
  auto scores = output.flat<float>();

  std::cout << "scores[:10]=" << scores(0);
  for (int i = 1; i < 10; i++) {
    std::cout << "," << scores(i);
  }
  std::cout << "\n";

}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_show_cvinfo) {
    show_cv_info();
  }
  if (FLAGS_run_demo) {
    run_demo();
  }
  return 0;
}
