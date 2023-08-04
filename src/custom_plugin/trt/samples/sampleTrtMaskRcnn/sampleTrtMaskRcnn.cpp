/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _MSC_VER
#include <unistd.h>
#include <sys/time.h>
#endif

#include <assert.h>
#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>
#include "NvInfer.h"
#include "fileops.hpp"
#include "maskrcnn_model.h"
#include "model.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

using namespace sample;

const std::string gSampleName = "TensorRT.sample_trt_maskrcnn";

class SampleTrtMaskRCNN: public SampleModel {

 public:
  SampleTrtMaskRCNN(SampleTrtMaskRCNNParams& params)
    : SampleModel(params), mParams(params) {
  }

  bool predict();

  bool constructNetwork();

 private:

  SampleTrtMaskRCNNParams mParams;

  bool processInput(const string& filename,
                    std::vector<int>& imgSzie,
                    std::vector<float>& im) const;

  bool saveOutput(std::map<std::string, std::pair<void*, nvinfer1::Dims>>&, const string& filename);
};


bool SampleTrtMaskRCNN::constructNetwork() {
  std::vector<ITensor*> inputs = _context->setInputNode(mParams.inputTensorNames,
  {Dims4{1, mParams.inputH, mParams.inputW, 3}},
  {DataType::kFLOAT});
  std::vector<ITensor*> outputs;
  buildMaskRcnn(_context, inputs, mParams, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  _context->setBatchSize(mParams.batchSize);
  _context->setWorkspaceSize(6_GiB);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }
  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

bool SampleTrtMaskRCNN::predict() {
  std::string txt_path = mParams.dataDirs[0] + "/image_file.txt";
  std::vector<std::string> files;
  fileops::files_in_txt(txt_path, files);
  gLogInfo << "image num:" << files.size() << std::endl;

  auto totalHost = 0;
  bool start_cnt = false;
  int cnt = 0;
  for (int i = 0; i < files.size(); ++i) {
    std::string name = files[i].substr(0, files[i].length() - 1);
    std::vector<int> imgSzie;
    std::vector<float> im;
    if (!processInput(name, imgSzie, im)) {
      return false;
    }

    auto tStart = std::chrono::high_resolution_clock::now();

    std::map<std::string, std::pair<void*, nvinfer1::Dims>> outputs;
    std::vector<void*> inputs = {im.data()};
    std::vector<Dims> dims = {Dims4(1, imgSzie[0], imgSzie[1], imgSzie[2])};
    int batch_size = dims[0].d[0];
    auto status = infer(batch_size, inputs, dims, outputs);

    auto tEnd = std::chrono::high_resolution_clock::now();
    totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();

    cnt++;
    if(cnt == 100 && !start_cnt) {
      start_cnt = true;
      cnt = 0;
      totalHost = 0;
    }
    if(start_cnt && cnt > 0 && cnt % 50 == 0)
      gLogInfo << "t:" <<totalHost << " cnt:" <<cnt << " t/per im:" <<1.0*totalHost/cnt << std::endl;

    if (!status) {
      return false;
    }

    saveOutput(outputs, name);
  }

  gLogInfo << "t:" <<totalHost << " cnt:" << cnt << " t/per im:" << 1.0*totalHost/cnt << std::endl;

  return true;
}


bool SampleTrtMaskRCNN::saveOutput(std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs, const string& filename) {

  void* detection = outputs[mParams.outputTensorNames[0]].first;
  float* res_det = (float*) detection;
  Dims det_dims = outputs[mParams.outputTensorNames[0]].second;

  void* mask = outputs[mParams.outputTensorNames[1]].first;
  float* res_mask = (float*) mask;
  Dims mask_dims = outputs[mParams.outputTensorNames[1]].second;

  // show<float>(res_det, det_dims, mParams.outputTensorNames[0]);
  // show<float>(res_mask, mask_dims, mParams.outputTensorNames[1]);
  if (!fileops::dir_exists(mParams.dataDirs[1])) {
    fileops::create_dir(mParams.dataDirs[1], 0755);
  }
  write_bin<float>(res_det, det_dims, mParams.dataDirs[1] + "/res_det", filename);
  write_bin<float>(res_mask, mask_dims, mParams.dataDirs[1] + "/res_mask", filename);

  bool pass = true;
  return pass;
}

bool SampleTrtMaskRCNN::processInput(const std::string& filename,
                                     std::vector<int>& imgSzie,
                                     std::vector<float>& im) const {
  // Available image
  std::string shape_name = mParams.dataDirs[0] + "/shape/" + filename;
  std::string im_name = mParams.dataDirs[0] + "/bin/" + filename;
  std::ifstream f_shape(shape_name, std::ios::binary);
  assert(f_shape.is_open() && "Unable to load image shape.");
  std::vector<float> shape(5);
  f_shape.read((char*)shape.data(), sizeof(float) * 5);

  int resized_h = int(shape[0]);
  int resized_w = int(shape[1]);
  int origin_h = int(shape[2]);
  int origin_w = int(shape[3]);
  float scale = shape[4];
  int c = 3;
  int size = c * resized_h * resized_w;
  imgSzie.push_back(resized_h);
  imgSzie.push_back(resized_w);
  imgSzie.push_back(c);
  std::ifstream f_im(im_name, std::ios::binary);
  assert(f_im.is_open() && "Unable to load image.");
  im.resize(size);
  f_im.read((char*)im.data(), sizeof(float) * size);
  return true;
}

SampleTrtMaskRCNNParams initializeSampleParams(const samplesCommon::Args& args) {
  SampleTrtMaskRCNNParams params;

  params.inputTensorNames.push_back("image");
  params.outputTensorNames.push_back("output_detections");
  params.outputTensorNames.push_back("output_masks");
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;
  params.batchSize = 1;
  params.weightsFile = "../test_data/models/ocr_maskrcnn_weights.wts";
  params.engineFile = "../test_data/models/maskrcnn_v5_1600_fp16_3090.trt";

  if (args.dataDirs.empty()) {
    // input image
    params.dataDirs.push_back("../test_data/ocr_maskrcnn_data/all_kinds_train_images_angle_val_fix_shape_hwc_1600");
    // result save path
    params.dataDirs.push_back("../test_data/ocr_maskrcnn_data/all_kinds_train_images_angle_val_fix_shape_hwc_1600/res");
  } else {
    params.dataDirs = args.dataDirs;
  }

  params.PRENMS_TOPK = 6000;
  params.KEEP_TOPK = 1000;
  params.RESULTS_PER_IM = 300;
  params.PROPOSAL_NMS_THRESH = 0.7;
  params.RESULT_SCORE_THRESH = 0.3;
  params.FRCNN_NMS_THRESH = 0.8;
  params.MAX_SIDE = 1600;
  params.inputH = 1600;
  params.inputW = 1600;
  params.channel_axis = 1;

  return params;
}

void printHelpInfo() {
  std::cout << "Usage: ./sample_maskRCNN [-h or --help] [-d or --datadir=<path to data directory>]" << std::endl;
  std::cout << "--help          Display help information" << std::endl;
  std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
            "multiple times to add multiple directories. If no data directories are given, the default is to use "
            "data/samples/maskrcnn/ and data/maskrcnn/"
            << std::endl;
  std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
  std::cout << "--batch         Specify inference batch size." << std::endl;
}

int main(int argc, char** argv) {
  // todo CUDA_VISIBLE_DEVICES=0 和 cudaSetDevice(device_id) 运行速度不一致
  std::string device_id_str = argv[1];
  int device_id = std::stoi(device_id_str);
  std::cout<< "device_id:" << device_id << std::endl;
  samplesCommon::Args args;
  bool argsOK = samplesCommon::parseArgs(args, argc, argv);
  if (!argsOK) {
    gLogError << "Invalid arguments" << std::endl;
    printHelpInfo();
    return EXIT_FAILURE;
  }
  if (args.help) {
    printHelpInfo();
    return EXIT_SUCCESS;
  }

  auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

  gLogger.reportTestStart(sampleTest);

  SampleTrtMaskRCNNParams params = initializeSampleParams(args);
  SampleTrtMaskRCNN sample(params);

  gLogInfo << "Building and running a GPU inference engine for Mask RCNN" << std::endl;

  cudaSetDevice(device_id);

  if (!sample.initIContext()) {
    return gLogger.reportFail(sampleTest);
  }

  if (!fileops::file_exists(params.engineFile)) {
    if (!sample.build()) {
      return gLogger.reportFail(sampleTest);
    }

    if (!sample.saveEngine()) {
      return gLogger.reportFail(sampleTest);
    }
  } else {
    if (!sample.loadModel()) {
      return gLogger.reportFail(sampleTest);
    }
  }
  if (!sample.initBuffer()) {
    return gLogger.reportFail(sampleTest);
  }
  if (!sample.predict()) {
    return gLogger.reportFail(sampleTest);
  }
  return gLogger.reportPass(sampleTest);
}

