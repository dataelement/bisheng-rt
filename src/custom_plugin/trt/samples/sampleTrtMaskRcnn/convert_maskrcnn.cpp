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
#include <sys/time.h>
#include <unistd.h>
#endif

#include <assert.h>
#include <cuda_runtime_api.h>
#include <jsoncpp/json/json.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "fileops.hpp"
#include "maskrcnn_model.h"
#include "model.h"

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

// max
#include <algorithm>

using namespace sample;

// MaskRCNN Parameter

const std::string gSampleName = "TensorRT.sample_trt_maskrcnn";

namespace MaskRCNNUtils {

}  // namespace MaskRCNNUtils

class SampleTrtMaskRCNN : public SampleModel {
 public:
  SampleTrtMaskRCNN(SampleTrtMaskRCNNParams& params)
      : SampleModel(params), mParams(params)
  {
  }
  bool constructNetwork();

 private:
  SampleTrtMaskRCNNParams mParams;
};

bool
SampleTrtMaskRCNN::constructNetwork()
{
  std::vector<ITensor*> inputs = _context->setInputNode(
      mParams.inputTensorNames, {Dims4{1, mParams.inputH, mParams.inputW, 3}},
      {DataType::kFLOAT});
  std::vector<ITensor*> outputs;
  buildMaskRcnn(_context, inputs, mParams, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  _context->setBatchSize(mParams.batchSize);
  _context->setWorkspaceSize(mParams.workspaceSize);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }
  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

SampleTrtMaskRCNNParams
initializeSampleParams(const std::string& config)
{
  SampleTrtMaskRCNNParams params;
  Json::Reader reader;
  Json::Value root;

  ifstream in(config, ios::binary);
  if (!in.is_open()) {
    std::cout << "Error opening file " << config << "\n";
    return params;
  }
  if (reader.parse(in, root)) {
    for (unsigned int i = 0; i < root["input_tensors"].size(); i++) {
      std::string input_tensor = root["input_tensors"][i].asString();
      params.inputTensorNames.push_back(input_tensor);
    }
    for (unsigned int i = 0; i < root["output_tensors"].size(); i++) {
      std::string output_tensor = root["output_tensors"][i].asString();
      params.outputTensorNames.push_back(output_tensor);
    }
    params.PRENMS_TOPK = root["model_params"]["PRENMS_TOPK"].asInt();
    params.KEEP_TOPK = root["model_params"]["KEEP_TOPK"].asInt();
    params.RESULTS_PER_IM = root["model_params"]["RESULTS_PER_IM"].asInt();
    params.PROPOSAL_NMS_THRESH =
        root["model_params"]["PROPOSAL_NMS_THRESH"].asFloat();
    params.RESULT_SCORE_THRESH =
        root["model_params"]["RESULT_SCORE_THRESH"].asFloat();
    params.FRCNN_NMS_THRESH =
        root["model_params"]["FRCNN_NMS_THRESH"].asFloat();
    params.MAX_SIDE = root["model_params"]["EDGE_SIZE"].asInt();
    params.inputH = root["model_params"]["EDGE_SIZE"].asInt();
    params.inputW = root["model_params"]["EDGE_SIZE"].asInt();
    params.channel_axis = root["model_params"]["CHANNEL_AXIS"].asInt();

    params.fp16 = root["fp16"].asBool();
    params.workspaceSize = root["workspaceSize"].asInt64();
    params.batchSize = root["batchSize"].asInt();
    params.weightsFile = root["weightsFile"].asString();
    params.engineFile = root["engineFile"].asString();
  } else {
    std::cout << "parse error\n" << std::endl;
  }
  in.close();
  return params;
}

void
printHelpInfo()
{
  std::cout << "Usage: ./sample_maskRCNN [-h or --help] [-d or --datadir=<path "
               "to data directory>]"
            << std::endl;
  std::cout << "--help          Display help information" << std::endl;
  std::cout << "--datadir       Specify path to a data directory, overriding "
               "the default. This option can be used "
               "multiple times to add multiple directories. If no data "
               "directories are given, the default is to use "
               "data/samples/maskrcnn/ and data/maskrcnn/"
            << std::endl;
  std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
  std::cout << "--batch         Specify inference batch size." << std::endl;
}

int
main(int argc, char** argv)
{
  // todo CUDA_VISIBLE_DEVICES=0 和 cudaSetDevice(device_id) 运行速度不一致
  std::string device_id_str = argv[1];
  int device_id = std::stoi(device_id_str);
  std::cout << "device_id:" << device_id << std::endl;
  std::string config = argv[2];
  std::cout << "config:" << config << std::endl;

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

  SampleTrtMaskRCNNParams params = initializeSampleParams(config);
  SampleTrtMaskRCNN sample(params);

  gLogInfo << "Building and running a GPU inference engine for Mask RCNN"
           << std::endl;

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
  return gLogger.reportPass(sampleTest);
}
