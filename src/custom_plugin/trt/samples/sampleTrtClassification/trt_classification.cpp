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

//!
//! SampleClassification.cpp
//! This file contains the implementation of the CLASSIFICATION sample. It creates the network
//! It can be run with the following command line:
//! Command: ./sample_mlp [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir] [--useDLACore=<int>]
//!

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "fileops.hpp"
#include "netUtils.h"
#include "model.h"
#include "classification_model.cpp"
#include "NvInfer.h"

using namespace sample;

const std::string gSampleName = "TensorRT.sample_classification";

struct SampleClassificationParams : public samplesCommon::SampleParams {
  int inputH;              //!< The input height
  int inputW;              //!< The input width
};

class SampleClassification: public SampleModel {

 public:
  SampleClassification(SampleClassificationParams& params) : SampleModel(params)
    , mParams(params) {
  }

  bool constructNetwork();

  bool predict();

 private:
  SampleClassificationParams mParams; //!< The parameters for the sample.

  bool processInput(
    const std::vector<std::string>& binFile,
    const std::vector<std::string>& shapeFile,
    std::vector<int>& imgSzie,
    std::vector<float>& im) const;

  bool verifyOutput(
    const std::string& dst_dir,
    const std::vector<std::string>& name,
    std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs
  ) const;

};

bool SampleClassification::constructNetwork() {
  std::vector<ITensor*> inputs = _context->setInputNode(mParams.inputTensorNames,
  {Dims4{1, mParams.inputH, mParams.inputW, 3}},
  {DataType::kFLOAT});
  std::vector<ITensor*> outputs;
  buildNetwork(_context, inputs, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  // Build engine
  _context->setBatchSize(mParams.batchSize);
  _context->setWorkspaceSize(5_GiB);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }

  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

bool SampleClassification::predict() {
  float total = 0;
  std::vector<std::string> names;
  fileops::files_in_dir(mParams.dataDirs[0]+"/bin", names);

  bool start_cnt = false;
  int cnt = 0;
  for (unsigned int i = 0; i < names.size(); i = i + mParams.batchSize) {
    bool outputCorrect = true;
    std::vector<std::string> shapeFiles;
    std::vector<std::string> binFiles;
    std::vector<std::string> nameFiles;
    for (unsigned j = 0; j < mParams.batchSize; j++) {
        if (i + j >= names.size()) {
            break;
        }
        std::string binFile = mParams.dataDirs[0]+"/bin/"+names[i+j];
        std::string shapeFile = mParams.dataDirs[0]+"/shape/"+names[i+j];
        nameFiles.emplace_back(names[i+j]);
        shapeFiles.emplace_back(shapeFile);
        binFiles.emplace_back(binFile);
    }
    std::vector<float> im;
    std::vector<int> imgSzie;
    if (!processInput(binFiles, shapeFiles, imgSzie, im)) {
      return false;
    }
    const auto t_start = std::chrono::high_resolution_clock::now();

    std::map<std::string, std::pair<void*, nvinfer1::Dims>> outputs;
    std::vector<void*> inputs = {im.data()};
    std::vector<Dims> dims = {Dims4(imgSzie[0], imgSzie[1], imgSzie[2], imgSzie[3])};
    int batch_size = imgSzie[0];
    auto status = infer(batch_size, inputs, dims, outputs);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    total += ms;

    cnt += mParams.batchSize;
    if(cnt / mParams.batchSize >= 50 && !start_cnt) {
      start_cnt = true;
      cnt = 0;
      total = 0;
    }
    if(start_cnt && cnt > 0 && cnt % (mParams.batchSize * 4) == 0)
      gLogInfo << "t:" << total << " cnt:" <<cnt << " t/per im:" << 1.0*total/cnt << std::endl;
    outputCorrect &= verifyOutput(mParams.dataDirs[1], nameFiles, outputs);
  }

  gLogInfo <<"t:"<< total << " cnt:" << cnt << " t/per im:"<< 1.0*total/cnt << std::endl;

  return true;
}

bool SampleClassification::processInput(
  const std::vector<std::string>& binFiles,
  const std::vector<std::string>& shapeFiles,
  std::vector<int>& imgSzie,
  std::vector<float>& im) const {
  int imageSzie = mParams.inputH * mParams.inputW * 3;
  int batchSize = binFiles.size();
  imgSzie.push_back(batchSize);
  imgSzie.push_back(mParams.inputH);
  imgSzie.push_back(mParams.inputW);
  imgSzie.push_back(3);
  im.resize(imageSzie * batchSize);
  for (unsigned int j = 0; j < batchSize; ++j) {
      std::ifstream f_shape(shapeFiles[j], std::ios::binary);
      assert(f_shape.is_open() && "Unable to load weight file.");
      std::vector<float> shape(5);
      f_shape.read((char*)shape.data(), sizeof(float) * 5);
      int h = int(shape[0]);
      int w = int(shape[1]);
      int c = int(shape[2]);
      int size = c * h * w;
      std::ifstream f_im(binFiles[j], std::ios::binary);
      std::vector<float> im_tmp(size);
      f_im.read((char*)im_tmp.data(), sizeof(float) * size);
      for (int i = 0; i < size; i++)
      {
          im[j * size + i] = im_tmp[i];
      }
  }
  return true;
}

bool SampleClassification::verifyOutput(
  const std::string& dst_dir,
  const std::vector<std::string>& names,
  std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs) const {
  if (!fileops::dir_exists(dst_dir)) {
    fileops::create_dir(dst_dir, 0755);
  }
  const float* scores = (float *)outputs[mParams.outputTensorNames[0]].first;
  Dims score_dims = outputs[mParams.outputTensorNames[0]].second;
  for (unsigned i = 0; i < names.size(); ++i) {
      const float* score = scores + DimsCount(score_dims) * i;
      write_bin<float>(score, score_dims, dst_dir+"/score", names[i]);
  }
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleClassificationParams initializeSampleParams(const samplesCommon::Args& args) {
  SampleClassificationParams params;

  params.inputTensorNames.push_back("inputs");
  params.outputTensorNames.push_back("output");
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;
  params.batchSize = 1;
  params.weightsFile = "../test_data/models/classification.wts";
  params.engineFile = "../test_data/models/classification_fix_trt.trt";

  if (args.dataDirs.empty()) { //!< Use default directories if user hasn't provided directory paths
    // input images
    params.dataDirs.push_back("../test_data/ocr_classification_data/im_raw_fix_nhwc");
    // result save path
    params.dataDirs.push_back("../test_data/ocr_classification_data/sampleTrtCLASSIFICATION_fix_trt_fp32");
  } else { //!< Use the data directory provided by the user
    params.dataDirs = args.dataDirs;
  }

  params.inputH = 224;
  params.inputW = 224;

  return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo() {
  std::cout << "Usage: ./sample_mlp [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
            << std::endl;
  std::cout << "--help          Display help information" << std::endl;
  std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
            "multiple times to add multiple directories. If no data directories are given, the default is to use "
            "(data/samples/mlp/, data/mlp/)"
            << std::endl;
  std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
            "where n is the number of DLA engines on the platform."
            << std::endl;
  std::cout << "--int8          Run in Int8 mode." << std::endl;
  std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv) {
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

  SampleClassificationParams params = initializeSampleParams(args);

  SampleClassification sample(params);

  gLogInfo << "Building and running a GPU inference engine for CLASSIFICATION" << std::endl;

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
