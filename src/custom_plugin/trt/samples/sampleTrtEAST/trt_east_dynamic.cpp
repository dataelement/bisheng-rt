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
//! SampleEast.cpp
//! This file contains the implementation of the EAST sample. It creates the network
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
#include "east_model.h"
#include "model.h"
#include "NvInfer.h"

using namespace sample;

const std::string gSampleName = "TensorRT.sample_east";

void write_bin(const float* data, nvinfer1::Dims dims, float scale_h, float scale_w, std::string dir, std::string name) {
  if (!fileops::dir_exists(dir)) {
    fileops::create_dir(dir, 0755);
  }
  if (!fileops::dir_exists(dir+"/bin/")) {
    fileops::create_dir(dir+"/bin/", 0755);
  }
  if (!fileops::dir_exists(dir+"/shape/")) {
    fileops::create_dir(dir+"/shape/", 0755);
  }
  std::ofstream f_bin(dir+"/bin/"+name, std::ios::out | std::ios::binary);
  std::ofstream f_shape(dir+"/shape/"+name, std::ios::out | std::ios::binary);
  int size = 1;
  int shape_size = dims.nbDims+2;
  std::vector<float> shape(shape_size);
  for(int i=0; i<dims.nbDims; i++) {
    size *= dims.d[i];
    shape[i] = float(dims.d[i]);
  }
  shape[dims.nbDims] = scale_h;
  shape[dims.nbDims+1] = scale_w;
  f_shape.write((char*)shape.data(), shape_size*sizeof(float));
  f_bin.write((char*)data, size*sizeof(float));
}

struct SampleEastParams : public samplesCommon::SampleParams {
};

class SampleEast: public SampleModelDynamic {

 public:
  SampleEast(SampleEastParams& params)
    : SampleModelDynamic(params), mParams(params) {
  }

  bool predict();

  bool constructNetwork();

 private:
  SampleEastParams mParams;

  bool processInput(
    const std::string& binFile,
    const std::string& shapeFile,
    float& ratio_h,
    float& ratio_w,
    std::vector<int>& imgSzie,
    std::vector<float>& im) const;

  bool verifyOutput(
    const float& scale_h,
    const float& scale_w,
    const std::string& dst_dir,
    const std::string& name,
    std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs
  ) const;
};

bool SampleEast::constructNetwork() {
  std::vector<ITensor*> inputs = _context->setInputNode(mParams.inputTensorNames,
  {Dims4{1, -1, -1, 3}},
  {DataType::kFLOAT});

  std::vector<ITensor*> outputs;
  buildNetwork(_context, inputs, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  // Build engine
  _context->setWorkspaceSize(5_GiB);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }
  std::vector<std::vector<Dims>> inputsProfileDims = {{
      Dims4{1, 32, 32, 3},
      Dims4{1, 1056, 1056, 3},
      Dims4{1, 1600, 1600, 3}
    }
  };
  _context->setOptimizationProfile(inputsProfileDims);

  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

bool SampleEast::predict() {
  float total = 0;
  std::vector<std::string> names;
  fileops::files_in_dir(mParams.dataDirs[0]+"/bin", names);

  bool start_cnt = false;
  int cnt = 0;

  for (unsigned int i = 0; i < names.size(); i++) {
    bool outputCorrect = true;
    std::string shapeFile = mParams.dataDirs[0]+"/shape/"+names[i];
    std::string binFile = mParams.dataDirs[0]+"/bin/"+names[i];

    float ratio_h = 0, ratio_w=0;
    std::vector<int> imgSzie;
    std::vector<float> im;
    if (!processInput(binFile, shapeFile, ratio_h, ratio_w, imgSzie, im)) {
      return false;
    }
    const auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<void*> inputs = {im.data()};
    std::vector<Dims> dims = {Dims4(1, imgSzie[0], imgSzie[1], imgSzie[2])};
    std::map<std::string, std::pair<void*, nvinfer1::Dims>> outputs;

    auto status = infer(inputs, dims, outputs);
    const auto t_end = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    total += ms;

    cnt++;
    if(cnt == 100 && !start_cnt) {
      start_cnt = true;
      cnt = 0;
      total = 0;
    }
    if(start_cnt && cnt > 0 && cnt % 50 == 0)
      gLogInfo << "t:" << total << " cnt:" <<cnt << " t/per im:" << 1.0*total/cnt << std::endl;

    outputCorrect &= verifyOutput(ratio_h, ratio_w,
                                  mParams.dataDirs[1],
                                  names[i], outputs);

  }

  gLogInfo <<"t:"<< total << " cnt:" << cnt << " t/per im:"<< 1.0*total/cnt << std::endl;

  return true;
}

bool SampleEast::processInput(
  const std::string& binFile,
  const std::string& shapeFile,
  float& ratio_h,
  float& ratio_w,
  std::vector<int>& imgSzie,
  std::vector<float>& im
) const {
  std::ifstream f_shape(shapeFile, std::ios::binary);
  assert(f_shape.is_open() && "Unable to load weight file.");
  std::vector<float> shape(5);
  f_shape.read((char*)shape.data(), sizeof(float) * 5);
  int h = int(shape[0]);
  int w = int(shape[1]);
  int c = int(shape[2]);
  imgSzie.push_back(h);
  imgSzie.push_back(w);
  imgSzie.push_back(c);
  int size = c * h * w;
  ratio_h = shape[3];
  ratio_w = shape[4];
  std::ifstream f_im(binFile, std::ios::binary);
  im.resize(size);
  f_im.read((char*)im.data(), sizeof(float) * size);
  return true;
}

bool SampleEast::verifyOutput(
  const float& scale_h,
  const float& scale_w,
  const std::string& dst_dir,
  const std::string& name,
  std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs
) const {
  if (!fileops::dir_exists(dst_dir)) {
    fileops::create_dir(dst_dir, 0755);
  }
  const float* score = (float *)outputs[mParams.outputTensorNames[0]].first;
  Dims score_dims = outputs[mParams.outputTensorNames[0]].second;

  const float* geometry = (float *)outputs[mParams.outputTensorNames[1]].first;
  Dims geometry_dims = outputs[mParams.outputTensorNames[1]].second;

  const float* cos_map = (float *)outputs[mParams.outputTensorNames[2]].first;
  Dims cos_dims = outputs[mParams.outputTensorNames[2]].second;

  const float* sin_map = (float *)outputs[mParams.outputTensorNames[3]].first;
  Dims sin_dims = outputs[mParams.outputTensorNames[3]].second;

  write_bin(score, score_dims, scale_h, scale_w, dst_dir+"/score", name);
  write_bin(geometry, geometry_dims, scale_h, scale_w, dst_dir+"/geometry", name);
  write_bin(cos_map, cos_dims, scale_h, scale_w, dst_dir+"/cos_map", name);
  write_bin(sin_map, sin_dims, scale_h, scale_w, dst_dir+"/sin_map", name);
}

SampleEastParams initializeSampleParams(const samplesCommon::Args& args) {
  SampleEastParams params;

  params.inputTensorNames.push_back("inputs");
  params.outputTensorNames.push_back("output/score");
  params.outputTensorNames.push_back("output/geometry");
  params.outputTensorNames.push_back("output/cos_map");
  params.outputTensorNames.push_back("output/sin_map");
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;
  params.weightsFile = "../test_data/models/east_v5_angle.wts";
  params.engineFile = "../test_data/models/east_dynamic_trt.trt";

  if (args.dataDirs.empty()) {
    // input images
    params.dataDirs.push_back("../test_data/ocr_east_data/im_raw_no_fix_nhwc");
    // result save path
    params.dataDirs.push_back("../test_data/ocr_east_data/sampleTrtEAST_dynamic_trt_fp32");
  } else {
    params.dataDirs = args.dataDirs;
  }

  return params;
}

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

  SampleEastParams params = initializeSampleParams(args);

  SampleEast sample(params);

  gLogInfo << "Building and running a GPU inference engine for EAST" << std::endl;

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
