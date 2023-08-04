#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "transformer_model.h"
#include "model.h"

#include <random>

#include <fstream>
#include <cmath>
#include <jsoncpp/json/json.h>
using namespace sample;

const std::string gSampleName = "TensorRT.sample_transformer";

class SampleTransformer: public SampleModelDynamic {

 public:
  SampleTransformer(SampleTrtTransformerParams& params)
    : SampleModelDynamic(params), mParams(params) {
  }

  ~SampleTransformer() {}

  bool constructNetwork();

 private:
  SampleTrtTransformerParams mParams;
};

bool SampleTransformer::constructNetwork() {
  std::vector<ITensor*> inputs = _context->setInputNode(mParams.inputTensorNames,
  {Dims4{-1, 32, -1, 1}, Dims2{-1, 2}},
  {DataType::kFLOAT, DataType::kINT32});

  std::vector<ITensor*> outputs;
  buildNetwork(_context, inputs, mParams, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  // Build engine
  _context->setWorkspaceSize(mParams.workspaceSize);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }
  std::vector<std::vector<Dims>> inputsProfileDims = {{
      Dims4{1, 32, 32, 1},
      Dims4{48, 32, 832, 1},
      Dims4{96, 32, 1032, 1}
    },
    {
      Dims2{1, 2},
      Dims2{48, 2},
      Dims2{96, 2}
    }
  };
  _context->setOptimizationProfile(inputsProfileDims);

  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

SampleTrtTransformerParams initializeSampleParams(const std::string& config) {
  SampleTrtTransformerParams params;
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
    params.hidden_size = root["model_params"]["hidden_size"].asInt();
    params.num_heads = root["model_params"]["num_heads"].asInt();
    params.beam_width = root["model_params"]["beam_width"].asInt();
    params.start_id = root["model_params"]["start_id"].asInt();
    params.end_id = root["model_params"]["end_id"].asInt();
    params.num_layer = root["model_params"]["num_layer"].asInt();
    params.channel_axis = root["model_params"]["channel_axis"].asInt();
    params.resnet_vd = root["model_params"]["resnet_vd"].asBool();
    params.vocab_size = root["model_params"]["vocab_size"].asInt();
    if (params.resnet_vd) {
      params.downsample = 4;
    } else {
      params.downsample = 8;
    }
    params.fp16 = root["fp16"].asBool();
    params.workspaceSize = root["workspaceSize"].asInt64();
    params.weightsFile = root["weightsFile"].asString();
    params.engineFile = root["engineFile"].asString();
  } else {
    std::cout << "parse error\n" << std::endl;
  }
  in.close();
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
  std::string config = argv[2];
  std::cout<< "config:" << config << std::endl;

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

  SampleTrtTransformerParams params = initializeSampleParams(config);

  SampleTransformer sample(params);

  gLogInfo << "Building and running a GPU inference engine for Transformer" << std::endl;

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
