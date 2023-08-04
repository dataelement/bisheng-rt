#include <random>
#include <fstream>
#include <cmath>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "transformer_model.h"
#include "model.h"

using namespace sample;

const std::string gSampleName = "TensorRT.sample_transformer";

class SampleTransformer: public SampleModelDynamic {

 public:
  SampleTransformer(SampleTrtTransformerParams& params)
    : SampleModelDynamic(params), mParams(params) {
  }

  ~SampleTransformer() {}

  bool predict();

  bool constructNetwork();

 private:
  SampleTrtTransformerParams mParams;

  bool processInput(
    const std::string& im_name,
    const std::string& shape_name,
    const std::string& input_shape_name,
    std::vector<int>& img_dims,
    std::vector<float>& im,
    std::vector<int>& im_shape) const;

  bool verifyOutput(
    const std::string& dst_dir,
    const std::string& name,
    std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs
  ) const;
};

bool SampleTransformer::constructNetwork() {
  std::vector<ITensor*> inputs = _context->setInputNode(mParams.inputTensorNames,
  {Dims4{-1, 32, -1, 1}, Dims2{-1, 2}},
  {DataType::kFLOAT, DataType::kINT32});

  std::vector<ITensor*> outputs;
  buildNetwork(_context, inputs, mParams, outputs);
  _context->setOutputNode(outputs, mParams.outputTensorNames);

  // Build engine
  _context->setWorkspaceSize(6_GiB);
  if (mParams.fp16) {
    _context->setFp16Mode();
  }
  std::vector<std::vector<Dims>> inputsProfileDims = {{
      Dims4{1, 32, 32, 1},
      Dims4{64, 32, 832, 1},
      Dims4{128, 32, 1032, 1}
    },
    {
      Dims2{1, 2},
      Dims2{64, 2},
      Dims2{128, 2}
    }
  };
  _context->setOptimizationProfile(inputsProfileDims);

  auto engine = _context->getICudaEngine();
  if (!engine) {
    return false;
  }
  return true;
}

bool SampleTransformer::predict() {
  float total = 0;
  std::string txt_path = mParams.dataDirs[0] + "/img_list_sort_decent.txt";
  std::vector<std::string> names;
  fileops::files_in_txt(txt_path, names);
  std::cout << "image num:" << names.size() << std::endl;

  bool start_cnt = false;
  int cnt = 0;

  for (unsigned int i = 0; i < names.size(); i++) {
    bool outputCorrect = true;
    std::string name = names[i];
    name = name.substr(0, name.length() - 1);
    std::string shape_name = mParams.dataDirs[0]+"/inputs/shape/"+name;
    std::string im_name = mParams.dataDirs[0] + "/inputs/bin/"+name;
    std::string input_shape_name = mParams.dataDirs[0] + "/inputs_shape/bin/"+name;

    std::vector<int> img_dims;
    std::vector<float> im;
    std::vector<int> im_shape;
    if (!processInput(im_name, shape_name, input_shape_name, img_dims, im, im_shape)) {
      return false;
    }
    const auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<void*> inputs = {im.data(), im_shape.data()};
    std::vector<Dims> dims = {Dims4(img_dims[0], img_dims[1], img_dims[2], img_dims[3]),
                              Dims2(img_dims[0], 2)
                             };
    std::map<std::string, std::pair<void*, nvinfer1::Dims>> outputs;
    auto status = infer(inputs, dims, outputs);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    total += ms;

    cnt++;
    if(cnt == 10 && !start_cnt) {
      start_cnt = true;
      cnt = 0;
      total = 0;
    }
    if(start_cnt && cnt > 0 && cnt % 10 == 0)
      gLogInfo << "t:" << total << " cnt:" <<cnt << " t/per im:" << 1.0*total/cnt << std::endl;

    outputCorrect &= verifyOutput(mParams.dataDirs[1], name, outputs);

  }

  gLogInfo <<"t:"<< total << " cnt:" << cnt << " t/per im:"<< 1.0*total/cnt << std::endl;

  return true;
}

bool SampleTransformer::processInput(
  const std::string& im_name,
  const std::string& shape_name,
  const std::string& input_shape_name,
  std::vector<int>& img_dims,
  std::vector<float>& im,
  std::vector<int>& im_shape
) const {
  std::ifstream f_shape(shape_name, std::ios::binary);
  assert(f_shape.is_open() && "Unable to load weight file.");
  img_dims.resize(4);
  f_shape.read((char*)img_dims.data(), sizeof(int) * 4);
  int b = img_dims[0];
  int h = img_dims[1];
  int w = img_dims[2];
  int c = img_dims[3];
  int size = b * h * w * c;

  std::ifstream f_inputs(im_name, std::ios::binary);
  im.resize(size);
  f_inputs.read((char*)im.data(), sizeof(float) * size);

  std::ifstream f_inputs_shape(input_shape_name, std::ios::binary);
  size = b * 2;
  im_shape.resize(size);
  f_inputs_shape.read((char*)im_shape.data(), sizeof(int) * size);
  return true;
}

bool SampleTransformer::verifyOutput(
  const std::string& dst_dir,
  const std::string& name,
  std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs
) const {
  if (!fileops::dir_exists(dst_dir)) {
    fileops::create_dir(dst_dir, 0755);
  }
  const int* output_ids = (int *)outputs[mParams.outputTensorNames[0]].first;
  Dims output_ids_dims = outputs[mParams.outputTensorNames[0]].second;

  const int* parent_ids = (int *)outputs[mParams.outputTensorNames[1]].first;
  Dims parent_ids_dims = outputs[mParams.outputTensorNames[1]].second;

  const int* sequence_length = (int *)outputs[mParams.outputTensorNames[2]].first;
  Dims sequence_length_dims = outputs[mParams.outputTensorNames[2]].second;

  // show<int>(output_ids, output_ids_dims, name);
  // show<int>(parent_ids, parent_ids_dims, name);
  // show<int>(sequence_length, sequence_length_dims, name);
  write_bin<int>(output_ids, output_ids_dims, dst_dir+"/output_ids", name);
  write_bin<int>(parent_ids, parent_ids_dims, dst_dir+"/parent_ids", name);
  write_bin<int>(sequence_length, sequence_length_dims, dst_dir+"/sequence_length", name);
}

SampleTrtTransformerParams initializeSampleParams(const samplesCommon::Args& args) {
  SampleTrtTransformerParams params;

  params.inputTensorNames.push_back("inputs");
  params.inputTensorNames.push_back("inputs_shape");
  params.outputTensorNames.push_back("output_ids");
  params.outputTensorNames.push_back("parent_ids");
  params.outputTensorNames.push_back("sequence_length");
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;
  params.weightsFile = "/home/gulixin/OCR-RECOGNITION-MODELS/transformer/2.8/tf/savedmodel/transformer_weights_trt_op.wts";
  params.engineFile = "../test_data/models/transformer_v2.8_fp16_3090.trt";

  if (args.dataDirs.empty()) {
    // input images
    params.dataDirs.push_back("../test_data/ocr_trans_ctc_data/im_raw_gray_sort_socr_hwc");
    // result save path
    params.dataDirs.push_back("../test_data/ocr_trans_ctc_data/transformer_v2.8_socr_trt_fp16_hwc");
  } else {
    params.dataDirs = args.dataDirs;
  }

  params.hidden_size = 512;
  params.num_heads = 8;
  params.beam_width = 5;
  params.start_id = 0;
  params.end_id = 1;
  params.num_layer = 3;
  params.channel_axis = 1;
  params.resnet_vd = false;
  params.vocab_size = 6412;
  if (params.resnet_vd) {
    params.downsample = 4;
  } else {
    params.downsample = 8;
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

  SampleTrtTransformerParams params = initializeSampleParams(args);

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
  if (!sample.predict()) {
    return gLogger.reportFail(sampleTest);
  }

  return gLogger.reportPass(sampleTest);
}
