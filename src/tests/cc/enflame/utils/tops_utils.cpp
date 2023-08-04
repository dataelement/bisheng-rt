#include "tops_utils.h"

#include <memory.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "TopsInference/TopsInferRuntime.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

int
get_dtype_size(TopsInference::DataType dtype)
{
  int dtype_size;
  switch (dtype) {
    case TopsInference::DataType::TIF_INT8:
    case TopsInference::DataType::TIF_UINT8:
      dtype_size = 1;
      break;
    case TopsInference::DataType::TIF_BF16:
    case TopsInference::DataType::TIF_FP16:
    case TopsInference::DataType::TIF_INT16:
    case TopsInference::DataType::TIF_UINT16:
      dtype_size = 2;
      break;
    case TopsInference::DataType::TIF_FP32:
    case TopsInference::DataType::TIF_INT32:
    case TopsInference::DataType::TIF_UINT32:
      dtype_size = 4;
      break;
    case TopsInference::DataType::TIF_FP64:
    case TopsInference::DataType::TIF_INT64:
    case TopsInference::DataType::TIF_UINT64:
      dtype_size = 8;
      break;
    default:  // to do.
      dtype_size = 1;
  }
  return dtype_size;
}

const char*
get_precision_str(int precision_type)
{
  const char* precision_str = "";
  switch ((TopsInference::BuildFlag)precision_type) {
    case TopsInference::BuildFlag::TIF_KTYPE_DEFAULT:
      precision_str = "default";
      break;
    case TopsInference::BuildFlag::TIF_KTYPE_FLOAT16:
      precision_str = "fp16";
      break;
    case TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16:
      precision_str = "mix";
      break;
  }
  return precision_str;
}

std::string
engine_name_construct(
    const char* onnx_path, const char* engine_folder, int batchsize,
    const char* precision)
{
  fs::path path(onnx_path);
  std::string stem = path.stem().string();
  std::ostringstream str_stream;
  str_stream.str("");
  str_stream << engine_folder << "/" << stem;
  str_stream << "-" << precision;
  if (batchsize > 0) {
    str_stream << "-bs" << batchsize;
  }
  str_stream << ".exec";
  std::string exec_path = str_stream.str();
  std::cout << "engine path: " << exec_path << '\n';
  return exec_path;
}

std::vector<ShapeInfo>
get_inputs_shape(TopsInference::IEngine* engine)
{
  std::vector<ShapeInfo> shapes_info;
  int num = engine->getInputNum();
  for (int i = 0; i < num; i++) {
    auto Dims = engine->getInputShape(i);
    auto dtype = engine->getInputDataType(i);

    std::vector<int> shape;
    int dtype_size = get_dtype_size(dtype);
    int mem_size = dtype_size;
    for (int j = 0; j < Dims.nbDims; j++) {
      shape.push_back(Dims.dimension[j]);
      mem_size *= Dims.dimension[j];
    }
    shapes_info.push_back(ShapeInfo(shape, dtype_size, mem_size));
  }
  return shapes_info;
}

std::vector<ShapeInfo>
get_outputs_shape(TopsInference::IEngine* engine)
{
  std::vector<ShapeInfo> shapes_info;
  int num = engine->getOutputNum();
  for (int i = 0; i < num; i++) {
    auto Dims = engine->getOutputShape(i);
    auto dtype = engine->getOutputDataType(i);

    std::vector<int> shape;
    int dtype_size = get_dtype_size(dtype);
    int mem_size = dtype_size;
    for (int j = 0; j < Dims.nbDims; j++) {
      shape.push_back(Dims.dimension[j]);
      mem_size *= Dims.dimension[j];
    }
    shapes_info.push_back(ShapeInfo(shape, dtype_size, mem_size));
  }
  return shapes_info;
}

std::vector<void*>
alloc_host_memory(std::vector<ShapeInfo>& shapes_info, int times, bool verbose)
{
  std::vector<void*> datum;
  for (auto& shape_info : shapes_info) {
    char* data = new char[shape_info.mem_size * times];
    memset(data, 0, shape_info.mem_size * times);
    datum.push_back((void*)data);
    if (verbose) {
      std::cout << "new data size: " << shape_info.mem_size << std::endl;
    }
  }
  return datum;
}

void
free_host_memory(std::vector<void*>& datum)
{
  for (auto& data : datum) {
    delete[](char*) data;
  }
  datum.clear();
}

bool
getEngineIOInfo(
    std::string& exec_path, std::vector<ShapeInfo>& inputs_shape_info,
    std::vector<ShapeInfo>& outputs_shape_info)
{
  if (access(exec_path.c_str(), 0) == -1) {
    return false;
  }

  void* tops_handler_;
  uint32_t clusterIds[] = {0};
  tops_handler_ = TopsInference::set_device(0, clusterIds);
  TopsInference::IEngine* engine = TopsInference::create_engine();
  engine->loadExecutable(exec_path.c_str());

  inputs_shape_info = get_inputs_shape(engine);
  outputs_shape_info = get_outputs_shape(engine);

  TopsInference::release_engine(engine);
  TopsInference::release_device(tops_handler_);
  // TopsInference::topsInference_finish();
  return true;
}

TopsInference::IEngine*
loadOrCreateEngine(
    const char* exec_path, const char* onnx_path,
    TopsInference::BuildFlag precision_flag, const char* input_names,
    const char* input_shapes)
{
  // load engine
  if (access(exec_path, 0) != -1) {
    TopsInference::IEngine* engine = TopsInference::create_engine();
    engine->loadExecutable(exec_path);
    std::cout << "[INFO] load engine file: " << exec_path << '\n';
    return engine;
  }
  // build engine from onnx
  if (access(onnx_path, 0) != -1) {
    TopsInference::IParser* parser_ =
        TopsInference::create_parser(TopsInference::TIF_ONNX);
    TopsInference::IOptimizer* optimizer_ = TopsInference::create_optimizer();
    if (input_names != NULL)
      parser_->setInputNames(input_names);
    if (input_shapes != NULL)
      parser_->setInputShapes(input_shapes);
    TopsInference::INetwork* network = parser_->readModel(onnx_path);
    TopsInference::IOptimizerConfig* optimizer_config = optimizer_->getConfig();
    optimizer_config->setBuildFlag(precision_flag);
    TopsInference::IEngine* engine = optimizer_->build(network);
    engine->saveExecutable(exec_path);
    std::cout << "[INFO] save engine file: " << exec_path << '\n';
    TopsInference::release_network(network);
    TopsInference::release_optimizer(optimizer_);
    TopsInference::release_parser(parser_);
    return engine;
  }
  std::cout << std::endl
            << "[ERROR] fail to load onnx: " << onnx_path << std::endl
            << std::endl;
  return NULL;
}
