// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#include <algorithm>
#include <memory>
#include <numeric>

#include "TopsInference/TopsInferRuntime.h"
#include "TopsInference/dtu/util/switch_logging.h"
#include "TopsInference/utils/tops_utils.h"
#include "enflame_backend_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

struct TRITONENFLAME_Tensor;

// Enflame Predictor Wrapper
struct TRITONENFLAME_Model;

typedef TopsInference::topsInferStream_t Stream;
typedef TopsInference::IEngine Predictor;

// struct TRITONENFLAME_STREAM;
// struct TRITONENFLAME_IEngine;
// typedef TRITONENFLAME_STREAM Stream;
// typedef TRITONENFLAME_IEngine Predictor;

typedef triton::common::TritonJson::Value JsonValue;

class ModelImpl {
 public:
  ModelImpl(const char* model_path, JsonValue* model_config, int instance_id);
  ~ModelImpl();

  TRITONENFLAME_Error* Run();

  TRITONENFLAME_Error* GetInputPtr(
      const char* name, const TRITONENFLAME_DataType dtype,
      const TRITONENFLAME_Shape& shape, char** ptr);

  TRITONENFLAME_Error* GetOutputMetadata(
      const char* name, TRITONENFLAME_DataType* dtype,
      TRITONENFLAME_Shape* shape, char** ptr);

  // TRITONENFLAME_Error* ZeroCopyRun();

 private:
  Predictor* predictor_;
  Stream stream_;

  void* tops_handler_;
  std::vector<void*> device_input_handlers_;
  std::vector<void*> device_output_handlers_;

  std::vector<void*> host_input_handlers_;
  std::vector<void*> host_output_handlers_;

  std::vector<ShapeInfo> inputs_shape_info_;
  std::vector<ShapeInfo> outputs_shape_info_;

  std::vector<TRITONENFLAME_Shape> inputs_shape_;
  std::vector<TRITONENFLAME_Shape> outputs_shape_;

  std::vector<TRITONENFLAME_DataType> inputs_type_;
  std::vector<TRITONENFLAME_DataType> outputs_type_;

  // input,output must in format INPUT_x, OUTPUT_x
  std::unordered_map<std::string, int> input_name_map_;
  std::unordered_map<std::string, int> output_name_map_;

  int card_id_ = 0;
  std::vector<uint32_t> cluster_ids_ = {0};
  int cluster_size_ = 1;
  int max_batch_size_;
  bool need_set_device_ = true;
};


ModelImpl::ModelImpl(
    const char* model_path, JsonValue* model_config, int instance_id)
{
  int cluster_id = 0;

  // uint32_t clusterIds[] = {0};

  // parse model config
  TRITONSERVER_Error* err = nullptr;
  do {
    JsonValue params, json_value;
    model_config->Find("parameters", &params);
    if (params.Find("card_id", &json_value)) {
      std::string string_value;
      CHECK_ERROR_WITH_BREAK(
          err = json_value.MemberAsString("string_value", &string_value));
      card_id_ = atoi(string_value.c_str());
    }
    if (params.Find("cluster_id", &json_value)) {
      std::string string_value;
      CHECK_ERROR_WITH_BREAK(
          err = json_value.MemberAsString("string_value", &string_value));
      cluster_id = atoi(string_value.c_str());
    }

    triton::common::TritonJson::Value ios;
    CHECK_ERROR_WITH_BREAK(err = model_config->MemberAsArray("input", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      CHECK_ERROR_WITH_BREAK(err = ios.IndexAsObject(i, &io));
      std::string io_dtype;
      CHECK_ERROR_WITH_BREAK(err = io.MemberAsString("data_type", &io_dtype));
      inputs_type_.push_back(ConvertDataType(io_dtype));
      std::string name;
      CHECK_ERROR_WITH_BREAK(err = io.MemberAsString("name", &name));
      input_name_map_.emplace(name, i);
    }

    CHECK_ERROR_WITH_BREAK(err = model_config->MemberAsArray("output", &ios));
    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      CHECK_ERROR_WITH_BREAK(err = ios.IndexAsObject(i, &io));
      std::string io_dtype;
      CHECK_ERROR_WITH_BREAK(err = io.MemberAsString("data_type", &io_dtype));
      outputs_type_.push_back(ConvertDataType(io_dtype));
      std::string name;
      CHECK_ERROR_WITH_BREAK(err = io.MemberAsString("name", &name));
      output_name_map_.emplace(name, i);
    }
  } while (false);

  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
  }

  TopsInference::topsInference_init();
  // std::string modelname;
  // model_config->MemberAsString("name", &modelname);

  cluster_ids_ = {(uint32_t)cluster_id};
  cluster_size_ = 1;
  std::cout<<"card_id_:"<<card_id_<<" cluster_id:"<<cluster_id<<std::endl;
  tops_handler_ =
      TopsInference::set_device(card_id_, cluster_ids_.data(), cluster_size_);

  // THROW_CHECK_ERROR(tops_handler_ != nullptr, "failed to create tops
  // handler");

  predictor_ = TopsInference::create_engine();
  predictor_->loadExecutable(model_path);

  inputs_shape_info_ = get_inputs_shape(predictor_);
  outputs_shape_info_ = get_outputs_shape(predictor_);

  for (size_t i = 0; i < inputs_shape_info_.size(); i++) {
    auto input_men_size = inputs_shape_info_[i].mem_size;
    void* storage_on_device = nullptr;
    TopsInference::mem_alloc(&storage_on_device, input_men_size);
    device_input_handlers_.push_back(storage_on_device);
    void* buff = malloc(input_men_size);
    host_input_handlers_.push_back(buff);
    //   input_name_map_.emplace(std::string("INPUT_") + std::to_string(i), i);
    inputs_shape_.emplace_back(inputs_shape_info_[i].dims);
  }
  // host_input_handlers_.resize(inputs_shape_info_.size());

  for (size_t i = 0; i < outputs_shape_info_.size(); i++) {
    auto output_men_size = outputs_shape_info_[i].mem_size;
    void* storage_on_device = nullptr;
    TopsInference::mem_alloc(&storage_on_device, output_men_size);
    device_output_handlers_.push_back(storage_on_device);
    void* buff = malloc(output_men_size);
    host_output_handlers_.push_back(buff);
    //   output_name_map_.emplace(std::string("OUTPUT_") + std::to_string(i),
    //   i);
    outputs_shape_.emplace_back(outputs_shape_info_[i].dims);
  }

  TopsInference::create_stream(&stream_);
}

ModelImpl::~ModelImpl()
{
  // // Release the resource
  // // make all streams synchronized
  TopsInference::synchronize_stream(stream_);

  // // destroy created stream and release resource
  TopsInference::destroy_stream(stream_);

  // // free memory on device
  for (auto* storage : device_input_handlers_) {
    TopsInference::mem_free(storage);
  }

  for (auto* storage : device_output_handlers_) {
    TopsInference::mem_free(storage);
  }

  for (auto* storage : host_input_handlers_) {
    free(storage);
  }

  for (auto* storage : host_output_handlers_) {
    free(storage);
  }

  TopsInference::release_engine(predictor_);
  TopsInference::release_device(tops_handler_);
  TopsInference::topsInference_finish();
}

TRITONENFLAME_Error*
ModelImpl::Run()
{
  if(need_set_device_){
    tops_handler_ = TopsInference::set_device(card_id_, cluster_ids_.data(), cluster_size_);
    need_set_device_ = false;
  }

  for (size_t i = 0; i < device_input_handlers_.size(); i++) {
    TopsInference::mem_copy_async(
        host_input_handlers_[i], device_input_handlers_[i],
        inputs_shape_info_[i].mem_size,
        TopsInference::MemcpyKind::TIF_MEMCPY_HOST_TO_DEVICE, stream_);
  }

  bool status = predictor_->run(
      (void**)device_input_handlers_.data(),
      (void**)device_output_handlers_.data(),
      TopsInference::BufferType::TIF_ENGINE_RSC_IN_DEVICE_OUT_DEVICE, stream_);

  if (!status) {
    return TRITONENFLAME_ErrorNew(std::string("engine run failed!"));
  }

  for (size_t i = 0; i < device_output_handlers_.size(); i++) {
    TopsInference::mem_copy_async(
        device_output_handlers_[i], host_output_handlers_[i],
        outputs_shape_info_[i].mem_size,
        TopsInference::MemcpyKind::TIF_MEMCPY_DEVICE_TO_HOST, stream_);
  }

  TopsInference::synchronize_stream(stream_);
  return nullptr;
}

TRITONENFLAME_Error*
ModelImpl::GetInputPtr(
    const char* name, const TRITONENFLAME_DataType dtype,
    const TRITONENFLAME_Shape& shape, char** ptr)
{
  int index = input_name_map_[name];

  if (inputs_type_[index] != dtype) {
    return TRITONENFLAME_ErrorNew("inputs types is not matched");
  }

  // todo: get the real tensor
  *ptr = reinterpret_cast<char*>(host_input_handlers_[index]);

  return nullptr;
}

TRITONENFLAME_Error*
ModelImpl::GetOutputMetadata(
    const char* name, TRITONENFLAME_DataType* dtype, TRITONENFLAME_Shape* shape,
    char** ptr)
{
  int index = output_name_map_[name];
  *dtype = outputs_type_[index];
  *shape = outputs_shape_[index];
  *ptr = reinterpret_cast<char*>(host_output_handlers_[index]);
  return nullptr;
}

TRITONSERVER_Error*
TRITONENFLAME_ModelCreate(
    TRITONENFLAME_Model** model, const char* model_path, JsonValue* config,
    int instance_id)
{
  try {
    ModelImpl* model_impl = new ModelImpl(model_path, config, instance_id);
    *model = reinterpret_cast<TRITONENFLAME_Model*>(model_impl);
  }
  catch (const TRITONENFLAME_Exception& ex) {
    RETURN_IF_TRITONENFLAME_ERROR(ex.err_);
  }
  return nullptr;
}

void

TRITONENFLAME_ModelDelete(TRITONENFLAME_Model* model)
{
  if (model != nullptr) {
    ModelImpl* mi = reinterpret_cast<ModelImpl*>(model);
    delete mi;
  }
}

TRITONENFLAME_Error*
TRITONENFLAME_ModelRun(TRITONENFLAME_Model* model)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  return m->Run();
}

class TensorImpl {
 public:
  TensorImpl(
      const char* name, TRITONENFLAME_DataType dtype,
      const TRITONENFLAME_Shape& shape, char* data_ptr);
  ~TensorImpl() = default;

  const std::string& Name() const { return name_; }
  TRITONENFLAME_DataType DataType() const { return dtype_; }
  TRITONENFLAME_Shape Shape() const { return shape_; }

  char* Base() const { return base_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  const std::string name_;
  const TRITONENFLAME_DataType dtype_;
  const TRITONENFLAME_Shape shape_;

  char* base_;
  size_t byte_size_;
};

TensorImpl::TensorImpl(
    const char* name, TRITONENFLAME_DataType dtype,
    const TRITONENFLAME_Shape& shape, char* data_ptr)
    : name_(name), dtype_(dtype), shape_(shape), base_(data_ptr)
{
  byte_size_ = shape.NumElements() * TRITONENFLAME_DataTypeByteSize(dtype);
}

TRITONENFLAME_Tensor*
TRITONENFLAME_TensorNew(
    TRITONENFLAME_Model* model, const char* name, TRITONENFLAME_DataType dtype,
    const TRITONENFLAME_Shape& shape)
{
  char* data_ptr;
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetInputPtr(name, dtype, shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONENFLAME_Tensor*>(tensor);
}

TRITONENFLAME_Tensor*
TRITONENFLAME_TensorNew(TRITONENFLAME_Model* model, const char* name)
{
  char* data_ptr;
  TRITONENFLAME_DataType dtype;
  TRITONENFLAME_Shape shape;

  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetOutputMetadata(name, &dtype, &shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONENFLAME_Tensor*>(tensor);
}

char*
TRITONENFLAME_TensorData(TRITONENFLAME_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Base();
}

size_t
TRITONENFLAME_TensorDataByteSize(TRITONENFLAME_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->ByteSize();
}

TRITONENFLAME_DataType
TRITONENFLAME_TensorDataType(TRITONENFLAME_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->DataType();
}

TRITONENFLAME_Shape
TRITONENFLAME_TensorShape(TRITONENFLAME_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Shape();
}

namespace triton { namespace backend { namespace enflame {

using TRITONENFLAMEModelHandle = std::shared_ptr<TRITONENFLAME_Model>;

class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Auto-complete the model configuration
  TRITONSERVER_Error* AutoCompleteConfig();

  // Validate that model configuration is supported by this backend
  TRITONSERVER_Error* ValidateModelConfig();
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration if requests
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for enflame backend")
          .c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        ConvertDataType(io_dtype) ==
            TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_TRUE(
        ConvertDataType(io_dtype) ==
            TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INVALID,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '" +
            io_name + "' for model '" + Name() + "'");
  }

  return nullptr;  // success
}

class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* DetermineModelAndParamsPath(
      const std::string& model_dir, std::string* model_path);

  void SetInputTensors(
      // BackendInputCollector* collector,
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<std::string>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;
  TRITONENFLAMEModelHandle triton_enflame_model_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::DetermineModelAndParamsPath(
    const std::string& model_dir, std::string* model_path)
{
  bool exists;
  *model_path = JoinPath({model_dir, "model.exec"});
  RETURN_IF_ERROR(FileExists(*model_path, &exists));
  if (not exists) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("enflame model should be named as 'model.exec'").c_str());
  }

  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  auto model_dir = JoinPath(
      {model_state->RepositoryPath(), std::to_string(model_state->Version())});
  auto& model_config = model_state->ModelConfig();
  std::string model_path;
  THROW_IF_BACKEND_INSTANCE_ERROR(
      DetermineModelAndParamsPath(model_dir, &model_path));

  const char* cname;
  TRITONBACKEND_ModelInstanceName(triton_model_instance, &cname);
  std::string name(cname);
  int instance_id = atoi(name.substr(name.size() - 1, 1).c_str());
  TRITONENFLAME_Model* triton_enflame_model = nullptr;
  THROW_IF_BACKEND_INSTANCE_ERROR(TRITONENFLAME_ModelCreate(
      &triton_enflame_model, model_path.c_str(), &model_config, instance_id));
  triton_enflame_model_.reset(triton_enflame_model, TRITONENFLAME_ModelDelete);
}

void
ModelInstanceState::SetInputTensors(
    // BackendInputCollector* collector, size_t total_batch_size,
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, responses,
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedInput(), CudaStream());

  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; ++input_idx) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &name, &datatype, &shape, &dims_count, nullptr, nullptr));

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(shape, shape + dims_count);

    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    TRITONENFLAME_Tensor* tensor = TRITONENFLAME_TensorNew(
        triton_enflame_model_.get(), name, ConvertDataType(datatype),
        TRITONENFLAME_Shape(batchn_shape));

    if (tensor == nullptr) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to create input tensor '") + name +
           "' with shape " + backend::ShapeToString(batchn_shape) +
           " and data type " + TRITONSERVER_DataTypeString(datatype) +
           " for '" + Name() + "'")
              .c_str());
      SendErrorForResponses(responses, request_count, err);
      return;
    }

    collector.ProcessTensor(
        name, TRITONENFLAME_TensorData(tensor),
        TRITONENFLAME_TensorDataByteSize(tensor), TRITONSERVER_MEMORY_CPU, 0);
  }

  cuda_copy |= collector.Finalize();
  if (cuda_copy) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("cannot do cuda copy ") + Name()).c_str());
    SendErrorForResponses(responses, request_count, err);
  }
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<std::string>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, StateForModel()->MaxBatchSize(),
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedOutput(), CudaStream());

  bool cuda_copy = false;
  for (size_t idx = 0; idx < output_names.size(); ++idx) {
    const std::string& name = output_names[idx];

    TRITONENFLAME_Tensor* tensor =
        TRITONENFLAME_TensorNew(triton_enflame_model_.get(), name.c_str());

    if (tensor == nullptr) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to create output tensor '") + name + " for '" +
           Name() + "'")
              .c_str());
      SendErrorForResponses(responses, request_count, err);
      return;
    }

    auto dtype = ConvertDataType(TRITONENFLAME_TensorDataType(tensor));
    auto shape = TRITONENFLAME_TensorShape(tensor).Shape();
    responder.ProcessTensor(
        name, dtype, shape, TRITONENFLAME_TensorData(tensor),
        TRITONSERVER_MEMORY_CPU, 0);
  }

  cuda_copy |= responder.Finalize();
  if (cuda_copy) {
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("can not do cuda copy ") + Name()).c_str());
    SendErrorForResponses(responses, request_count, err);
  }
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; ++i) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to enflame backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests
  if ((total_batch_size != 1) and
      (total_batch_size > static_cast<size_t>(max_batch_size))) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response pointer will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error,  we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // BackendInputCollector collector(
  //     requests, request_count, &responses,
  //     StateForModel()->TritonMemoryManager(),
  //     StateForModel()->EnablePinnedInput(), CudaStream());

  // SetInputTensors(
  //     &collector, total_batch_size, requests, request_count, &responses);

  SetInputTensors(total_batch_size, requests, request_count, &responses);
  //
  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  // TODO: understand here
  std::vector<std::string> required_outputs;
  std::vector<std::vector<std::string>> request_required_outputs(request_count);
  for (size_t idx = 0; idx < request_count; ++idx) {
    const auto& request = requests[idx];
    auto& response = responses[idx];
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));

          if (response != nullptr) {
            required_outputs.push_back(output_name);
            request_required_outputs[idx].push_back(output_name);
          }
        }
      }
    }
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  TRITONENFLAME_ModelRun(triton_enflame_model_.get());

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  ReadOutputTensors(
      total_batch_size, required_outputs, requests, request_count, &responses);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send enflame backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // TODO: Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
       " released " + std::to_string(request_count) + " requests")
          .c_str());
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // TopsInference::topsInference_init();
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  // TopsInference::topsInference_finish();
  // release state
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
}}}  // namespace triton::backend::enflame
