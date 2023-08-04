// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#include <algorithm>
#include <memory>
#include <numeric>

#include "huawei_backend_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

struct TRITONHUAWEI_Tensor;

// HUAWEI Predictor Wrapper
struct TRITONHUAWEI_Model;

typedef triton::common::TritonJson::Value JsonValue;

class ModelImpl {
 public:
  ModelImpl(const char* model_path, JsonValue* model_config, int instance_id);
  ~ModelImpl();

  TRITONHUAWEI_Error* Run();

  TRITONHUAWEI_Error* GetInputPtr(
      const char* name, const TRITONHUAWEI_DataType dtype,
      const TRITONHUAWEI_Shape& shape, char** ptr);

  TRITONHUAWEI_Error* GetOutputMetadata(
      const char* name, TRITONHUAWEI_DataType* dtype,
      TRITONHUAWEI_Shape* shape, char** ptr);

  // TRITONHUAWEI_Error* ZeroCopyRun();

 private:
  int32_t device_id_ = 0;
  std::string acl_config_path_;
  aclrtContext context_ = nullptr;
  aclrtContext cur_context_ = nullptr;
  aclrtStream stream_;
  aclrtRunMode run_mode_;
  aclmdlDesc *model_desc_;
  uint32_t model_id_;
  size_t model_worksize_ = 0;
  size_t model_weightsize_ = 0;
  void *model_workptr_;
  void *model_weightptr_;
  aclmdlDataset *device_inputs_;
  aclmdlDataset *device_outputs_;
  uint32_t num_inputs_;
  uint32_t num_outputs_;
  std::vector<void*> host_inputs_;
  std::vector<void*> host_outputs_;

  std::vector<TRITONHUAWEI_Shape> inputs_shape_;
  std::vector<TRITONHUAWEI_Shape> outputs_shape_;

  std::vector<TRITONHUAWEI_DataType> inputs_type_;
  std::vector<TRITONHUAWEI_DataType> outputs_type_;

  std::unordered_map<std::string, int> input_name_map_;
  std::unordered_map<std::string, int> output_name_map_;
};


ModelImpl::ModelImpl(
    const char* model_path, JsonValue* model_config, int instance_id)
{
  // parse model config
  TRITONSERVER_Error* err = nullptr;
  do {
    JsonValue params, json_value;
    model_config->Find("parameters", &params);
    if (params.Find("device_id", &json_value)) {
      std::string string_value;
      CHECK_ERROR_WITH_BREAK(
          err = json_value.MemberAsString("string_value", &string_value));
      device_id_ = atoi(string_value.c_str());
    }
    if (params.Find("acl_config_path", &json_value)) {
      CHECK_ERROR_WITH_BREAK(
          err = json_value.MemberAsString("string_value", &acl_config_path_));
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
  
  std::cout<<"acl_config_path_:"<<acl_config_path_<<std::endl;
  // aclError ret = aclInit(acl_config_path_.c_str());
  // if (ret != ACL_SUCCESS) {
  //   std::cout<<"aclInit failed!"<<std::endl;
  // }
  aclError ret = aclrtSetDevice(device_id_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtSetDevice failed!"<<std::endl;
  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtCreateContext failed!"<<std::endl;
  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtCreateStream failed!"<<std::endl;
  ret = aclrtGetRunMode(&run_mode_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtGetRunMode failed!"<<std::endl;
  ret = aclmdlQuerySize(model_path, &model_worksize_, &model_weightsize_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclmdlQuerySize failed!"<<std::endl;
  ret = aclrtMalloc(&model_workptr_, model_worksize_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtMalloc model_workptr_!"<<std::endl;
  ret = aclrtMalloc(&model_weightptr_, model_weightsize_, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclrtMalloc model_weightptr_ failed!"<<std::endl;
  ret = aclmdlLoadFromFileWithMem(model_path, &model_id_, model_workptr_,
                                  model_worksize_, model_weightptr_, model_weightsize_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclmdlLoadFromFileWithMem failed!"<<std::endl;
  model_desc_ = aclmdlCreateDesc();
  ret = aclmdlGetDesc(model_desc_, model_id_);
  if (ret != ACL_SUCCESS)
    std::cout<<"aclmdlGetDesc failed!"<<std::endl;
  num_inputs_ = aclmdlGetNumInputs(model_desc_);
  num_outputs_ = aclmdlGetNumOutputs(model_desc_);

  inputs_shape_.resize(num_inputs_);
  outputs_shape_.resize(num_outputs_);
  host_inputs_.resize(num_inputs_);
  host_outputs_.resize(num_outputs_);
  device_inputs_ = aclmdlCreateDataset();
  device_outputs_ = aclmdlCreateDataset();
  for(size_t i=0; i<num_inputs_; i++){
    size_t data_len = aclmdlGetInputSizeByIndex(model_desc_, i);
    void *data = nullptr;
    ret = aclrtMalloc(&data, data_len, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMalloc input device failed!"<<std::endl;
    aclDataBuffer *data_buf = aclCreateDataBuffer(data, data_len);
    ret = aclmdlAddDatasetBuffer(device_inputs_, data_buf);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclmdlAddDatasetBuffer input device failed!"<<std::endl;
    ret = aclrtMallocHost(&host_inputs_[i], data_len);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMallocHost input host failed!"<<std::endl;
  }
  for(size_t i=0; i<num_outputs_; i++){
    size_t data_len = aclmdlGetOutputSizeByIndex(model_desc_, i);
    void *data = nullptr;
    ret = aclrtMalloc(&data, data_len, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMalloc output device failed!"<<std::endl;
    aclDataBuffer *data_buf = aclCreateDataBuffer(data, data_len);
    ret = aclmdlAddDatasetBuffer(device_outputs_, data_buf);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMalloc output device failed!"<<std::endl;
    ret = aclrtMallocHost(&host_outputs_[i], data_len);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMalloc output device failed!"<<std::endl;
  }
}

ModelImpl::~ModelImpl()
{
  // // Release the resource
  for (size_t i = 0; i < num_inputs_; ++i) {
    (void)aclrtFreeHost(host_inputs_[i]);
    aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(device_inputs_, i);
    void *data = aclGetDataBufferAddr(data_buffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(data_buffer);
  }
  (void)aclmdlDestroyDataset(device_inputs_);
  device_inputs_ = nullptr;

  for (size_t i = 0; i < num_outputs_; ++i) {
    (void)aclrtFreeHost(host_outputs_[i]);
    aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(device_outputs_, i);
    void *data = aclGetDataBufferAddr(data_buffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(data_buffer);
  }
  (void)aclmdlDestroyDataset(device_outputs_);
  device_outputs_ = nullptr;

  if (model_desc_ != nullptr) {
    (void)aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }

  if (model_workptr_ != nullptr) {
    (void)aclrtFree(model_workptr_);
    model_workptr_ = nullptr;
    model_worksize_ = 0;
  }

  if (model_weightptr_ != nullptr) {
    (void)aclrtFree(model_weightptr_);
    model_weightptr_ = nullptr;
    model_weightsize_ = 0;
  }
  model_id_ = 0;

  if (stream_ != nullptr) {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
  }

  if (context_ != nullptr) {
    aclrtDestroyContext(context_);
    context_ = nullptr;
  }

  aclrtResetDevice(device_id_);
  // aclFinalize();
}

TRITONHUAWEI_Error*
ModelImpl::Run()
{
  if(cur_context_ == nullptr){
    cur_context_ = context_;
    aclError ret = aclrtSetCurrentContext(cur_context_);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtSetCurrentContext failed! "<<ret<<std::endl;
  }
  
  for(size_t i=0; i<num_inputs_; i++){
    std::vector<int64_t> dims = inputs_shape_[i].Shape();
    aclmdlIODims current_dims;
    // std::cout<<"current_dims:";
    current_dims.dimCount = dims.size();
    for(size_t k=0; k<dims.size(); k++){
      current_dims.dims[k] = (int)dims[k];
      // std::cout<<current_dims.dims[k]<<",";
    }
    // std::cout<<std::endl;
    aclmdlSetInputDynamicDims(model_id_, device_inputs_, i, &current_dims);
  }

  for (size_t i = 0; i < num_inputs_; i++) {
    aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(device_inputs_, i);
    void *data = aclGetDataBufferAddr(data_buffer);
    uint32_t data_len = aclGetDataBufferSizeV2(data_buffer);
    aclError ret = aclrtMemcpy(data, data_len, host_inputs_[i], data_len, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMemcpy ACL_MEMCPY_HOST_TO_DEVICE failed! "<<ret<<std::endl;
  }

  // aclmdlExecute(model_id_, device_inputs_, device_outputs_);
  aclError ret =aclmdlExecuteAsync(model_id_, device_inputs_, device_outputs_, stream_);
  if (ret != ACL_SUCCESS)
      std::cout<<"aclmdlExecuteAsync failed! "<<ret<<std::endl;
  ret =aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS)
      std::cout<<"aclrtSynchronizeStream failed! " <<ret<<std::endl;

  for (size_t i = 0; i < num_outputs_; i++) {
    aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(device_outputs_, i);
    void *data = aclGetDataBufferAddr(data_buffer);
    uint32_t data_len = aclGetDataBufferSizeV2(data_buffer);
    aclError ret =aclrtMemcpy(host_outputs_[i], data_len, data, data_len, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
      std::cout<<"aclrtMemcpy ACL_MEMCPY_DEVICE_TO_HOST failed! "<<ret<<std::endl;

    aclmdlIODims current_dims;
    aclmdlGetCurOutputDims(model_desc_, i, &current_dims);
    std::vector<int64_t> dims(current_dims.dimCount);
    for(size_t k=0; k<current_dims.dimCount; k++){
      dims[k] = (int64_t)current_dims.dims[k];
    }
    outputs_shape_[i] = TRITONHUAWEI_Shape(dims);
  }

  return nullptr;
}

TRITONHUAWEI_Error*
ModelImpl::GetInputPtr(
    const char* name, const TRITONHUAWEI_DataType dtype,
    const TRITONHUAWEI_Shape& shape, char** ptr)
{
  int index = input_name_map_[name];

  if (inputs_type_[index] != dtype) {
    return TRITONHUAWEI_ErrorNew("inputs types is not matched");
  }

  inputs_shape_[index] = TRITONHUAWEI_Shape(shape.Shape());
  // todo: get the real tensor
  *ptr = reinterpret_cast<char*>(host_inputs_[index]);

  return nullptr;
}

TRITONHUAWEI_Error*
ModelImpl::GetOutputMetadata(
    const char* name, TRITONHUAWEI_DataType* dtype, TRITONHUAWEI_Shape* shape,
    char** ptr)
{
  int index = output_name_map_[name];
  *dtype = outputs_type_[index];
  *shape = outputs_shape_[index];
  *ptr = reinterpret_cast<char*>(host_outputs_[index]);
  return nullptr;
}

TRITONSERVER_Error*
TRITONHUAWEI_ModelCreate(
    TRITONHUAWEI_Model** model, const char* model_path, JsonValue* config,
    int instance_id)
{
  try {
    ModelImpl* model_impl = new ModelImpl(model_path, config, instance_id);
    *model = reinterpret_cast<TRITONHUAWEI_Model*>(model_impl);
  }
  catch (const TRITONHUAWEI_Exception& ex) {
    RETURN_IF_TRITONHUAWEI_ERROR(ex.err_);
  }
  return nullptr;
}

void

TRITONHUAWEI_ModelDelete(TRITONHUAWEI_Model* model)
{
  if (model != nullptr) {
    ModelImpl* mi = reinterpret_cast<ModelImpl*>(model);
    delete mi;
  }
}

TRITONHUAWEI_Error*
TRITONHUAWEI_ModelRun(TRITONHUAWEI_Model* model)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  return m->Run();
}

class TensorImpl {
 public:
  TensorImpl(
      const char* name, TRITONHUAWEI_DataType dtype,
      const TRITONHUAWEI_Shape& shape, char* data_ptr);
  ~TensorImpl() = default;

  const std::string& Name() const { return name_; }
  TRITONHUAWEI_DataType DataType() const { return dtype_; }
  TRITONHUAWEI_Shape Shape() const { return shape_; }

  char* Base() const { return base_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  const std::string name_;
  const TRITONHUAWEI_DataType dtype_;
  const TRITONHUAWEI_Shape shape_;

  char* base_;
  size_t byte_size_;
};

TensorImpl::TensorImpl(
    const char* name, TRITONHUAWEI_DataType dtype,
    const TRITONHUAWEI_Shape& shape, char* data_ptr)
    : name_(name), dtype_(dtype), shape_(shape), base_(data_ptr)
{
  byte_size_ = shape.NumElements() * TRITONHUAWEI_DataTypeByteSize(dtype);
}

TRITONHUAWEI_Tensor*
TRITONHUAWEI_TensorNew(
    TRITONHUAWEI_Model* model, const char* name, TRITONHUAWEI_DataType dtype,
    const TRITONHUAWEI_Shape& shape)
{
  char* data_ptr;
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetInputPtr(name, dtype, shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONHUAWEI_Tensor*>(tensor);
}

TRITONHUAWEI_Tensor*
TRITONHUAWEI_TensorNew(TRITONHUAWEI_Model* model, const char* name)
{
  char* data_ptr;
  TRITONHUAWEI_DataType dtype;
  TRITONHUAWEI_Shape shape;

  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  auto err = m->GetOutputMetadata(name, &dtype, &shape, &data_ptr);
  if (err != nullptr) {
    return nullptr;
  }

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, data_ptr);
  return reinterpret_cast<TRITONHUAWEI_Tensor*>(tensor);
}

char*
TRITONHUAWEI_TensorData(TRITONHUAWEI_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Base();
}

size_t
TRITONHUAWEI_TensorDataByteSize(TRITONHUAWEI_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->ByteSize();
}

TRITONHUAWEI_DataType
TRITONHUAWEI_TensorDataType(TRITONHUAWEI_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->DataType();
}

TRITONHUAWEI_Shape
TRITONHUAWEI_TensorShape(TRITONHUAWEI_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Shape();
}

namespace triton { namespace backend { namespace HUAWEI {

using TRITONHUAWEIModelHandle = std::shared_ptr<TRITONHUAWEI_Model>;

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
       Name() + "': not supported for HUAWEI backend")
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
            TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INVALID,
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
            TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INVALID,
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
      BackendInputCollector* collector,
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<std::string>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;
  TRITONHUAWEIModelHandle triton_HUAWEI_model_;
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
  *model_path = JoinPath({model_dir, "model.om"});
  RETURN_IF_ERROR(FileExists(*model_path, &exists));
  if (not exists) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("HUAWEI model should be named as 'model.om'").c_str());
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
  TRITONHUAWEI_Model* triton_HUAWEI_model = nullptr;
  THROW_IF_BACKEND_INSTANCE_ERROR(TRITONHUAWEI_ModelCreate(
      &triton_HUAWEI_model, model_path.c_str(), &model_config, instance_id));
  triton_HUAWEI_model_.reset(triton_HUAWEI_model, TRITONHUAWEI_ModelDelete);
}

void
ModelInstanceState::SetInputTensors(
    BackendInputCollector* collector, 
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  bool cuda_copy = false;
  // BackendInputCollector collector(
  //     requests, request_count, responses,
  //     StateForModel()->TritonMemoryManager(),
  //     StateForModel()->EnablePinnedInput(), CudaStream());

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

    TRITONHUAWEI_Tensor* tensor = TRITONHUAWEI_TensorNew(
        triton_HUAWEI_model_.get(), name, ConvertDataType(datatype),
        TRITONHUAWEI_Shape(batchn_shape));

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

    collector->ProcessTensor(
        name, TRITONHUAWEI_TensorData(tensor),
        TRITONHUAWEI_TensorDataByteSize(tensor), TRITONSERVER_MEMORY_CPU, 0);
  }

  cuda_copy |= collector->Finalize();
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

    TRITONHUAWEI_Tensor* tensor =
        TRITONHUAWEI_TensorNew(triton_HUAWEI_model_.get(), name.c_str());

    if (tensor == nullptr) {
      auto err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("Failed to create output tensor '") + name + " for '" +
           Name() + "'")
              .c_str());
      SendErrorForResponses(responses, request_count, err);
      return;
    }

    auto dtype = ConvertDataType(TRITONHUAWEI_TensorDataType(tensor));
    auto shape = TRITONHUAWEI_TensorShape(tensor).Shape();
    responder.ProcessTensor(
        name, dtype, shape, TRITONHUAWEI_TensorData(tensor),
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
                  "null request given to HUAWEI backend for '" + Name() + "'")
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

  BackendInputCollector collector(
      requests, request_count, &responses,
      StateForModel()->TritonMemoryManager(),
      StateForModel()->EnablePinnedInput(), CudaStream());

  SetInputTensors(
      &collector, total_batch_size, requests, request_count, &responses);

  // SetInputTensors(total_batch_size, requests, request_count, &responses);
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

  TRITONHUAWEI_ModelRun(triton_HUAWEI_model_.get());

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
          "failed to send HUAWEI backend response");
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

  //std::string acl_config_path = "/home/liuqingjie/model_repo/ocr_lite_huawei/resource/acl.json";
  const char* location;
  TRITONBACKEND_ArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  std::string acl_config_path = std::string(location) + "/acl.json";
  aclError ret = aclInit(acl_config_path.c_str());
  if (ret != ACL_SUCCESS) {
    std::cout<<"aclInit failed!"<<std::endl;
  }else{
    std::cout<<"aclInit success!"<<std::endl;
  }
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  aclFinalize();
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
}}}  // namespace triton::backend::HUAWEI
