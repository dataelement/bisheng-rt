// Copyright 2022, DataElem, Inc. All rights reserved.

#include "dataelem/alg/distribution.h"
#include "dataelem/framework/alg_factory.h"
#include "dataelem/framework/alg_utils.h"
#include "dataelem/framework/app_factory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

//
// Algorithmer backend is developed by DataElem, Inc., the backend used for
// doing the algorithm pipelines, integrate prprocess and postprocess step
//

namespace dataelem { namespace alg {


//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public triton::backend::BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model, true /* allow_optional */)
  {
  }
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const triton::backend::BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public triton::backend::BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  TRITONSERVER_Error* UpdateInputsOutputsInfo();

  TRITONSERVER_Error* ValidateIOs(AlgRunContext* context, bool is_input);

  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

 private:
  ModelState* model_state_;
  std::string op_path_;
  std::shared_ptr<IOp> op_;
  std::shared_ptr<IApp> app_;
  bool app_mode_ = false;

  int device_;

  std::set<std::string> optional_inputs_;
  StringList input_names_;
  std::unordered_map<std::string, TRITONSERVER_DataType> input_dtype_map_;
  StringList output_names_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  auto& model_config = model_state->ModelConfig();
  triton::common::TritonJson::Value params;
  bool status = model_config.Find("parameters", &params);

  if (status == false) {
    THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Model config must contains parameters, model:" + model_state_->Name())
            .c_str()));
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(UpdateInputsOutputsInfo());

  std::string algorithm_type;
  THROW_IF_BACKEND_MODEL_ERROR(triton::backend::TryParseModelStringParameter(
      params, "algorithm_type", &algorithm_type, ""));
  if (!algorithm_type.empty()) {
    // Create Algorithmer instance and do initialize
    op_ = std::shared_ptr<IOp>(AlgRegistry::CreateAlg(algorithm_type));
    if (op_.get() == nullptr) {
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Algorithm not found:" + algorithm_type).c_str()));
    }

    op_->init(model_state);
    if (algorithm_type.compare("Distribution") == 0) {
      const char* cname;
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &cname);
      std::string name(cname);
      int instance_id = atoi(name.substr(name.size() - 1, 1).c_str());
      reinterpret_cast<Distribution*>(op_.get())->set_instance_id(instance_id);
    }
  }

  // Create Application instance and do initialize
  std::string app_type;
  THROW_IF_BACKEND_MODEL_ERROR(triton::backend::TryParseModelStringParameter(
      params, "app_type", &app_type, ""));

  if (!app_type.empty()) {
    // Create Algorithmer instance and do initialize
    app_ = std::shared_ptr<IApp>(AppRegistry::CreateApp(app_type));
    if (app_.get() == nullptr) {
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("App not found:" + app_type).c_str()));
    }

    app_->init(model_state);
    app_mode_ = true;
  }

  if (app_type.empty() && algorithm_type.empty()) {
    THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "one of app_type and algorithm_type must be set"));
  }
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const triton::backend::BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      triton::backend::RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to BLS backend for '" + Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
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

  // ModelState* model_state = reinterpret_cast<ModelState*>(Model());

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  for (size_t r = 0; r < request_count; r++) {
    auto context = std::make_unique<AlgRunContext>();

    context->SetMemoryManager(model_state_->TritonMemoryManager());
    context->UpdateCudaStream(HostPolicyName().c_str(), CudaStream());

    TRITONSERVER_Error* err = nullptr;
    do {
      BackendRequestInfo req_info;
      UpdateBackendRequestInfo(requests[r], req_info);
      context->SetBackendRequestInfo(req_info);

      if (!app_mode_) {
        // run algorithm
        CHECK_ERROR_WITH_BREAK(
            err = ParseTensorsFromBackendRequest(requests[r], context.get()));

        CHECK_ERROR_WITH_BREAK(err = ValidateIOs(context.get(), true));

        CHECK_ERROR_WITH_BREAK(err = op_->Execute(context.get()));

        CHECK_ERROR_WITH_BREAK(err = ValidateIOs(context.get(), false));

        CHECK_ERROR_WITH_BREAK(
            err = ConstructFinalResponse(
                &responses[r], context.get(), output_names_));
      } else {
        CHECK_ERROR_WITH_BREAK(
            err = ParseTensorsFromBackendRequest(requests[r], context.get()));
        CHECK_ERROR_WITH_BREAK(err = ValidateIOs(context.get(), true));
        std::string resp;
        CHECK_ERROR_WITH_BREAK(err = app_->Execute(context.get(), &resp));
        // CHECK_ERROR_WITH_BREAK(err = ValidateIOs(context.get(), false));
        CHECK_ERROR_WITH_BREAK(
            err =
                ConstructFinalResponse(&responses[r], resp, output_names_[0]));
      }

    } while (false);

    RESPOND_AND_SET_NULL_IF_ERROR(&responses[r], err);
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

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
          "failed to send BLS backend response");
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

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), 1 /*total_batch_size*/, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
       " released " + std::to_string(request_count) + " requests")
          .c_str());
}


TRITONSERVER_Error*
ModelInstanceState::UpdateInputsOutputsInfo()
{
  {
    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
    if (ios.ArraySize() == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "model configuration must contain at least one input, none were "
          "specified.");
    }

    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      input_names_.emplace_back(io_name);

      std::string io_dtype;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
      input_dtype_map_[io_name] = ModelConfigDataTypeToServerType(io_dtype);
      if (input_dtype_map_[io_name] == TRITONSERVER_TYPE_INVALID) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("unsupported datatype " + io_dtype + " for input '" + io_name +
             "' for model '" + model_state_->Name() + "'")
                .c_str());
      }
      // parse optional input
      if (io.Find("optional")) {
        bool is_optional = false;
        RETURN_IF_ERROR(io.MemberAsBool("optional", &is_optional));
        if (is_optional) {
          optional_inputs_.insert(io_name);
        }
      }
    }
  }
  {
    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));

    if (ios.ArraySize() == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "model configuration must contain at least one output, none were "
          "specified.");
    }

    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      output_names_.emplace_back(io_name);

      std::string io_dtype;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
      output_dtype_map_[io_name] = ModelConfigDataTypeToServerType(io_dtype);
      if (output_dtype_map_[io_name] == TRITONSERVER_TYPE_INVALID) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("unsupported datatype " + io_dtype + " for output '" + io_name +
             "' for model '" + model_state_->Name() + "'")
                .c_str());
      }
    }
  }
  return nullptr;  // success
}


TRITONSERVER_Error*
ModelInstanceState::ValidateIOs(AlgRunContext* context, bool is_input)
{
  std::unordered_map<std::string, TRITONSERVER_DataType>* type_map_ptr =
      (is_input ? &input_dtype_map_ : &output_dtype_map_);
  StringList* names_ptr = is_input ? &input_names_ : &output_names_;

  OCTensor* tensor;
  for (const auto& name : *names_ptr) {
    if (is_input && (optional_inputs_.find(name) != optional_inputs_.end())) {
      continue;
    }

    RETURN_ERROR_IF_FALSE(
        context->GetTensor(name, &tensor), TRITONSERVER_ERROR_INTERNAL,
        ("io tensor [" + name + "] for algorithm " + model_state_->Name() +
         " not exists"));

    RETURN_ERROR_IF_FALSE(
        type_map_ptr->at(name) == ConvertDataType(tensor->dtype()),
        TRITONSERVER_ERROR_INTERNAL,
        ("type for tensor [" + name + "] is wrong for algorithm " +
         model_state_->Name()));
  }
  return nullptr;
}

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
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

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
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

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
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

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
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
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
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

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
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
  ModelState* model_state =
      reinterpret_cast<ModelState*>(instance_state->Model());

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}  // namespace dataelem::alg
