#include "dataelem/framework/alg_utils.h"

namespace dataelem { namespace alg {


TRITONSERVER_Error*
UpdateBackendRequestInfo(
    TRITONBACKEND_Request* request, BackendRequestInfo& info)
{
  RETURN_IF_ERROR(TRITONBACKEND_RequestId(request, &info.request_id));
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestCorrelationId(request, &info.correlation_id));
  RETURN_IF_ERROR(TRITONBACKEND_RequestFlags(request, &info.flags));
  return nullptr;
}

// This function will return a tensor's contents as a contiguous
// chunk in system memory. In some cases this will require copying the data.
// If that  happens, 'contiguous_buffer' will be set to hold the contiguous
// chunk and 'cuda_copy' will be set to indicate whether CUDA copy is
// conducted.  The data copy can be avoided if the input is already in
// a contiguous chunk and the input is located in memory type and id
// specified.
TRITONSERVER_Error*
GetContiguousInputContent(
    TRITONBACKEND_Input* rinput, const char* host_policy_name,
    const uint32_t buffer_count, const char** content,
    size_t* content_byte_size, char** contiguous_buffer, cudaStream_t stream,
    AlgRunContext* context, bool* cuda_copy)
{
  *cuda_copy = false;
  *contiguous_buffer = nullptr;

  // Check input buffers to see if data copy is necessary
  size_t chunk_count = 0;
  bool type_mismatch = false;
  uint64_t total_byte_size = 0;
  for (size_t idx = 0; idx < buffer_count; ++idx) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    size_t src_byte_size;
    const void* src_ptr;

    RETURN_IF_ERROR(TRITONBACKEND_InputBufferForHostPolicy(
        rinput, host_policy_name, idx, &src_ptr, &src_byte_size,
        &src_memory_type, &src_memory_type_id));

    if (src_ptr != nullptr) {
      chunk_count++;
      total_byte_size += src_byte_size;
      type_mismatch |= (src_memory_type == TRITONSERVER_MEMORY_GPU);
    }
  }

  if (chunk_count == 0) {
    *content = nullptr;
    *content_byte_size = 0;
  } else if ((chunk_count == 1) && !type_mismatch) {
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBufferForHostPolicy(
        rinput, host_policy_name, 0, (const void**)content, content_byte_size,
        &src_memory_type, &src_memory_type_id));
  } else {
    auto alloc_type =
        (type_mismatch
             ?
#ifdef TRITON_ENABLE_GPU
             triton::backend::BackendMemory::AllocationType::CPU_PINNED_POOL
#else
             triton::backend::BackendMemory::AllocationType::CPU
#endif  // TRITON_ENABLE_GPU
             : triton::backend::BackendMemory::AllocationType::CPU);

    triton::backend::BackendMemory* input_memory;
    RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
        context->GetMemoryManager(), {alloc_type}, 0 /* memory_type_id */,
        total_byte_size, &input_memory));
    context->AddBackendMemory(input_memory);

    // const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
    *contiguous_buffer = input_memory->MemoryPtr();

    size_t offset = 0;
    for (size_t i = 0; i < chunk_count; i++) {
      bool cuda_used;
      TRITONSERVER_MemoryType src_memory_type;
      int64_t src_memory_type_id;
      size_t src_byte_size;
      const void* src_ptr;

      RETURN_IF_ERROR(TRITONBACKEND_InputBufferForHostPolicy(
          rinput, host_policy_name, i, &src_ptr, &src_byte_size,
          &src_memory_type, &src_memory_type_id));

      // H2H or D2H, for D2H, host memory must in pinned memory
      RETURN_IF_ERROR(triton::backend::CopyBuffer(
          "Contiguous input", src_memory_type, src_memory_type_id,
          TRITONSERVER_MEMORY_CPU, 0, src_byte_size, src_ptr,
          *contiguous_buffer + offset, stream, &cuda_used));
      *cuda_copy |= cuda_used;
      offset += src_byte_size;
    }

    *content = *contiguous_buffer;
    *content_byte_size = total_byte_size;
  }

  return nullptr;  // success
}

// Parse tensors from inference requests, put tensors in system memory
TRITONSERVER_Error*
ParseTensorsFromBackendRequest(
    TRITONBACKEND_Request* request, AlgRunContext* context)
{
  // max_batch_size must equal zero
  // const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));

  bool cuda_copy = false;
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(request, input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    // Parse tensor from continuous buffer
    uint32_t buffer_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
        input, context->HostPolicyName(), nullptr, nullptr, nullptr, nullptr,
        nullptr, &buffer_count));

    // RESPOND_AND_SET_NULL_IF_ERROR(
    //     &response, TRITONBACKEND_InputPropertiesForHostPolicy(
    //                    input, context->HostPolicyName(), nullptr, nullptr,
    //                    nullptr, nullptr, nullptr, &buffer_count));

    bool cuda_used = false;
    const char* content = nullptr;
    size_t content_byte_size = 0;
    char* contiguous_buffer = nullptr;
    auto err = GetContiguousInputContent(
        input, context->HostPolicyName(), buffer_count, &content,
        &content_byte_size, &contiguous_buffer, context->CudaStream(), context,
        &cuda_used);
    RETURN_IF_ERROR(err);

    cuda_copy |= cuda_used;
    // RESPOND_AND_SET_NULL_IF_ERROR(&response, err);

    auto tensor = OCTensor::from_blob(
        input_dims_count, input_shape, input_datatype, content,
        content_byte_size);
    // RETURN_ERROR_IF_TRUE(
    //    tensor.empty(), TRITONSERVER_ERROR_INTERNAL,
    //    std::string("Failed to create tensor name:") +
    //    std::string(input_name));

    context->SetTensor(std::string(input_name), std::move(tensor));
  }

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(context->CudaStream());
    cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  return nullptr;
}

TRITONSERVER_Error*
CreateServerRequestWithTensors(
    BackendRequestInfo* request_info, TRITONSERVER_Server* server,
    const char* graph_name, const OCTensorList* input_tensors,
    const StringList& input_names, const StringList& output_names,
    TRITONSERVER_InferenceRequest** irequest)
{
  // Create an inference request object. The inference request object
  // is where we set the name of the model we want to use for
  // inference and the input tensors.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
      irequest, server, graph_name, -1 /* model_version */));

  // Set request_id, correlation_id, and flags for the new request.
  RETURN_IF_ERROR(
      TRITONSERVER_InferenceRequestSetId(*irequest, request_info->request_id));

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetCorrelationId(
      *irequest, request_info->correlation_id));

  RETURN_IF_ERROR(
      TRITONSERVER_InferenceRequestSetFlags(*irequest, request_info->flags));

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      *irequest, GraphInferRequestComplete,
      nullptr /* request_release_userp */));

  uint32_t input_count = input_tensors->size();
  const char* name;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint32_t dims_count;
  size_t data_byte_size;
  TRITONSERVER_MemoryType data_memory_type;


  int64_t data_memory_id = 0;
  const char* data_buffer;

  for (size_t idx = 0; idx < input_count; idx++) {
    auto tensor = &input_tensors->at(idx);

    data_memory_type =
        (tensor->pinned() ? TRITONSERVER_MEMORY_CPU_PINNED
                          : TRITONSERVER_MEMORY_CPU);

    name = input_names[idx].c_str();
    datatype = ConvertDataType(tensor->dtype());
    shape = tensor->shape_ptr();
    dims_count = tensor->shape().size();
    data_buffer = tensor->data_ptr();
    data_byte_size = tensor->ByteSize();
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        *irequest, name, datatype, shape, dims_count));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        *irequest, name, data_buffer, data_byte_size, data_memory_type,
        data_memory_id));
  }

  uint32_t output_count = output_names.size();
  const char* output_name;
  for (size_t count = 0; count < output_count; count++) {
    output_name = output_names[count].c_str();
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        *irequest, output_name));
  }

  return nullptr;  // success
}


// Parse Tensors from Response, ParseTensorsFromResponse

TRITONSERVER_Error*
ParseTensorsFromServerResponse(
    TRITONSERVER_InferenceResponse* response, const StringList& output_names,
    OCTensorList* tensors)
{
  const char* output_name;
  TRITONSERVER_DataType output_datatype;
  const int64_t* output_shape;
  uint64_t dims_count;
  size_t output_byte_size;
  TRITONSERVER_MemoryType output_memory_type;
  int64_t output_memory_id;
  const void* output_base;
  void* userp;
  uint32_t count;

  RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutputCount(response, &count));
  std::set<std::string> names(output_names.begin(), output_names.end());
  std::unordered_map<std::string, OCTensor> values;
  for (uint32_t idx = 0; idx < count; idx++) {
    RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutput(
        response, idx, &output_name, &output_datatype, &output_shape,
        &dims_count, &output_base, &output_byte_size, &output_memory_type,
        &output_memory_id, &userp));

    if (names.find(output_name) == names.end()) {
      continue;
    }

    RETURN_ERROR_IF_TRUE(
        (output_byte_size > 0 && output_base == nullptr),
        TRITONSERVER_ERROR_INTERNAL,
        std::string("tensor is null when size > 0:") +
            std::string(output_name));

    // Resposne buffer in all continuous manner
    auto tensor = OCTensor::from_blob(
        (uint32_t)dims_count, output_shape, output_datatype,
        reinterpret_cast<const char*>(output_base), output_byte_size);

    values.emplace(output_name, std::move(tensor));
    // RETURN_ERROR_IF_TRUE(
    //     tensor.empty(), TRITONSERVER_ERROR_INTERNAL,
    //     std::string("Failed to create tensor name:") +
    //         std::string(output_name));

    // tensors->emplace_back(std::move(tensor));
  }

  for (const auto& name : output_names) {
    tensors->emplace_back(std::move(values[name]));
  }

  return nullptr;
}


TRITONSERVER_Error*
ConstructFinalResponse(
    TRITONBACKEND_Response** response, AlgRunContext* context,
    const StringList& names)
{
  const char* output_name;
  TRITONSERVER_DataType output_datatype;
  const int64_t* output_shape;
  uint64_t dims_count;
  size_t output_byte_size;
  TRITONSERVER_MemoryType output_memory_type;
  int64_t output_memory_id;
  const void* output_base;
  OCTensor* tensor = nullptr;

  TRITONSERVER_MemoryType data_memory_type;

  bool cuda_copy = false;
  auto stream = context->CudaStream();

  std::vector<absl::string_view> b64_outputs;
  for (size_t i = 0; i < names.size(); i++) {
    RETURN_ERROR_IF_FALSE(
        context->GetTensor(names[i], &tensor), TRITONSERVER_ERROR_INTERNAL,
        std::string("Failed to get output tensor name:") + names[i]);

    output_name = names[i].c_str();

    output_datatype = ConvertDataType(tensor->dtype());
    // output_shape = tensor->shape().data();
    output_shape = tensor->shape_ptr();
    dims_count = tensor->shape().size();
    output_base = tensor->data_ptr();
    output_byte_size = tensor->ByteSize();

    // Create an output tensor in the final response with
    // the information retrieved above.
    TRITONBACKEND_Output* output;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        *response, &output, output_name, output_datatype, output_shape,
        dims_count));

    if (tensor->base64()) {
      b64_outputs.emplace_back(names[i]);
    }

    // RESPOND_AND_SET_NULL_IF_ERROR(
    //     response, TRITONBACKEND_ResponseOutput(
    //                   *response, &output, output_name, output_datatype,
    //                   output_shape, dims_count));

    // Get a buffer that holds the tensor data for the output.
    // We request a buffer in CPU memory but we have to handle any returned
    // type. If we get back a buffer in GPU memory we just fail the request.
    void* output_buffer;
    output_memory_type = TRITONSERVER_MEMORY_CPU;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
        output, &output_buffer, output_byte_size, &output_memory_type,
        &output_memory_id));

    data_memory_type =
        (tensor->pinned() ? TRITONSERVER_MEMORY_CPU_PINNED
                          : TRITONSERVER_MEMORY_CPU);

    // H2H or H2D, for H2D, memory in host must in pinned memory
    // Otherwise, must be H2H
    RETURN_ERROR_IF_TRUE(
        (output_memory_type == TRITONSERVER_MEMORY_GPU &&
         data_memory_type != TRITONSERVER_MEMORY_CPU_PINNED),
        TRITONSERVER_ERROR_INTERNAL,
        std::string("For H2D output, H must be pinned memory"));

    RETURN_IF_ERROR(triton::backend::CopyBuffer(
        "output buffer", data_memory_type, 0, output_memory_type,
        output_memory_id, output_byte_size, output_base, output_buffer, stream,
        &cuda_copy));

    // RESPOND_AND_SET_NULL_IF_ERROR(
    //     response, TRITONBACKEND_OutputBuffer(
    //                   output, &output_buffer, output_byte_size,
    //                   &output_memory_type, &output_memory_id));

    // if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
    //   RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
    //       TRITONSERVER_ERROR_INTERNAL,
    //       "failed to create output buffer in CPU memory"));

    //   // RESPOND_AND_SET_NULL_IF_ERROR(
    //   //     response, TRITONSERVER_ErrorNew(
    //   //                   TRITONSERVER_ERROR_INTERNAL,
    //   //                   "failed to create output buffer in CPU memory"));
    // }
    // memcpy(output_buffer, output_base, output_byte_size);
  }

  auto b64_outputs_str = absl::StrJoin(b64_outputs, " ");
  TRITONBACKEND_ResponseSetStringParameter(
      *response, "b64_outputs", b64_outputs_str.c_str());

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream);
  }
#endif  // TRITON_ENABLE_GPU

  return nullptr;
}


TRITONSERVER_Error*
ConstructFinalResponse(
    TRITONBACKEND_Response** response, const std::string& resp,
    const std::string& name)
{
  const char* output_name = name.c_str();
  TRITONSERVER_DataType output_datatype = TRITONSERVER_TYPE_BYTES;
  std::vector<int64_t> shape = {1};
  size_t output_byte_size = resp.length();
  TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t output_memory_id;
  const void* output_base = resp.data();

  // std::cout << "ConstructFinalResponse:" << resp << "," << output_byte_size
  // << "\n";

  TRITONBACKEND_Output* output;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      *response, &output, output_name, output_datatype, shape.data(),
      shape.size()));

  void* output_buffer;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      output, &output_buffer, output_byte_size, &output_memory_type,
      &output_memory_id));

  memcpy(output_buffer, output_base, output_byte_size);
  return nullptr;
}


//
// Graph Request/Response helper functions
//
TRITONSERVER_Error*
GraphInferResponseAllocator(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // For simplicity, this backend example always uses CPU memory regardless of
  // the preferred memory type. You can make the actual memory type and id that
  // we allocate be the same as preferred memory type. You can also provide a
  // customized allocator to support different preferred_memory_type, and reuse
  // memory buffer when possible.
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE, ("allocated " + std::to_string(byte_size) +
                                   " bytes for result tensor " + tensor_name)
                                      .c_str());
  } else {
    void* allocated_ptr = nullptr;
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    auto* context = reinterpret_cast<AlgRunContext*>(userp);

    triton::backend::BackendMemory* input_memory;
    RETURN_IF_ERROR(triton::backend::BackendMemory::Create(
        context->GetMemoryManager(),
#ifdef TRITON_ENABLE_GPU
        {triton::backend::BackendMemory::AllocationType::CPU_PINNED_POOL,
         triton::backend::BackendMemory::AllocationType::CPU},
#else
        {triton::backend::BackendMemory::AllocationType::CPU},
#endif  // TRITON_ENABLE_GPU
        0 /* memory_type_id */, byte_size, &input_memory));
    context->AddBackendMemory(input_memory);
    allocated_ptr = input_memory->MemoryPtr();

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = nullptr;
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          ("allocated " + std::to_string(byte_size) + " bytes in " +
           TRITONSERVER_MemoryTypeString(*actual_memory_type) +
           " for result tensor " + tensor_name)
              .c_str());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
GraphInferResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;  // Success
}

void
GraphInferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
GraphInferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // The following logic only works for non-decoupled models as for decoupled
  // models it may send multiple responses for a request or not send any
  // responses for a request. Need to modify this function if the model is using
  // decoupled API.
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

void
GraphInferResponseDelete(TRITONSERVER_InferenceResponse* response)
{
  if (response != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(response),
        "Failed to delete inference response.");
  }
}


GraphExecutor::GraphExecutor(TRITONSERVER_Server* server) : server_(server)
{
  // When triton needs a buffer to hold an output tensor, it will ask
  // us to provide the buffer. In this way we can have any buffer
  // management and sharing strategy that we want. To communicate to
  // triton the functions that we want it to call to perform the
  // allocations, we create a "response allocator" object. We pass
  // this response allocate object to triton when requesting
  // inference. We can reuse this response allocator object for any
  // number of inference requests.
  allocator_ = nullptr;
  // THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
  //     &allocator_, GraphInferResponseAllocator, GraphInferResponseRelease,
  //     nullptr /* start_fn */));
  THROW_IF_TRITON_ERROR(TRITONSERVER_ResponseAllocatorNew(
      &allocator_, GraphInferResponseAllocator, GraphInferResponseRelease,
      nullptr /* start_fn */));
}


GraphExecutor::~GraphExecutor()
{
  if (allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "Failed to delete allocator.");
  }
}


TRITONSERVER_Error*
GraphExecutor::AsyncExecute(
    TRITONSERVER_InferenceRequest* irequest, AlgRunContext* context,
    std::future<TRITONSERVER_InferenceResponse*>* future)
{
  // Perform inference by calling TRITONSERVER_ServerInferAsync. This
  // call is asychronous and therefore returns immediately. The
  // completion of the inference and delivery of the response is done
  // by triton by calling the "response complete" callback functions
  // (InferResponseComplete in this case).
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  *future = p->get_future();

  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
      irequest, allocator_,
      reinterpret_cast<void*>(context) /* response_allocator_userp */,
      GraphInferResponseComplete, reinterpret_cast<void*>(p)));

  RETURN_IF_ERROR(
      TRITONSERVER_ServerInferAsync(server_, irequest, nullptr /* trace */));

  return nullptr;  // success
}
////////////////////////////////////////////

}}  // namespace dataelem::alg
