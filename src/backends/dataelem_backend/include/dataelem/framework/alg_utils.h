#ifndef DATAELEM_FRAMEWORK_ALG_UTILS_H_
#define DATAELEM_FRAMEWORK_ALG_UTILS_H_

#include <future>

#include "dataelem/framework/alg_context.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

// #ifdef TRITON_ENABLE_GPU
// #include <cuda_runtime_api.h>
// #endif  // TRITON_ENABLE_GPU

namespace dataelem { namespace alg {

// // !TRITON_ENABLE_GPU
// #ifndef TRITON_ENABLE_GPU
// using cudaStream_t = void*;
// #endif

TRITONSERVER_Error* UpdateBackendRequestInfo(
    TRITONBACKEND_Request* request, BackendRequestInfo& info);


TRITONSERVER_Error* GetContiguousInputContent(
    TRITONBACKEND_Input* rinput, const char* host_policy_name,
    const uint32_t buffer_count, const char** content,
    size_t* content_byte_size, char** contiguous_buffer, cudaStream_t stream,
    AlgRunContext* context, bool* cuda_copy);

TRITONSERVER_Error* ParseTensorsFromBackendRequest(
    TRITONBACKEND_Request* request, AlgRunContext* context);

TRITONSERVER_Error* CreateServerRequestWithTensors(
    BackendRequestInfo* request_info, TRITONSERVER_Server* server,
    const char* graph_name, const OCTensorList* input_tensors,
    const StringList& input_names, const StringList& output_names,
    TRITONSERVER_InferenceRequest** irequest);

TRITONSERVER_Error* ParseTensorsFromServerResponse(
    TRITONSERVER_InferenceResponse* response, const StringList& output_names,
    OCTensorList* tensors);

TRITONSERVER_Error* ConstructFinalResponse(
    TRITONBACKEND_Response** response, AlgRunContext* context,
    const StringList& names);

TRITONSERVER_Error* ConstructFinalResponse(
    TRITONBACKEND_Response** response, const std::string& resp,
    const std::string& name, bool use_raw_output);

//
// Graph Request/Response helper functions
//
TRITONSERVER_Error* GraphInferResponseAllocator(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id);

TRITONSERVER_Error* GraphInferResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

void GraphInferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp);

void GraphInferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags,
    void* userp);

void GraphInferResponseDelete(TRITONSERVER_InferenceResponse* response);

//
// GraphExecutor
//
// Execute inference request on a graph model.
//
class GraphExecutor {
 public:
  GraphExecutor(TRITONSERVER_Server* server);
  ~GraphExecutor();
  // Performs async inference request.
  TRITONSERVER_Error* AsyncExecute(
      TRITONSERVER_InferenceRequest* irequest, AlgRunContext* context,
      std::future<TRITONSERVER_InferenceResponse*>* future);

  TRITONSERVER_Server* GetServer() { return server_; }

 private:
  // The server object that encapsulates all the functionality of the Triton
  // server and allows access to the Triton server API.
  TRITONSERVER_Server* server_;

  // The allocator object that will be used for allocating output tensors.
  TRITONSERVER_ResponseAllocator* allocator_;
};

}}      // namespace dataelem::alg
#endif  // DATAELEM_FRAMEWORK_ALG_UTILS_H_
