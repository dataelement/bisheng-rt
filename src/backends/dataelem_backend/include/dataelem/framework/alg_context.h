#ifndef DATAELEM_FRAMEWORK_ALG_CONTEXT_H_
#define DATAELEM_FRAMEWORK_ALG_CONTEXT_H_

#include "triton/backend/backend_memory.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

#include "dataelem/common/json_utils.h"
#include "dataelem/framework/types.h"

namespace dataelem { namespace alg {


struct BackendRequestInfo {
  const char* request_id;
  uint64_t correlation_id;
  uint32_t flags;
};

// Most member function is not thread safe.
class AlgRunContext {
 public:
  AlgRunContext() = default;
  ~AlgRunContext()
  {
    tensors_.clear();

    for (auto& mem : tensor_memories_) {
      if (mem != nullptr) {
        delete mem;
      }
    }
    tensor_memories_.clear();
  }

  void Reset(std::vector<std::string>& reserved_names)
  {
    std::set<std::string> names(reserved_names.begin(), reserved_names.end());

    for (auto it = tensors_.begin(); it != tensors_.end();) {
      if (names.find(it->first) != names.end()) {
        ++it;
      } else {
        it = tensors_.erase(it);
      }
    }
  }

  bool GetParameter(const std::string& name, int* value)
  {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return false;
    }
    *value = it->second.m().at<int>(0);
    return true;
  }

  bool GetParameter(const std::string& name, bool* value)
  {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return false;
    }
    *value = it->second.m().at<uint8_t>(0);
    return true;
  }

  bool GetParameter(const std::string& name, float* value)
  {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return false;
    }
    *value = it->second.m().at<float>(0);
    return true;
  }

  bool GetParameter(const std::string& name, absl::string_view& value)
  {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return false;
    }
    value = it->second.GetString(0);
    return true;
  }

  bool GetTensor(const std::string& name, OCTensor& tensor)
  {
    bool state = true;
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
      tensor = it->second;
    } else {
      state = false;
    }
    return state;
  }

  bool GetTensor(const std::string& name, OCTensor** tensor)
  {
    bool state = true;
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
      *tensor = &it->second;
    } else {
      state = false;
    }
    return state;
  }

  // the interface
  bool GetTensor(const StringList& names, OCTensorList& tensors)
  {
    bool state = true;
    for (size_t i = 0; i < names.size(); i++) {
      auto it = tensors_.find(names[i]);
      if (it != tensors_.end()) {
        tensors.emplace_back(it->second);
      } else {
        state = false;
        break;
      }
    }
    return state;
  }

  bool SetTensor(absl::string_view name, cv::Mat&& m)
  {
    tensors_.emplace(std::make_pair(name, std::move(m)));
    return true;
  }

  bool SetTensor(absl::string_view name, std::vector<cv::Mat>&& mats)
  {
    for (auto& m : mats) {
      tensors_.emplace(std::make_pair(name, std::move(m)));
    }
    return true;
  }

  bool SetTensor(absl::string_view name, OCTensor&& tensor)
  {
    tensors_.emplace(std::make_pair(name, std::move(tensor)));
    return true;
  }

  bool SetTensor(const StringList& names, OCTensorList&& tensors)
  {
    for (size_t i = 0; i < names.size(); i++) {
      tensors_.emplace(std::make_pair(names[i], std::move(tensors[i])));
    }
    return true;
  }

  void AddBackendMemory(triton::backend::BackendMemory* ptr)
  {
    std::lock_guard<std::mutex> lock{mutex_};
    tensor_memories_.emplace_back(ptr);
  }

  void SetMemoryManager(TRITONBACKEND_MemoryManager* backend_memory_manager)
  {
    backend_memory_manager_ = backend_memory_manager;
  }

  TRITONBACKEND_MemoryManager* GetMemoryManager()
  {
    return backend_memory_manager_;
  }

  const char* HostPolicyName() { return host_policy_name_; }

  cudaStream_t CudaStream() { return stream_; }

  BackendRequestInfo* GetBackendRequestInfo() { return &request_info_; }

  void UpdateCudaStream(const char* policy_name, cudaStream_t stream)
  {
    host_policy_name_ = policy_name;
    stream_ = stream;
  }

  void SetBackendRequestInfo(const BackendRequestInfo& info)
  {
    request_info_ = info;
  }

 private:
  // BackendMemory for context level buffer
  std::vector<triton::backend::BackendMemory*> tensor_memories_;

  TRITONBACKEND_MemoryManager* backend_memory_manager_;
  std::mutex mutex_;

  // Tensors used in Algorithmer execuation
  std::unordered_map<std::string, OCTensor> tensors_;

  BackendRequestInfo request_info_;

  // For CUDA device data control
  const char* host_policy_name_;
  cudaStream_t stream_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_ALG_CONTEXT_H_
