#ifndef DATAELEM_COMMON_JSON_UTILS_H_
#define DATAELEM_COMMON_JSON_UTILS_H_

#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include "nlohmann/json.hpp"

#include "triton/backend/backend_common.h"
#include "triton/common/triton_json.h"

#include "dataelem/framework/types.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace dataelem { namespace alg {

struct AppRequestInfo {
  int64_t request_id;
  int elapse;
};

#define SET_MILLI_TIMESTAMP(TS_NS)                                   \
  {                                                                  \
    TS_NS = std::chrono::duration_cast<std::chrono::milliseconds>(   \
                std::chrono::steady_clock::now().time_since_epoch()) \
                .count();                                            \
  }

// !TRITON_ENABLE_GPU
#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif

using JValue = triton::common::TritonJson::Value;

#define THROW_IF_TRITON_ERROR(X)                                       \
  do {                                                                 \
    TRITONSERVER_Error* tie_err__ = (X);                               \
    if (tie_err__ != nullptr) {                                        \
      throw AlgBackendException(TRITONSERVER_ErrorMessage(tie_err__)); \
    }                                                                  \
  } while (false)


#define CHECK_ERROR_WITH_BREAK(err) \
  if ((err) != nullptr)             \
  break


#define ALG_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())

#define ALG_STATUS_OK nullptr
//
// AlgBackendException
//
// Exception thrown if error occurs in BLSBackend.
//
struct AlgBackendException : std::exception {
  AlgBackendException(const std::string& message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

inline bool
ParseArrayFromString(const std::string& content, std::vector<float>& values)
{
  auto vstr_arr = absl::StrSplit(content, ' ');
  bool ret = true;
  float v = 0;
  ;
  for (auto& vstr : vstr_arr) {
    if (absl::SimpleAtof(vstr, &v)) {
      ret = false;
      break;
    }
    values.emplace_back(v);
  }
  return ret;
}

inline bool
ParseArrayFromString(const std::string& content, std::vector<double>& values)
{
  auto vstr_arr = absl::StrSplit(content, ' ');
  bool ret = true;
  double v = 0;
  for (auto& vstr : vstr_arr) {
    if (!absl::SimpleAtod(vstr, &v)) {
      ret = false;
      break;
    }
    values.emplace_back(v);
  }
  return ret;
}

inline bool
ParseArrayFromString(const std::string& content, std::vector<int>& values)
{
  auto vstr_arr = absl::StrSplit(content, ' ');
  bool ret = true;
  int v = 0;
  ;
  for (auto& vstr : vstr_arr) {
    if (!absl::SimpleAtoi(vstr, &v)) {
      ret = false;
      break;
    }
    values.emplace_back(v);
  }
  return ret;
}

inline bool
ParseArrayFromString(
    const std::string& content, std::vector<std::string>& values)
{
  values = absl::StrSplit(content, ' ');
  return true;
}

void SafeParseParameter(
    JValue& params, const std::string& mkey, float* value,
    const float& default_value);

void SafeParseParameter(
    JValue& params, const std::string& mkey, int* value,
    const int& default_value);

void SafeParseParameter(
    JValue& params, const std::string& mkey, std::string* value,
    const std::string& default_value);

void SafeParseParameter(
    JValue& params, const std::string& mkey, bool* value,
    const bool& default_value);


void SafeParseParameter(JValue& params, const std::string& mkey, float* value);

void SafeParseParameter(JValue& params, const std::string& mkey, int* value);

void SafeParseParameter(
    JValue& params, const std::string& mkey, std::string* value);

void SafeParseParameter(JValue& params, const std::string& mkey, bool* value);

void SafeParseParameter(JValue& params, const std::string& mkey, double* value);

inline TRITONSERVER_Error*
parse_nlo_json(absl::string_view content, nlohmann::json& json)
{
  try {
    json = nlohmann::json::parse(
        content.data(), content.data() + content.length());
  }
  catch (nlohmann::json::parse_error& e) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
  return nullptr;
}

bool WriteOKResponse(
    rapidjson::StringBuffer* buffer, const OCTensorList& tensors,
    const StringList& names, AppRequestInfo* info);

inline std::string
WriteErrorResponse(
    int error_code, std::string& error_message, AppRequestInfo* info = nullptr)
{
  rapidjson::StringBuffer buffer;
  triton::common::SimpleWriter writer(buffer);
  writer.StartObject();
  WriteKeyValue(&writer, "code", error_code);
  WriteKeyValue(&writer, "message", error_message.c_str());
  if (info != nullptr) {
    WriteKeyValue(&writer, "request_id", info->request_id);
  }
  writer.EndObject();
  return buffer.GetString();
}


inline std::string
WriteErrorResponse(TRITONSERVER_Error* err, AppRequestInfo* info = nullptr)
{
  rapidjson::StringBuffer buffer;
  int error_code = 300 + int(TRITONSERVER_ErrorCode(err));
  const char* error_message = TRITONSERVER_ErrorMessage(err);
  triton::common::SimpleWriter writer(buffer);
  writer.StartObject();
  WriteKeyValue(&writer, "code", error_code);
  WriteKeyValue(&writer, "message", error_message);
  if (info != nullptr) {
    WriteKeyValue(&writer, "request_id", info->request_id);
  }
  writer.EndObject();
  TRITONSERVER_ErrorDelete(err);
  return buffer.GetString();
}


}}  // namespace dataelem::alg

#endif  // DATAELEM_COMMON_JSON_UTILS_H_