// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "triton/core/tritonserver.h"

#define THROW_CHECK_ERROR(x, msg)                            \
  do {                                                       \
    if (!(x)) {                                              \
      TRITONHUAWEI_Error* error = TRITONHUAWEI_Error(msg); \
      THROW_IF_TRITONHUAWEI_ERROR(error);                   \
    }                                                        \
  } while (false)


#define CHECK_ERROR_WITH_BREAK(err) \
  if ((err) != nullptr)             \
  break

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

#define RETURN_IF_TRITONHUAWEI_ERROR(ERR)                                   \
  do {                                                                       \
    TRITONHUAWEI_Error* error__ = (ERR);                                    \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONHUAWEI_ErrorDelete(error__);                                    \
      return status;                                                         \
    }                                                                        \
  } while (false)

#define THROW_IF_TRITONHUAWEI_ERROR(X)         \
  do {                                          \
    TRITONHUAWEI_Error* tie_err__ = (X);       \
    if (tie_err__ != nullptr) {                 \
      throw TRITONHUAWEI_Exception(tie_err__); \
    }                                           \
  } while (false)

typedef struct {
  char* msg_;
} TRITONHUAWEI_Error;

struct TRITONHUAWEI_Exception {
  TRITONHUAWEI_Exception(TRITONHUAWEI_Error* err) : err_(err) {}
  TRITONHUAWEI_Error* err_;
};

TRITONHUAWEI_Error* TRITONHUAWEI_ErrorNew(const std::string& str);

void TRITONHUAWEI_ErrorDelete(TRITONHUAWEI_Error* error);

// TRITONHUAWEI TYPE
typedef enum {
  TRITONHUAWEI_TYPE_FP32,
  TRITONHUAWEI_TYPE_INT64,
  TRITONHUAWEI_TYPE_INT32,
  TRITONHUAWEI_TYPE_UINT8,
  TRITONHUAWEI_TYPE_INT8,
  TRITONHUAWEI_TYPE_FP16,
  TRITONHUAWEI_TYPE_INVALID
} TRITONHUAWEI_DataType;

// TRITONHUAWEI SHAPE
class TRITONHUAWEI_Shape {
 public:
  TRITONHUAWEI_Shape() = default;

  template <typename T>
  TRITONHUAWEI_Shape(const std::vector<T>& shape);

  size_t NumElements() const { return numel_; };
  std::vector<int64_t> Shape() const { return shape_; };

 private:
  std::vector<int64_t> shape_;
  size_t numel_;
};

TRITONHUAWEI_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONHUAWEI_DataType ConvertDataType(const std::string& dtype);

TRITONSERVER_DataType ConvertDataType(TRITONHUAWEI_DataType dtype);

size_t TRITONHUAWEI_DataTypeByteSize(TRITONHUAWEI_DataType dtype);
