// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <cnrt.h>
#include "mm_builder.h"
#include "mm_network.h"
#include "mm_runtime.h"
#include "triton/core/tritonserver.h"


#define THROW_CHECK_ERROR(x, msg)                            \
  do {                                                       \
    if (!(x)) {                                              \
      TRITONCAMBRICON_Error* error = TRITONCAMBRICON_Error(msg); \
      THROW_IF_TRITONCAMBRICON_ERROR(error);                   \
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

#define RETURN_IF_TRITONCAMBRICON_ERROR(ERR)                                   \
  do {                                                                       \
    TRITONCAMBRICON_Error* error__ = (ERR);                                    \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONCAMBRICON_ErrorDelete(error__);                                    \
      return status;                                                         \
    }                                                                        \
  } while (false)

#define THROW_IF_TRITONCAMBRICON_ERROR(X)         \
  do {                                          \
    TRITONCAMBRICON_Error* tie_err__ = (X);       \
    if (tie_err__ != nullptr) {                 \
      throw TRITONCAMBRICON_Exception(tie_err__); \
    }                                           \
  } while (false)

typedef struct {
  char* msg_;
} TRITONCAMBRICON_Error;

struct TRITONCAMBRICON_Exception {
  TRITONCAMBRICON_Exception(TRITONCAMBRICON_Error* err) : err_(err) {}
  TRITONCAMBRICON_Error* err_;
};

TRITONCAMBRICON_Error* TRITONCAMBRICON_ErrorNew(const std::string& str);

void TRITONCAMBRICON_ErrorDelete(TRITONCAMBRICON_Error* error);

// TRITONCAMBRICON TYPE
typedef enum {
  TRITONCAMBRICON_TYPE_FP32,
  TRITONCAMBRICON_TYPE_INT64,
  TRITONCAMBRICON_TYPE_INT32,
  TRITONCAMBRICON_TYPE_UINT8,
  TRITONCAMBRICON_TYPE_INT8,
  TRITONCAMBRICON_TYPE_FP16,
  TRITONCAMBRICON_TYPE_INVALID
} TRITONCAMBRICON_DataType;

// TRITONCAMBRICON SHAPE
class TRITONCAMBRICON_Shape {
 public:
  TRITONCAMBRICON_Shape() = default;

  template <typename T>
  TRITONCAMBRICON_Shape(const std::vector<T>& shape);

  size_t NumElements() const { return numel_; };
  std::vector<int64_t> Shape() const { return shape_; };

 private:
  std::vector<int64_t> shape_;
  size_t numel_;
};

TRITONCAMBRICON_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONCAMBRICON_DataType ConvertDataType(const std::string& dtype);

TRITONSERVER_DataType ConvertDataType(TRITONCAMBRICON_DataType dtype);

size_t TRITONCAMBRICON_DataTypeByteSize(TRITONCAMBRICON_DataType dtype);
