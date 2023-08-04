// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "TopsInference/TopsInferRuntime.h"
#include "TopsInference/dtu/util/switch_logging.h"
#include "TopsInference/utils/tops_utils.h"
#include "triton/core/tritonserver.h"


#define THROW_CHECK_ERROR(x, msg)                            \
  do {                                                       \
    if (!(x)) {                                              \
      TRITONENFLAME_Error* error = TRITONENFLAME_Error(msg); \
      THROW_IF_TRITONENFLAME_ERROR(error);                   \
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

#define RETURN_IF_TRITONENFLAME_ERROR(ERR)                                   \
  do {                                                                       \
    TRITONENFLAME_Error* error__ = (ERR);                                    \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONENFLAME_ErrorDelete(error__);                                    \
      return status;                                                         \
    }                                                                        \
  } while (false)

#define THROW_IF_TRITONENFLAME_ERROR(X)         \
  do {                                          \
    TRITONENFLAME_Error* tie_err__ = (X);       \
    if (tie_err__ != nullptr) {                 \
      throw TRITONENFLAME_Exception(tie_err__); \
    }                                           \
  } while (false)

typedef struct {
  char* msg_;
} TRITONENFLAME_Error;

struct TRITONENFLAME_Exception {
  TRITONENFLAME_Exception(TRITONENFLAME_Error* err) : err_(err) {}
  TRITONENFLAME_Error* err_;
};

TRITONENFLAME_Error* TRITONENFLAME_ErrorNew(const std::string& str);

void TRITONENFLAME_ErrorDelete(TRITONENFLAME_Error* error);

// TRITONENFLAME TYPE
typedef enum {
  TRITONENFLAME_TYPE_FP32,
  TRITONENFLAME_TYPE_INT64,
  TRITONENFLAME_TYPE_INT32,
  TRITONENFLAME_TYPE_UINT8,
  TRITONENFLAME_TYPE_INT8,
  TRITONENFLAME_TYPE_FP16,
  TRITONENFLAME_TYPE_INVALID
} TRITONENFLAME_DataType;

// TRITONENFLAME SHAPE
class TRITONENFLAME_Shape {
 public:
  TRITONENFLAME_Shape() = default;

  template <typename T>
  TRITONENFLAME_Shape(const std::vector<T>& shape);

  size_t NumElements() const { return numel_; };
  std::vector<int64_t> Shape() const { return shape_; };

 private:
  std::vector<int64_t> shape_;
  size_t numel_;
};

TRITONENFLAME_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONENFLAME_DataType ConvertDataType(const std::string& dtype);

TRITONSERVER_DataType ConvertDataType(TRITONENFLAME_DataType dtype);

size_t TRITONENFLAME_DataTypeByteSize(TRITONENFLAME_DataType dtype);

// TRITON ENFLAME MODE
typedef enum {
  TRITONENFLAME_MODE_DEFAULT,
  TRITONENFLAME_MODE_FP16,
  TRITONENFLAME_MODE_MIXED,
} TRITONENFLAME_Precision;
