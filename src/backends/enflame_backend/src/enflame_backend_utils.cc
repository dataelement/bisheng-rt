// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#include "enflame_backend_utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>


template TRITONENFLAME_Shape::TRITONENFLAME_Shape(
    const std::vector<int64_t>& shape);

template TRITONENFLAME_Shape::TRITONENFLAME_Shape(
    const std::vector<int32_t>& shape);

template <typename T>
TRITONENFLAME_Shape::TRITONENFLAME_Shape(const std::vector<T>& shape)
{
  shape_ = std::vector<int64_t>(shape.cbegin(), shape.cend());
  numel_ = std::accumulate(
      shape_.cbegin(), shape_.cend(), 1, std::multiplies<int64_t>());
}

TRITONENFLAME_DataType
ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return TRITONENFLAME_TYPE_INVALID;
    case TRITONSERVER_TYPE_UINT8:
      return TRITONENFLAME_TYPE_UINT8;
    case TRITONSERVER_TYPE_INT8:
      return TRITONENFLAME_TYPE_INT8;
    case TRITONSERVER_TYPE_INT32:
      return TRITONENFLAME_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return TRITONENFLAME_TYPE_INT64;
    case TRITONSERVER_TYPE_FP32:
      return TRITONENFLAME_TYPE_FP32;
    case TRITONSERVER_TYPE_FP16:
      return TRITONENFLAME_TYPE_FP16;
    default:
      break;
  }
  return TRITONENFLAME_TYPE_INVALID;
}

TRITONSERVER_DataType
ConvertDataType(TRITONENFLAME_DataType dtype)
{
  switch (dtype) {
    case TRITONENFLAME_TYPE_INVALID:
      return TRITONSERVER_TYPE_INVALID;
    case TRITONENFLAME_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case TRITONENFLAME_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case TRITONENFLAME_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case TRITONENFLAME_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case TRITONENFLAME_TYPE_FP32:
      return TRITONSERVER_TYPE_FP32;
    case TRITONENFLAME_TYPE_FP16:
      return TRITONSERVER_TYPE_FP16;
    default:
      break;
  }
  return TRITONSERVER_TYPE_INVALID;
}

TRITONENFLAME_DataType
ConvertDataType(const std::string& dtype)
{
  if (dtype == "TYPE_INVALID") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INVALID;
  } else if (dtype == "TYPE_FP32") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_FP32;
  } else if (dtype == "TYPE_UINT8") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_UINT8;
  } else if (dtype == "TYPE_INT8") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT8;
  } else if (dtype == "TYPE_INT32") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT32;
  } else if (dtype == "TYPE_INT64") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT64;
  } else if (dtype == "TYPE_FP16") {
    return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_FP16;
  }
  return TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INVALID;
}

size_t
TRITONENFLAME_DataTypeByteSize(TRITONENFLAME_DataType dtype)
{
  switch (dtype) {
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_FP32:
      return sizeof(float);
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT64:
      return sizeof(int64_t);
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT32:
      return sizeof(int32_t);
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_UINT8:
      return sizeof(uint8_t);
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_INT8:
      return sizeof(int8_t);
    case TRITONENFLAME_DataType::TRITONENFLAME_TYPE_FP16:
      return 2 * sizeof(int8_t);
      // return sizeof(phi::dtype::float16);
    default:
      break;
  }
  return 0;  // Should not happened, TODO: Error handling
}

/* Error message */

TRITONENFLAME_Error*
TRITONENFLAME_ErrorNew(const std::string& str)
{
  TRITONENFLAME_Error* error = new TRITONENFLAME_Error();
  error->msg_ = new char[str.size() + 1];
  std::strcpy(error->msg_, str.c_str());
  return error;
}

void
TRITONENFLAME_ErrorDelete(TRITONENFLAME_Error* error)
{
  if (error == nullptr) {
    return;
  }

  delete[] error->msg_;
  delete error;
}
