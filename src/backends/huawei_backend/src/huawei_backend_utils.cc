// Copyright (c) 2022, DATAELEM INC. All rights reserved.
//

#include "huawei_backend_utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>


template TRITONHUAWEI_Shape::TRITONHUAWEI_Shape(
    const std::vector<int64_t>& shape);

template TRITONHUAWEI_Shape::TRITONHUAWEI_Shape(
    const std::vector<int32_t>& shape);

template <typename T>
TRITONHUAWEI_Shape::TRITONHUAWEI_Shape(const std::vector<T>& shape)
{
  shape_ = std::vector<int64_t>(shape.cbegin(), shape.cend());
  numel_ = std::accumulate(
      shape_.cbegin(), shape_.cend(), 1, std::multiplies<int64_t>());
}

TRITONHUAWEI_DataType
ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return TRITONHUAWEI_TYPE_INVALID;
    case TRITONSERVER_TYPE_UINT8:
      return TRITONHUAWEI_TYPE_UINT8;
    case TRITONSERVER_TYPE_INT8:
      return TRITONHUAWEI_TYPE_INT8;
    case TRITONSERVER_TYPE_INT32:
      return TRITONHUAWEI_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return TRITONHUAWEI_TYPE_INT64;
    case TRITONSERVER_TYPE_FP32:
      return TRITONHUAWEI_TYPE_FP32;
    case TRITONSERVER_TYPE_FP16:
      return TRITONHUAWEI_TYPE_FP16;
    default:
      break;
  }
  return TRITONHUAWEI_TYPE_INVALID;
}

TRITONSERVER_DataType
ConvertDataType(TRITONHUAWEI_DataType dtype)
{
  switch (dtype) {
    case TRITONHUAWEI_TYPE_INVALID:
      return TRITONSERVER_TYPE_INVALID;
    case TRITONHUAWEI_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case TRITONHUAWEI_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case TRITONHUAWEI_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case TRITONHUAWEI_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case TRITONHUAWEI_TYPE_FP32:
      return TRITONSERVER_TYPE_FP32;
    case TRITONHUAWEI_TYPE_FP16:
      return TRITONSERVER_TYPE_FP16;
    default:
      break;
  }
  return TRITONSERVER_TYPE_INVALID;
}

TRITONHUAWEI_DataType
ConvertDataType(const std::string& dtype)
{
  if (dtype == "TYPE_INVALID") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INVALID;
  } else if (dtype == "TYPE_FP32") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_FP32;
  } else if (dtype == "TYPE_UINT8") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_UINT8;
  } else if (dtype == "TYPE_INT8") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT8;
  } else if (dtype == "TYPE_INT32") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT32;
  } else if (dtype == "TYPE_INT64") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT64;
  } else if (dtype == "TYPE_FP16") {
    return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_FP16;
  }
  return TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INVALID;
}

size_t
TRITONHUAWEI_DataTypeByteSize(TRITONHUAWEI_DataType dtype)
{
  switch (dtype) {
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_FP32:
      return sizeof(float);
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT64:
      return sizeof(int64_t);
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT32:
      return sizeof(int32_t);
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_UINT8:
      return sizeof(uint8_t);
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_INT8:
      return sizeof(int8_t);
    case TRITONHUAWEI_DataType::TRITONHUAWEI_TYPE_FP16:
      return 2 * sizeof(int8_t);
      // return sizeof(phi::dtype::float16);
    default:
      break;
  }
  return 0;  // Should not happened, TODO: Error handling
}

/* Error message */

TRITONHUAWEI_Error*
TRITONHUAWEI_ErrorNew(const std::string& str)
{
  TRITONHUAWEI_Error* error = new TRITONHUAWEI_Error();
  error->msg_ = new char[str.size() + 1];
  std::strcpy(error->msg_, str.c_str());
  return error;
}

void
TRITONHUAWEI_ErrorDelete(TRITONHUAWEI_Error* error)
{
  if (error == nullptr) {
    return;
  }

  delete[] error->msg_;
  delete error;
}
