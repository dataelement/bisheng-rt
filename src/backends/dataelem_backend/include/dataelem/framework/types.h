#ifndef DATAELEM_FRAMEWORK_TYPES_H_
#define DATAELEM_FRAMEWORK_TYPES_H_

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "opencv2/opencv.hpp"
#include "triton/core/tritonserver.h"

namespace dataelem { namespace alg {

class OCTensor;

typedef std::vector<cv::Mat> MatList;
typedef std::pair<cv::Mat, cv::Mat> PairMat;
typedef std::vector<PairMat> PairMatList;

typedef std::vector<std::string> StringList;
typedef std::vector<StringList> StringListList;
typedef std::vector<cv::Point2f> Point2fList;
typedef std::vector<int> IntegerList;

typedef std::vector<OCTensor> OCTensorList;

struct SeqList {
  std::vector<std::string> texts;
  std::vector<float> scores;
};

// define type compatible for tensorflow
typedef unsigned char uchar;
typedef unsigned int uint;
typedef cv::Scalar_<float> Scalarf;

typedef enum {
  TRITONOPENCV_TYPE_UINT8 = 0,
  TRITONOPENCV_TYPE_INT8,
  TRITONOPENCV_TYPE_UINT16,
  TRITONOPENCV_TYPE_INT16,
  TRITONOPENCV_TYPE_INT32,
  TRITONOPENCV_TYPE_FP32,
  TRITONOPENCV_TYPE_FP64,
  TRITONOPENCV_TYPE_UINT32 = 100,
  TRITONOPENCV_TYPE_UINT64,
  TRITONOPENCV_TYPE_INT64,
  TRITONOPENCV_TYPE_STRING,
  TRITONOPENCV_TYPE_INVALID = 200
} TRITONOPENCV_DataType;


TRITONOPENCV_DataType inline ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return TRITONOPENCV_TYPE_INVALID;
    case TRITONSERVER_TYPE_UINT8:
      return TRITONOPENCV_TYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return TRITONOPENCV_TYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return TRITONOPENCV_TYPE_UINT32;
    case TRITONSERVER_TYPE_INT8:
      return TRITONOPENCV_TYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return TRITONOPENCV_TYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return TRITONOPENCV_TYPE_INT32;
    case TRITONSERVER_TYPE_FP32:
      return TRITONOPENCV_TYPE_FP32;
    case TRITONSERVER_TYPE_FP64:
      return TRITONOPENCV_TYPE_FP64;
    case TRITONSERVER_TYPE_INT64:
      return TRITONOPENCV_TYPE_INT64;
    case TRITONSERVER_TYPE_UINT64:
      return TRITONOPENCV_TYPE_UINT64;
    case TRITONSERVER_TYPE_BYTES:
      return TRITONOPENCV_TYPE_STRING;
    default:
      break;
  }
  return TRITONOPENCV_TYPE_INVALID;
}

TRITONSERVER_DataType inline ConvertDataType(TRITONOPENCV_DataType dtype)
{
  switch (dtype) {
    case TRITONOPENCV_TYPE_INVALID:
      return TRITONSERVER_TYPE_INVALID;
    case TRITONOPENCV_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case TRITONOPENCV_TYPE_UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case TRITONOPENCV_TYPE_UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case TRITONOPENCV_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case TRITONOPENCV_TYPE_INT16:
      return TRITONSERVER_TYPE_INT16;
    case TRITONOPENCV_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case TRITONOPENCV_TYPE_FP32:
      return TRITONSERVER_TYPE_FP32;
    case TRITONOPENCV_TYPE_FP64:
      return TRITONSERVER_TYPE_FP64;
    case TRITONOPENCV_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case TRITONOPENCV_TYPE_UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case TRITONOPENCV_TYPE_STRING:
      return TRITONSERVER_TYPE_BYTES;
    default:
      break;
  }
  return TRITONSERVER_TYPE_INVALID;
}


inline TRITONSERVER_DataType
ModelConfigDataTypeToServerType(const std::string& data_type_str)
{
  TRITONSERVER_DataType type = TRITONSERVER_TYPE_INVALID;

  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return type;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    type = TRITONSERVER_TYPE_UINT8;
  } else if (dtype == "UINT8") {
    type = TRITONSERVER_TYPE_UINT8;
  } else if (dtype == "INT8") {
    type = TRITONSERVER_TYPE_INT8;
  } else if (dtype == "INT16") {
    type = TRITONSERVER_TYPE_INT16;
  } else if (dtype == "INT32") {
    type = TRITONSERVER_TYPE_INT32;
  } else if (dtype == "INT64") {
    type = TRITONSERVER_TYPE_INT64;
  } else if (dtype == "FP32") {
    type = TRITONSERVER_TYPE_FP32;
  } else if (dtype == "FP64") {
    type = TRITONSERVER_TYPE_FP64;
  } else if (dtype == "STRING") {
    type = TRITONSERVER_TYPE_BYTES;
  } else {
    return type;
  }

  return type;
}

// OpenCV Extended Tensor
class OCTensor {
 public:
  OCTensor() = default;

  // bytes_data may empty, the data fo tensor may be reference
  // so bytes_view should copy from bytes_view
  OCTensor(const OCTensor& tensor)
      : bytes_data_(tensor.bytes_data_), bytes_view_(tensor.bytes_view_),
        mat_(tensor.mat_), shape_(tensor.shape_), type_(tensor.type_),
        device_(tensor.device_), is_pinned_(tensor.is_pinned_),
        is_b64_(tensor.is_b64_), is_jsonstr_(tensor.is_jsonstr_)
  {
    // Trickly, but very important, if bytes data moved, update view
    if (!bytes_data_.empty()) {
      bytes_view_ = bytes_data_;
    }
  }

  OCTensor(OCTensor&& tensor)
      : bytes_data_(std::move(tensor.bytes_data_)),
        bytes_view_(tensor.bytes_view_), mat_(std::move(tensor.mat_)),
        shape_(std::move(tensor.shape_)), type_(tensor.type_),
        device_(tensor.device_), is_pinned_(tensor.is_pinned_),
        is_b64_(tensor.is_b64_), is_jsonstr_(tensor.is_jsonstr_)
  {
    // Trickly, but very important, if bytes data moved, update view
    if (!bytes_data_.empty()) {
      bytes_view_ = bytes_data_;
    }
  }

  OCTensor(const cv::Mat& m) : mat_(m)
  {
    type_ = static_cast<TRITONOPENCV_DataType>(mat_.depth());
    for (int i = 0; i < mat_.size.dims(); i++) {
      shape_.emplace_back(int64_t(mat_.size.p[i]));
    }
    if ((mat_.cols != -1 && mat_.channels() > 1) ||
        (mat_.size.dims() > 2 && mat_.channels() > 1)) {
      shape_.emplace_back(mat_.channels());
    }
    device_ = -1;
  }

  OCTensor(cv::Mat&& m) : mat_(std::move(m))
  {
    type_ = static_cast<TRITONOPENCV_DataType>(mat_.depth());
    for (int i = 0; i < mat_.size.dims(); i++) {
      shape_.emplace_back(int64_t(mat_.size.p[i]));
    }
    if ((mat_.cols != -1 && mat_.channels() > 1) ||
        (mat_.size.dims() > 2 && mat_.channels() > 1)) {
      shape_.emplace_back(mat_.channels());
    }
    device_ = -1;
  }

  OCTensor(
      uint32_t rank, const int64_t* dims, TRITONSERVER_DataType type,
      const char* buffer, size_t byte_size, int device = -1)
  {
    type_ = ConvertDataType(type);
    if (type_ == TRITONOPENCV_TYPE_INVALID) {
      return;
    }

    for (uint32_t i = 0; i < rank; i++) {
      shape_.push_back(dims[i]);
    }
    device_ = device;

    if (type_ == TRITONOPENCV_TYPE_STRING) {
      bytes_view_ = absl::string_view(buffer, byte_size);
    } else {
      // opencv do not support uint64, int64, manually use raw data
      auto mat_type = type_;
      if (type_ == TRITONOPENCV_TYPE_UINT64 ||
          type_ == TRITONOPENCV_TYPE_INT64) {
        mat_type = TRITONOPENCV_TYPE_FP64;
      } else if (type_ == TRITONOPENCV_TYPE_UINT32) {
        mat_type = TRITONOPENCV_TYPE_INT32;
      }
      std::vector<int> opencv_shape;
      for (const auto& s : shape_) {
        opencv_shape.emplace_back((int)s);
      }
      mat_ = cv::Mat(opencv_shape, mat_type, const_cast<char*>(buffer));
    }
  }

  // Empty tensor construct, shape must contains zero dim
  OCTensor(const std::vector<int>& shape, int dtype)
  {
    type_ = static_cast<TRITONOPENCV_DataType>(dtype);
    for (auto& s : shape) {
      shape_.emplace_back(int64_t(s));
    }
    device_ = -1;
    mat_ = cv::Mat(shape, dtype);
  }

  OCTensor(
      const std::vector<std::string>& vs, const std::vector<int64_t>& shape)
  {
    shape_.assign(shape.begin(), shape.end());
    type_ = TRITONOPENCV_TYPE_STRING;
    device_ = -1;

    // serilize data to bytes_data_
    for (size_t e = 0; e < vs.size(); ++e) {
      SerializeString(&bytes_data_, vs[e]);
    }
    bytes_view_ = absl::string_view(bytes_data_);
  }

  OCTensor(
      const std::vector<absl::string_view>& vs,
      const std::vector<int64_t>& shape)
  {
    shape_.assign(shape.begin(), shape.end());
    type_ = TRITONOPENCV_TYPE_STRING;
    device_ = -1;

    // serilize data to bytes_data_
    for (size_t e = 0; e < vs.size(); ++e) {
      SerializeString(&bytes_data_, vs[e]);
    }
    bytes_view_ = absl::string_view(bytes_data_);
  }

  OCTensor(absl::string_view v, bool copy = true)
  {
    if (copy == true) {
      SerializeString(&bytes_data_, v);
      bytes_view_ = bytes_data_;
    } else {
      bytes_view_ = v;
    };
    shape_.assign({1});
    type_ = TRITONOPENCV_TYPE_STRING;
    device_ = -1;
  }

  static OCTensor from_blob(
      uint32_t rank, const int64_t* dims, TRITONSERVER_DataType type,
      const char* buffer, size_t byte_size, int device = -1)
  {
    return OCTensor(rank, dims, type, buffer, byte_size, device);
  }

  OCTensor& operator=(cv::Mat& v)
  {
    // !!cv::Mat is used as high dimention tensor, type equals depth
    type_ = static_cast<TRITONOPENCV_DataType>(v.type());
    shape_.assign(v.size.p, v.size.p + v.size.dims());
    mat_ = v;
    device_ = -1;
    return *this;
  }

  OCTensor& operator=(const OCTensor& tensor)
  {
    type_ = tensor.type_;
    shape_ = tensor.shape_;
    mat_ = tensor.mat_;
    device_ = tensor.device_;
    bytes_view_ = tensor.bytes_view_;

    // Need copy bytes_data, otherwise it is very dangerous.
    // large bytes tensor Copy construct op is somehow costly
    bytes_data_ = tensor.bytes_data_;
    if (!bytes_data_.empty()) {
      bytes_view_ = bytes_data_;
    }

    return *this;
  }

  // operator cv::Mat&() const {
  //   return mat_;
  // }
  cv::Mat GetMatrix() const
  {
    int dim0 = shape_[0];
    int dim1 = 1;
    for (size_t i = 1; i < shape_.size(); i++) dim1 *= shape_[i];
    std::vector<int> m_shape = {dim0, dim1};
    return mat_.reshape(1, m_shape);
  }

  cv::Mat GetImage() const
  {
    int dim0 = shape_[0];
    int dim1 = shape_[1];
    int dim2 = shape_[2];
    auto type = dim2 == 3 ? CV_8UC3 : CV_8UC1;
    return cv::Mat(dim0, dim1, type, mat_.data);
  }

  cv::Mat GetMat() const { return mat_; }

  const cv::Mat& m() const { return mat_; }

  absl::string_view GetString(size_t idx) const
  {
    uint32_t offset = 0;
    uint32_t len = 0;
    auto LEN_BYTES = sizeof(uint32_t);
    uint32_t max_offset = bytes_view_.length() - LEN_BYTES;
    const char* ptr = bytes_view_.data();
    for (size_t i = 0; i < idx && offset <= max_offset; i++) {
      len = *(reinterpret_cast<const uint32_t*>(ptr + offset));
      offset += len + LEN_BYTES;
    }

    len = *(reinterpret_cast<const uint32_t*>(ptr + offset));
    if (len <= max_offset - offset) {
      return absl::string_view(ptr + offset + LEN_BYTES, len);
    } else {
      return "null";
    }
  }

  bool GetStrings(std::vector<absl::string_view>& strs)
  {
    uint32_t offset = 0;
    uint32_t len = 0;
    auto LEN_BYTES = sizeof(uint32_t);
    uint32_t max_offset = bytes_view_.length() - LEN_BYTES;
    const char* ptr = bytes_view_.data();
    size_t n = shape_[0];
    for (size_t i = 0; i < n && offset <= max_offset; i++) {
      len = *(reinterpret_cast<const uint32_t*>(ptr + offset));
      if (len <= max_offset - offset) {
        strs.push_back(absl::string_view(ptr + offset + LEN_BYTES, len));
      }
      offset += len + LEN_BYTES;
    }
    return strs.size() == n;
  }

  TRITONOPENCV_DataType dtype() const { return type_; }

  std::vector<int64_t> shape() const { return shape_; }
  const int64_t* shape_ptr() const { return shape_.data(); }
  void set_shape(const std::vector<int64_t>& s) { shape_ = s; }

  size_t ElemCount() const
  {
    size_t n = 1;
    for (const auto& s : shape_) {
      n *= s;
    }
    return n;
  }

  absl::string_view bytes_view() const { return bytes_view_; }

  bool empty() { return mat_.empty() && bytes_view_.length() == 0; }

  size_t ByteSize() const
  {
    const std::map<size_t, size_t> TYPE2BYTES_MAP = {
        {TRITONOPENCV_TYPE_UINT8, 1},  {TRITONOPENCV_TYPE_INT8, 1},
        {TRITONOPENCV_TYPE_UINT16, 2}, {TRITONOPENCV_TYPE_INT16, 2},
        {TRITONOPENCV_TYPE_INT32, 4},  {TRITONOPENCV_TYPE_UINT32, 4},
        {TRITONOPENCV_TYPE_FP32, 4},   {TRITONOPENCV_TYPE_FP64, 8},
        {TRITONOPENCV_TYPE_UINT64, 8}, {TRITONOPENCV_TYPE_INT64, 8},
        {TRITONOPENCV_TYPE_INVALID, 0}};

    if (type_ == TRITONOPENCV_TYPE_STRING) {
      return bytes_view_.length();
    } else {
      size_t n = TYPE2BYTES_MAP.at(type_);
      for (auto& d : shape_) {
        n *= (size_t)d;
      }
      return n;
    }
  }

  const char* data_ptr() const
  {
    if (type_ == TRITONOPENCV_TYPE_STRING) {
      return bytes_view_.data();
    } else {
      return reinterpret_cast<const char*>(mat_.data);
    }
  }

  int device() const { return device_; };

  void set_pinned() { is_pinned_ = true; };
  bool pinned() const { return is_pinned_; };

  void set_base64() { is_b64_ = true; };
  bool base64() const { return is_b64_; };

  void set_jsonstr() { is_jsonstr_ = true; };
  bool jsonstr() const { return is_jsonstr_; };

 private:
  inline void SerializeString(std::string* buffer, absl::string_view v)
  {
    const char* cstr = v.data();
    uint32_t len = v.length();
    buffer->append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    if (len > 0) {
      buffer->append(cstr, len);
    }
  }

 private:
  // Store for strings, mats
  std::string bytes_data_;
  absl::string_view bytes_view_;
  cv::Mat mat_;

  std::vector<int64_t> shape_;
  TRITONOPENCV_DataType type_;
  int device_;  // -1: CPU , n: GPU0
  bool is_pinned_ = false;
  bool is_b64_ = false;
  bool is_jsonstr_ = false;
};


inline void
print_tensor(const OCTensor& t, const std::string& name = "")
{
  std::cout << ">>>>>>>>>>>>>>>>>>>>\n";
  std::cout << "tensor.name:" << name << "\n";
  std::cout << "tensor::type=" << t.dtype() << "\n";

  auto dims = t.shape();
  std::cout << "tensor::dims=[";
  size_t i = 0;
  for (; i < dims.size() - 1; i++) {
    std::cout << dims[i] << ",";
  }
  std::cout << dims[i];
  std::cout << "]\n";

  std::cout << "tensor::data_ptr=" << (const void*)t.data_ptr() << "\n";
  std::cout << "tensor::byte_size=" << t.ByteSize() << "\n";

  if (t.dtype() == TRITONOPENCV_TYPE_STRING) {
    size_t n = std::min(t.ElemCount(), size_t(3));
    std::cout << "tensor::strings(0-2)=[";
    for (size_t i = 0; i < n; i++) {
      std::cout << t.GetString(i) << ",";
    }
    std::cout << "]\n";
  }
}

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_TYPES_H_