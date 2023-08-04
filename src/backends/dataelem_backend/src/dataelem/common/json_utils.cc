#include "dataelem/common/json_utils.h"

namespace dataelem { namespace alg {

void
SafeParseParameter(
    JValue& params, const std::string& mkey, float* value,
    const float& default_value)
{
  triton::common::TritonJson::Value json_value;
  *value = default_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtof(string_value, value)) {
        *value = default_value;
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(
    JValue& params, const std::string& mkey, int* value,
    const int& default_value)
{
  triton::common::TritonJson::Value json_value;
  *value = default_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtoi(string_value, value)) {
        *value = default_value;
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(
    JValue& params, const std::string& mkey, std::string* value,
    const std::string& default_value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      *value = string_value;
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  } else {
    *value = default_value;
  }
}


void
SafeParseParameter(
    JValue& params, const std::string& mkey, bool* value,
    const bool& default_value)
{
  triton::common::TritonJson::Value json_value;
  *value = default_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtob(string_value, value)) {
        *value = default_value;
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}


void
SafeParseParameter(JValue& params, const std::string& mkey, float* value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtof(string_value, value)) {
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(JValue& params, const std::string& mkey, int* value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtoi(string_value, value)) {
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(JValue& params, const std::string& mkey, std::string* value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    auto err = json_value.MemberAsString("string_value", value);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(JValue& params, const std::string& mkey, bool* value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtob(string_value, value)) {
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}

void
SafeParseParameter(JValue& params, const std::string& mkey, double* value)
{
  triton::common::TritonJson::Value json_value;
  if (params.Find(mkey.c_str(), &json_value)) {
    std::string string_value;
    auto err = json_value.MemberAsString("string_value", &string_value);
    if (err == nullptr) {
      if (!absl::SimpleAtod(string_value, value)) {
      }
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
  }
}


template <typename stype, typename dtype>
bool
WriteValuesWithShape(
    triton::common::RapidJsonWriter* writer, const stype* val, int n,
    const std::vector<int64_t>& shape)
{
  size_t dim_n = shape.size();
  bool ret = true;
  if (dim_n == 1) {
    writer->StartArray();
    for (int i = 0; i < n; i++) {
      ret += triton::common::WriteValue(writer, dtype(val[i]));
    }
    writer->EndArray();
    return ret;
  }

  size_t dim0 = 1, dim1 = 1;
  if (dim_n == 2) {
    dim0 = shape[0];
    dim1 = shape[1];
  } else if (dim_n == 3) {
    auto t = shape[2] * shape[1];
    if (t <= 128) {
      dim0 = shape[0];
      dim1 = t;
    } else {
      dim0 = shape[0] * shape[1];
      dim1 = shape[2];
    }
  } else if (dim_n > 3) {
    for (size_t i = 0; i < dim_n - 1; i++) {
      dim0 *= shape[i];
    }
    dim1 = shape[dim_n - 1];
  }

  writer->StartArray();
  for (size_t i = 0; i < dim0; i++) {
    writer->StartArray();
    for (size_t j = 0; j < dim1; j++) {
      size_t index = i * dim1 + j;
      ret += triton::common::WriteValue(writer, dtype(val[index]));
    }
    writer->EndArray();
  }
  writer->EndArray();

  return ret;
}


template <>
inline bool
WriteValuesWithShape<uint8_t, bool>(
    triton::common::RapidJsonWriter* writer, const uint8_t* val, int n,
    const std::vector<int64_t>& shape)
{
  size_t dim_n = shape.size();
  bool ret = true;
  if (dim_n == 1) {
    writer->StartArray();
    for (int i = 0; i < n; i++) {
      ret += triton::common::WriteValue(writer, (val[i] == 0) ? false : true);
    }
    writer->EndArray();
    return ret;
  }

  size_t dim0 = 1, dim1 = 1;
  if (dim_n == 2) {
    dim0 = shape[0];
    dim1 = shape[1];
  } else if (dim_n == 3) {
    auto t = shape[1] * shape[1];
    if (t <= 128) {
      dim0 = shape[0];
      dim1 = t;
    } else {
      dim0 = shape[0] * shape[1];
      dim1 = shape[2];
    }
  } else if (dim_n > 3) {
    for (size_t i = 0; i < dim_n - 1; i++) {
      dim0 *= shape[i];
    }
    dim1 = shape[dim_n - 1];
  }

  writer->StartArray();
  for (size_t i = 0; i < dim0; i++) {
    writer->StartArray();
    for (size_t j = 0; j < dim1; j++) {
      size_t index = i * dim1 + j;
      ret += (triton::common::WriteValue(
          writer, (val[index] == 0) ? false : true));
    }
    writer->EndArray();
  }
  writer->EndArray();

  return ret;
}


#define MACRO_WRITE_VALUE_C1(TRTSERV_TYPE, STYPE, DTYPE)                     \
  case TRTSERV_TYPE: {                                                       \
    const STYPE* cbase = reinterpret_cast<const STYPE*>(base);               \
    triton::common::WriteValues<STYPE, DTYPE>(writer, cbase, element_count); \
    break;                                                                   \
  }

#define MACRO_WRITE_VALUE_C2(TRTSERV_TYPE, STYPE, DTYPE, SHAPE)              \
  case TRTSERV_TYPE: {                                                       \
    const STYPE* cbase = reinterpret_cast<const STYPE*>(base);               \
    WriteValuesWithShape<STYPE, DTYPE>(writer, cbase, element_count, SHAPE); \
    break;                                                                   \
  }

bool
WriteDataToJsonExt(
    triton::common::RapidJsonWriter* writer, const std::string& output_name,
    const TRITONSERVER_DataType datatype, const void* base,
    const size_t byte_size, const size_t element_count, bool is_b64,
    const std::vector<int64_t>& shape, bool is_jsonstr)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL: {
      const uint8_t* bool_base = reinterpret_cast<const uint8_t*>(base);
      if (byte_size != (element_count * sizeof(uint8_t))) {
        return false;
      }
      triton::common::WriteValues<uint8_t, bool>(
          writer, bool_base, element_count);
      break;
    }
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_UINT8, uint8_t, unsigned, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_UINT16, uint16_t, unsigned, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_UINT32, uint32_t, unsigned, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_UINT64, uint64_t, unsigned, shape)

      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_INT8, int8_t, int, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_INT16, int16_t, int, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_INT32, int32_t, int, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_INT64, int64_t, int, shape)

    // FP16 not supported via JSON
    case TRITONSERVER_TYPE_BF16:
    case TRITONSERVER_TYPE_FP16:
      return false;
      // default:
      //   break;
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_FP32, float, float, shape)
      MACRO_WRITE_VALUE_C2(TRITONSERVER_TYPE_FP64, double, double, shape)

    case TRITONSERVER_TYPE_BYTES: {
      if (is_jsonstr) {
        const char* cbase = reinterpret_cast<const char*>(base);
        size_t offset = 0;
        if (element_count == 1) {
          if ((offset + sizeof(uint32_t)) > byte_size) {
            return false;
          }

          const size_t len =
              *(reinterpret_cast<const uint32_t*>(cbase + offset));
          offset += sizeof(uint32_t);

          if ((offset + len) > byte_size) {
            return false;
          }
          writer->RawValue(cbase + offset, len, rapidjson::kObjectType);
        } else {
          writer->StartArray();
          const char* cbase = reinterpret_cast<const char*>(base);
          size_t offset = 0;
          for (size_t e = 0; e < element_count; ++e) {
            if ((offset + sizeof(uint32_t)) > byte_size) {
              return false;
            }

            const size_t len =
                *(reinterpret_cast<const uint32_t*>(cbase + offset));
            offset += sizeof(uint32_t);

            if ((offset + len) > byte_size) {
              return false;
            }
            writer->RawValue(cbase + offset, len, rapidjson::kObjectType);
            offset += len;
          }
          writer->EndArray();
        }
        break;
      }

      writer->StartArray();
      const char* cbase = reinterpret_cast<const char*>(base);
      size_t offset = 0;
      for (size_t e = 0; e < element_count; ++e) {
        if ((offset + sizeof(uint32_t)) > byte_size) {
          return false;
        }

        const size_t len = *(reinterpret_cast<const uint32_t*>(cbase + offset));
        offset += sizeof(uint32_t);

        if ((offset + len) > byte_size) {
          return false;
        }
        // Notice: for the enscaped string use escaped version
        if (!is_b64) {
          triton::common::WriteValue(writer, cbase + offset, len);
        } else {
          triton::common::WriteEscapedString(writer, cbase + offset, len);
        }
        offset += len;
      }
      writer->EndArray();
      break;
    }

    case TRITONSERVER_TYPE_INVALID:
      return false;
  }

  return true;  // success
}

bool
WriteOKResponse(
    rapidjson::StringBuffer* buffer, const OCTensorList& tensors,
    const StringList& names, AppRequestInfo* info)
{
  triton::common::SimpleWriter writer_(*buffer);
  triton::common::SimpleWriter* writer = &writer_;
  for (uint32_t idx = 0; idx < tensors.size(); ++idx) {
    auto* t = &tensors[idx];
    size_t byte_size = t->ByteSize();
    const void* base = t->data_ptr();
    if(byte_size > 0 && base==nullptr){
      writer->StartObject();
      WriteKeyValue(writer, "code", 500);
      WriteKeyValue(writer, "message", "err");
      WriteKeyValue(writer, "request_id", info->request_id);
      WriteKeyValue(writer, "elapse", info->elapse);
      WriteKeyValue(writer, "info", "idpserver excute error!");
      writer->EndObject();
      return false;
    }
  }
  writer->StartObject();
  WriteKeyValue(writer, "code", 200);
  WriteKeyValue(writer, "message", "ok");
  WriteKeyValue(writer, "request_id", info->request_id);
  WriteKeyValue(writer, "elapse", info->elapse);
  writer->Key("result");
  writer->StartObject();
  for (uint32_t idx = 0; idx < tensors.size(); ++idx) {
    const char* cname = names[idx].c_str();
    auto* t = &tensors[idx];
    bool is_b64 = t->base64();
    bool is_jsonstr = t->jsonstr();
    size_t byte_size = t->ByteSize();
    const void* base = t->data_ptr();
    TRITONSERVER_DataType datatype = ConvertDataType(t->dtype());
    auto dims = t->shape();
    size_t element_count = 1;
    for (auto& dim : dims) {
      element_count *= dim;
    }
    writer->Key(cname);
    auto ret = WriteDataToJsonExt(
        writer, cname, datatype, base, byte_size, element_count, is_b64, dims,
        is_jsonstr);
    if (!ret) {
      return ret;
    }
  }
  writer->EndObject();
  writer->EndObject();
  return true;
}

}}  // namespace dataelem::alg