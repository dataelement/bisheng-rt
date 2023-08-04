#include "client.h"

#include <chrono>
#include <curlpp/Easy.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "absl/strings/escaping.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "triton/common/cipher/aes.hpp"

namespace license {

typedef rapidjson::Value Value;
typedef rapidjson::Document Document;
#define JSON_TYPE "Content-Type: application/json"

class HttpClient {
 public:
  static void Get(const std::string& url, int& outcode, std::string& outstr)
  {
    curlpp::Cleanup cl;
    std::ostringstream os;
    curlpp::Easy request;
    curlpp::options::WriteStream ws(&os);
    curlpp::options::CustomRequest pr("GET");
    request.setOpt(curlpp::options::Url(url));
    request.setOpt(ws);
    request.setOpt(pr);
    request.setOpt(cURLpp::Options::FollowLocation(true));
    try {
      request.perform();
      outstr = os.str();
      outcode = curlpp::infos::ResponseCode::get(request);
    }
    catch (const std::exception& e) {
      outstr = e.what();
      outcode = 404;
    }
  }

  static void Post(
      const std::string& url, const std::string& jcontent, int& outcode,
      std::string& outstr, const std::string& content_type = JSON_TYPE)
  {
    curlpp::Cleanup cl;
    std::ostringstream os;
    curlpp::Easy request_put;
    curlpp::options::WriteStream ws(&os);
    curlpp::options::CustomRequest pr("POST");
    request_put.setOpt(curlpp::options::Url(url));
    request_put.setOpt(ws);
    request_put.setOpt(pr);
    std::list<std::string> header;
    header.push_back(content_type);
    request_put.setOpt(curlpp::options::HttpHeader(header));
    request_put.setOpt(curlpp::options::PostFields(jcontent));
    request_put.setOpt(curlpp::options::PostFieldSize(jcontent.length()));
    try {
      request_put.perform();
      outstr = os.str();
      outcode = curlpp::infos::ResponseCode::get(request_put);
    }
    catch (const std::exception& e) {
      outstr = e.what();
      outcode = 404;
    }
  }
};

bool
EncrypteContent(const std::string& content, std::string* out)
{
  std::string private_key = "a9ad0566-cfb3-4427-8f8f-c672fa2b5fcf";
  cipher::KeyBytes hashKey;
  picosha2::hash256_bytes(private_key, hashKey);
  std::vector<char> buffer(content.begin(), content.end());
  if (cipher::EncryptAES(hashKey, buffer) != 0) {
    return false;
  }
  std::string in(buffer.begin(), buffer.end());
  // return Base64::Encode(in, out);
  *out = absl::Base64Escape(in);
  return true;
}


bool
DecryptContent(const std::string& content, std::string* out)
{
  std::string raw_content;
  absl::Base64Unescape(content, &raw_content);
  // if (!Base64::Decode(content, &raw_content)) {
  //   return false;
  // }
  std::string private_key = "a9ad0566-cfb3-4427-8f8f-c672fa2b5fcf";
  cipher::KeyBytes hashKey;
  picosha2::hash256_bytes(private_key, hashKey);
  std::vector<char> buffer(raw_content.begin(), raw_content.end());
  if (cipher::DecryptAES(hashKey, buffer.size(), buffer) != 0) {
    return false;
  }
  *out = std::string(buffer.begin(), buffer.end());
  return true;
}


template <typename... Args>
std::string
string_format(const std::string& format, Args... args)
{
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

inline int
GetTimestamp()
{
  uint64_t sec = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  return int(sec);
}


bool
LicenseClient::license_read(KeyInfo& info)
{
  std::string ep = url_ + "/keyinfo";
  std::string resp;
  int resp_code;
  HttpClient::Get(ep, resp_code, resp);
  if (resp_code != 200) {
    return false;
  }

  Document doc;
  doc.Parse(resp.c_str());
  if (doc.HasParseError()) {
    return false;
  }
  if (!doc.HasMember("content")) {
    return false;
  }
  Document d;
  d.Parse(doc["content"].GetString());
  if (d.HasParseError()) {
    return false;
  }
  // std::cout << "content:" << doc["content"].GetString() << std::endl;

  info.pc_code = doc["pc_code"].GetString();
  info.expire_date =
      (d["expire_date"].IsNull() ? "null" : d["expire_date"].GetString());
  info.feat_id = d["feat_id"].IsNull() ? -1 : d["feat_id"].GetInt();
  info.qps_quota = d["qps_quota"].IsNull() ? -1 : d["qps_quota"].GetInt();
  info.scene_quota = d["scene_quota"].IsNull() ? -1 : d["scene_quota"].GetInt();
  info.stations = d["stations"].IsNull() ? -1 : d["stations"].GetInt();
  info.gpu_num = d["gpu_num"].IsNull() ? -1 : d["gpu_num"].GetInt();
  info.reserver1 = d["reserver1"].IsNull() ? "" : d["reserver1"].GetString();
  info.reserver2 = d["reserver2"].IsNull() ? "" : d["reserver2"].GetString();
  info.reserver3 = d["reserver3"].IsNull() ? "" : d["reserver3"].GetString();
  return true;
}

bool
LicenseClient::license_login(const Conn& conn)
{
  Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();
  auto req_timestamp = GetTimestamp();
  doc.AddMember(
      "machine", Value().SetString(conn.machine.c_str(), conn.machine.size()),
      alloc);
  doc.AddMember("timestamp", Value(req_timestamp), alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  std::string raw_content = buffer.GetString();
  std::string enc_content;
  EncrypteContent(raw_content, &enc_content);
  std::string req_data =
      string_format("{\"content\":\"%s\"}", enc_content.c_str());

  std::string ep = url_ + "/login";
  std::string resp;
  int resp_code;
  HttpClient::Post(ep, req_data, resp_code, resp);
  if (resp_code != 200) {
    return false;
  }

  Document docResp;
  docResp.Parse(resp.c_str());
  if (docResp.HasParseError()) {
    return false;
  }
  if (docResp["code"].GetInt() != 200) {
    return false;
  }
  if (!docResp.HasMember("encr_message")) {
    return false;
  }

  std::string enc_message = docResp["encr_message"].GetString();
  std::string raw_message;
  DecryptContent(enc_message, &raw_message);
  Document docMsg;
  docMsg.Parse(raw_message.c_str());
  if (docMsg.HasParseError()) {
    return false;
  }
  if (!docMsg.HasMember("timestamp")) {
    return false;
  }
  auto ts_diff = docMsg["timestamp"].GetInt() - req_timestamp;
  return ts_diff >= 1024 && ts_diff < 2048;
}

bool
LicenseClient::license_logout(const Conn& conn)
{
  rapidjson::Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();
  doc.AddMember(
      "machine", Value().SetString(conn.machine.c_str(), conn.machine.size()),
      alloc);
  doc.AddMember("timestamp", Value(GetTimestamp()), alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  std::string raw_content = buffer.GetString();
  std::string enc_content;
  EncrypteContent(raw_content, &enc_content);
  std::string req_data =
      string_format("{\"content\":\"%s\"}", enc_content.c_str());

  std::string ep = url_ + "/logout";
  std::string resp;
  int resp_code;
  HttpClient::Post(ep, req_data, resp_code, resp);
  return resp_code == 200;
}

bool
LicenseClient::license_heart(const Conn& conn)
{
  rapidjson::Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();
  doc.AddMember(
      "machine", Value().SetString(conn.machine.c_str(), conn.machine.size()),
      alloc);
  doc.AddMember("timestamp", Value(GetTimestamp()), alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  std::string raw_content = buffer.GetString();
  std::string enc_content;
  EncrypteContent(raw_content, &enc_content);
  std::string req_data =
      string_format("{\"content\":\"%s\"}", enc_content.c_str());

  std::string ep = url_ + "/heart";
  std::string resp;
  int resp_code;
  HttpClient::Post(ep, req_data, resp_code, resp);
  return resp_code == 200;
}


bool
LicenseClient::license_update_dyn(
    const std::string& pc_code, const DynamicInfo& info)
{
  auto req_timestamp = GetTimestamp();
  Document wdoc;
  wdoc.SetObject();
  auto& alloc = wdoc.GetAllocator();
  wdoc.AddMember("timestamp", Value(req_timestamp), alloc);
  wdoc.AddMember("value1", Value(info.value1), alloc);
  wdoc.AddMember("value2", Value(info.value2), alloc);
  wdoc.AddMember("value3", Value(info.value3), alloc);
  wdoc.AddMember("value4", Value(info.value4), alloc);
  wdoc.AddMember("value5", Value(info.value5), alloc);
  wdoc.AddMember("value6", Value(info.value6), alloc);
  wdoc.AddMember("value7", Value(info.value7), alloc);
  wdoc.AddMember("value8", Value(info.value8), alloc);
  wdoc.AddMember("value9", Value(info.value9), alloc);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  wdoc.Accept(writer);

  std::string raw_content = buffer.GetString();
  std::string enc_content;
  EncrypteContent(raw_content, &enc_content);

  std::string req_data =
      string_format("{\"content\":\"%s\"}", enc_content.c_str());
  std::string ep = url_ + "/dyninfo/" + pc_code;
  std::string resp;
  int resp_code;
  HttpClient::Post(ep, req_data, resp_code, resp);

  Document docResp;
  docResp.Parse(resp.c_str());
  if (docResp.HasParseError()) {
    return false;
  }
  if (docResp["code"].GetInt() != 200) {
    return false;
  }
  if (!docResp.HasMember("encr_message")) {
    return false;
  }

  std::string enc_message = docResp["encr_message"].GetString();
  std::string raw_message;
  DecryptContent(enc_message, &raw_message);
  Document docMsg;
  docMsg.Parse(raw_message.c_str());
  if (docMsg.HasParseError()) {
    return false;
  }
  if (!docMsg.HasMember("timestamp")) {
    return false;
  }
  auto ts_diff = docMsg["timestamp"].GetInt() - req_timestamp;
  return ts_diff >= 1024 && ts_diff < 2048;
}

bool
LicenseClient::license_read_dyn(const std::string& pc_code, DynamicInfo& info)
{
  std::string ep = url_ + "/dyninfo/" + pc_code;
  std::string resp;
  int resp_code;
  HttpClient::Get(ep, resp_code, resp);
  if (resp_code != 200) {
    return false;
  }

  Document doc;
  doc.Parse(resp.c_str());
  if (doc.HasParseError()) {
    return false;
  }

  // if (!doc.HasMember("encr_message")) { return false; }
  // std::string encr_msg = doc["encr_message"].GetString();
  // std::string raw_msg;
  // DecryptContent(encr_msg, &raw_msg);

  if (!doc.HasMember("message")) {
    return false;
  }
  std::string raw_msg = doc["message"].GetString();

  Document docMsg;
  docMsg.Parse(raw_msg.c_str());
  if (docMsg.HasParseError()) {
    return false;
  }
  if (!docMsg.HasMember("timestamp")) {
    return false;
  }

  info.value1 = docMsg["value1"].GetInt();
  info.value2 = docMsg["value2"].GetInt();
  info.value3 = docMsg["value3"].GetInt();
  info.value4 = docMsg["value4"].GetInt();
  info.value5 = docMsg["value5"].GetInt();
  info.value6 = docMsg["value6"].GetInt();
  info.value7 = docMsg["value7"].GetInt();
  info.value8 = docMsg["value8"].GetInt();
  info.value9 = docMsg["value9"].GetInt();

  return true;
}


}  // namespace license