#ifndef SIMPLE_SERVER_HTTPCLIENT_H
#define SIMPLE_SERVER_HTTPCLIENT_H

#include <curlpp/Easy.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>

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

#endif  // SIMPLE_SERVER_HTTPCLIENT_H