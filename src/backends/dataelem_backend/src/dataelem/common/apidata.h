#ifndef DATAELEM_COMMON_APIDATA_H_
#define DATAELEM_COMMON_APIDATA_H_

#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <opencv2/opencv.hpp>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "ext/mapbox/variant.hpp"
#include "ext/rmustache/mustache.h"
#include "triton/common/local_filesystem.h"


namespace dataelem { namespace alg {

typedef rapidjson::Document JDoc;
typedef rapidjson::Value JVal;

class APIData;

// recursive variant container,
//   see utils/variant.hpp and utils/recursive_wrapper.hpp
typedef mapbox::util::variant<
    std::string, double, float, int, bool, std::vector<std::string>,
    std::vector<double>, std::vector<float>, std::vector<int>,
    std::vector<bool>, cv::Mat, std::vector<cv::Mat>,
    mapbox::util::recursive_wrapper<APIData>,
    mapbox::util::recursive_wrapper<std::vector<APIData>>>
    ad_variant_type;

/**
 * \brief data conversion exception
 */
class DataConversionException : public std::exception {
 public:
  DataConversionException(const std::string& s) : _s(s) {}
  ~DataConversionException() {}
  const char* what() const noexcept { return _s.c_str(); }

 private:
  std::string _s;
};

/**
 * \brief object for visitor output
 */
class vout {
 public:
  vout() {}
  vout(const APIData& ad) { _vad.push_back(ad); }
  vout(const std::vector<APIData>& vad) : _vad(vad) {}
  ~vout() {}
  std::vector<APIData> _vad;
};

/**
 * \brief visitor class for easy access to variant vector container
 */
class visitor_vad : public mapbox::util::static_visitor<vout> {
 public:
  visitor_vad() {}
  ~visitor_vad(){};
  vout process(const std::string& str);
  vout process(const double& d);
  vout process(const float& f);
  vout process(const int& i);
  vout process(const bool& b);
  vout process(const std::vector<double>& vd);
  vout process(const std::vector<float>& vd);
  vout process(const std::vector<int>& vd);
  vout process(const std::vector<bool>& vd);
  vout process(const std::vector<std::string>& vs);
  vout process(const APIData& ad);
  vout process(const std::vector<APIData>& vad);
  vout process(const cv::Mat& m);
  vout process(const std::vector<cv::Mat>& vm);

  template <typename T>
  vout operator()(const T& t)
  {
    return process(t);
  }
};

/**
 * \brief main deepdetect API data object, uses recursive variant types
 */
class APIData {
 public:
  /**
   * \brief empty constructor
   */
  APIData() {}

  /**
   * \brief constructor from rapidjson JSON object, see dd_types.h
   */
  APIData(const JVal& jval);

  APIData(const APIData& ad) : _data(ad._data) {}

  /**
   * \brief destructor
   */
  ~APIData() {}

  /**
   * \brief add key / object to data object
   * @param key string unique key
   * @param val variant value
   */
  inline void add(const std::string& key, const ad_variant_type& val)
  {
    auto hit = _data.begin();
    if ((hit = _data.find(key)) != _data.end()) {
      _data.erase(hit);
    }
    _data.insert(std::pair<std::string, ad_variant_type>(key, val));
  }

  /**
   * \brief erase key / object from data object
   * @param key string unique key
   */
  inline void erase(const std::string& key)
  {
    auto hit = _data.begin();
    if ((hit = _data.find(key)) != _data.end()) {
      _data.erase(hit);
    }
  }

  /**
   * \brief get value from data object
   *        at this stage, type of value is unknown and the typed object
   *        must be later acquired with e.g. 'get<std::string>(val)
   * @param key string unique key
   * @return variant value
   */
  inline ad_variant_type get(const std::string& key) const
  {
    std::unordered_map<std::string, ad_variant_type>::const_iterator hit;
    if ((hit = _data.find(key)) != _data.end()) {
      return (*hit).second;
    } else {
      return "";  // beware
    }
  }

  /**
   * \brief get vector container as variant value
   * @param key string unique value
   * @return vector of APIData as recursive variant value object
   */
  inline std::vector<APIData> getv(const std::string& key) const
  {
    visitor_vad vv;
    vout v = mapbox::util::apply_visitor(vv, get(key));
    return v._vad;
  }

  /**
   * \brief get data object value as variant value
   * @param key string unique value
   * @return APIData as recursive variant value object
   */
  inline APIData getobj(const std::string& key) const
  {
    visitor_vad vv;
    vout v = mapbox::util::apply_visitor(vv, get(key));
    if (v._vad.empty()) {
      return APIData();
    }
    return v._vad.at(0);
  }

  /**
   * \brief find APIData object from vector, and that has a given key
   * @param vad vector of objects to search
   * @param key string unique key to look for
   * @return APIData as recursive variant value object
   */
  static APIData findv(const std::vector<APIData>& vad, const std::string& key)
  {
    for (const APIData& v : vad) {
      if (v.has(key)) {
        return v;
      }
    }
    return APIData();
  }

  /**
   * \brief test whether the object contains a key at first level
   * @param key string unique key to look for
   * @return true if key is present, false otherwise
   */
  inline bool has(const std::string& key) const
  {
    std::unordered_map<std::string, ad_variant_type>::const_iterator hit;
    if ((hit = _data.find(key)) != _data.end()) {
      return true;
    } else {
      return false;
    }
  }

  std::vector<std::string> list_keys() const
  {
    std::vector<std::string> keys;
    for (auto kv : _data) {
      keys.push_back(kv.first);
    }
    return keys;
  }

  /**
   * \brief number of hosted keys at this level of the object
   * @return size
   */
  inline size_t size() const { return _data.size(); }

  // convert in and out from json.
  /**
   * \brief converts rapidjson JSON to APIData
   * @param jval JSON object
   */
  void fromJVal(const JVal& jval);

  /**
   * \brief converts APIData to rapidjson JSON document
   * @param jd destination JSON Document
   */
  void toJDoc(JDoc& jd) const;

  /**
   * \brief converts APIData to rapidjson JSON value
   * @param jd JSON Document hosting the destination JSON value
   * @param jval destination JSON value
   */
  void toJVal(JDoc& jd, JVal& jv) const;

 public:
  /**
   * \brief render Mustache template based on this APIData object
   * @param tp template string
   */
  inline std::string render_template(const std::string& tpl)
  {
    std::stringstream ss;
    JDoc d;
    d.SetObject();
    toJDoc(d);

    /*rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);
    std::string reststring = buffer.GetString();
    std::cout << "to jdoc=" << reststring << std::endl;*/

    mustache::RenderTemplate(tpl, "", d, &ss);
    return ss.str();
  }

  inline std::string to_str() const
  {
    std::stringstream ss;
    JDoc d;
    d.SetObject();
    toJDoc(d);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);
    std::string reststring = buffer.GetString();
    return reststring;
  }

  inline bool empty() const { return _data.empty(); }
  /**< data as hashtable of variant types. */
  std::unordered_map<std::string, ad_variant_type> _data;
};

/**
 * \brief visitor class for conversion to JSON
 */
class visitor_rjson : public mapbox::util::static_visitor<> {
 public:
  visitor_rjson(JDoc* jd) : _jd(jd) {}
  visitor_rjson(JDoc* jd, JVal* jv) : _jd(jd), _jv(jv) {}
  visitor_rjson(const visitor_rjson& vrj) : _jd(vrj._jd), _jv(vrj._jv)
  {
    _jvkey.CopyFrom(vrj._jvkey, _jd->GetAllocator());
  }
  ~visitor_rjson() {}

  void set_key(const std::string& key)
  {
    _jvkey.SetString(key.c_str(), _jd->GetAllocator());
  }

  void process(const std::string& str)
  {
    if (!_jv) {
      _jd->AddMember(
          _jvkey, JVal().SetString(str.c_str(), _jd->GetAllocator()),
          _jd->GetAllocator());
    } else {
      _jv->AddMember(
          _jvkey, JVal().SetString(str.c_str(), _jd->GetAllocator()),
          _jd->GetAllocator());
    }
  }

  void process(const int& i)
  {
    if (!_jv) {
      _jd->AddMember(_jvkey, JVal(i), _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, JVal(i), _jd->GetAllocator());
    }
  }

  void process(const double& d)
  {
    if (!_jv) {
      _jd->AddMember(_jvkey, JVal(d), _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, JVal(d), _jd->GetAllocator());
    }
  }

  void process(const float& d)
  {
    if (!_jv) {
      _jd->AddMember(_jvkey, JVal(d), _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, JVal(d), _jd->GetAllocator());
    }
  }

  void process(const bool& b)
  {
    if (!_jv) {
      _jd->AddMember(_jvkey, JVal(b), _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, JVal(b), _jd->GetAllocator());
    }
  }

  void process(const APIData& ad)
  {
    JVal jv(rapidjson::kObjectType);
    visitor_rjson vrj(_jd, &jv);
    auto hit = ad._data.begin();
    while (hit != ad._data.end()) {
      vrj.set_key((*hit).first);
      mapbox::util::apply_visitor(vrj, (*hit).second);
      ++hit;
    }

    if (!_jv) {
      _jd->AddMember(_jvkey, jv, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jv, _jd->GetAllocator());
    }
  }

  void process(const std::vector<double>& vd)
  {
    JVal jarr(rapidjson::kArrayType);
    for (size_t i = 0; i < vd.size(); i++) {
      jarr.PushBack(JVal(vd.at(i)), _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }


  void process(const std::vector<float>& vd)
  {
    JVal jarr(rapidjson::kArrayType);
    for (size_t i = 0; i < vd.size(); i++) {
      jarr.PushBack(JVal(vd.at(i)), _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }

  void process(const std::vector<int>& vd)
  {
    JVal jarr(rapidjson::kArrayType);
    for (size_t i = 0; i < vd.size(); i++) {
      jarr.PushBack(JVal(vd.at(i)), _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }

  void process(const std::vector<bool>& vd)
  {
    JVal jarr(rapidjson::kArrayType);
    for (size_t i = 0; i < vd.size(); i++) {
      jarr.PushBack(JVal(vd.at(i)), _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }

  void process(const std::vector<std::string>& vs)
  {
    JVal jarr(rapidjson::kArrayType);
    for (size_t i = 0; i < vs.size(); i++) {
      jarr.PushBack(
          JVal().SetString(vs.at(i).c_str(), _jd->GetAllocator()),
          _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }

  void process(const std::vector<APIData>& vad)
  {
    JVal jov(rapidjson::kObjectType);
    jov = JVal(rapidjson::kArrayType);
    for (size_t i = 0; i < vad.size(); i++) {
      JVal jv(rapidjson::kObjectType);
      visitor_rjson vrj(_jd, &jv);
      APIData ad = vad.at(i);
      auto hit = ad._data.begin();
      while (hit != ad._data.end()) {
        vrj.set_key((*hit).first);
        mapbox::util::apply_visitor(vrj, (*hit).second);
        ++hit;
      }
      jov.PushBack(jv, _jd->GetAllocator());
    }
    if (!_jv) {
      _jd->AddMember(_jvkey, jov, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jov, _jd->GetAllocator());
    }
  }

  void process(const cv::Mat& m)
  {
    // support mat depth: CV_8U,CV_32S,CV_32F,CV_64F
    // assure the data of mat is continuous
    cv::Mat mat1 = m.reshape(1);
    cv::Mat mat2;
    bool is_int = true;
    auto type = mat1.depth();
    if (type == CV_8U) {
      mat1.convertTo(mat2, CV_32S);
    } else if (type == CV_32F) {
      mat1.convertTo(mat2, CV_64F);
      is_int = false;
    } else if (type == CV_64F) {
      is_int = false;
    } else {
      mat2 = mat1;
    }

    JVal jarr(rapidjson::kArrayType);
    if (is_int) {
      cv::Mat_<int> data(mat2);
      for (int i = 0; i < data.rows; i++) {
        JVal jrow(rapidjson::kArrayType);
        for (int j = 0; j < data.cols; j++) {
          jrow.PushBack(JVal(data(i, j)), _jd->GetAllocator());
        }
        jarr.PushBack(jrow, _jd->GetAllocator());
      }
    } else {
      cv::Mat_<double> data(mat2);
      for (int i = 0; i < data.rows; i++) {
        JVal jrow(rapidjson::kArrayType);
        for (int j = 0; j < data.cols; j++) {
          jrow.PushBack(JVal(data(i, j)), _jd->GetAllocator());
        }
        jarr.PushBack(jrow, _jd->GetAllocator());
      }
    }

    if (!_jv) {
      _jd->AddMember(_jvkey, jarr, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jarr, _jd->GetAllocator());
    }
  }

  // process for opencv mats
  void process(const std::vector<cv::Mat>& mats)
  {
    JVal jarr2(rapidjson::kArrayType);
    for (const auto& m : mats) {
      cv::Mat mat1 = m.reshape(1);
      bool is_int = true;
      cv::Mat mat2;
      auto type = mat1.depth();
      if (type == CV_8U) {
        mat1.convertTo(mat2, CV_32S);
      } else if (type == CV_32F) {
        mat1.convertTo(mat2, CV_64F);
        is_int = false;
      } else if (type == CV_64F) {
        is_int = false;
      } else {
        mat2 = mat1;
      }

      JVal jarr(rapidjson::kArrayType);
      if (is_int) {
        cv::Mat_<int> data(mat2);
        for (int i = 0; i < data.rows; i++) {
          JVal jrow(rapidjson::kArrayType);
          for (int j = 0; j < data.cols; j++) {
            jrow.PushBack(JVal(data(i, j)), _jd->GetAllocator());
          }
          jarr.PushBack(jrow, _jd->GetAllocator());
        }
      } else {
        cv::Mat_<double> data(mat2);
        for (int i = 0; i < data.rows; i++) {
          JVal jrow(rapidjson::kArrayType);
          for (int j = 0; j < data.cols; j++) {
            jrow.PushBack(JVal(data(i, j)), _jd->GetAllocator());
          }
          jarr.PushBack(jrow, _jd->GetAllocator());
        }
      }
      jarr2.PushBack(jarr, _jd->GetAllocator());
    }

    JVal jobj(rapidjson::kObjectType);
    jobj.AddMember("arrs", jarr2, _jd->GetAllocator());

    if (!_jv) {
      _jd->AddMember(_jvkey, jobj, _jd->GetAllocator());
    } else {
      _jv->AddMember(_jvkey, jobj, _jd->GetAllocator());
    }
  }

  template <typename T>
  void operator()(T& t)
  {
    process(t);
  }

  JVal _jvkey;
  JDoc* _jd = nullptr;
  JVal* _jv = nullptr;
};

/////////////// APIdata helper functions /////////////////////
template <typename T>
inline void
get_ad_value(
    const APIData& ad, const std::string& key, T& v, const T& default_v)
{
  v = ad.has(key) ? ad.get(key).get<T>() : default_v;
}

template <typename T>
inline void
get_ad_value(const APIData& ad, const std::string& key, T& v)
{
  if (ad.has(key)) {
    v = ad.get(key).get<T>();
  }
}

inline bool
get_mat_from_ad(
    const APIData& ad, const std::string& key, cv::Mat& dst, int depth = CV_32F)
{
  bool ret = false;
  if (ad.has(key)) {
    auto v = ad.get(key);
    cv::Mat m;
    if (v.is<std::vector<int>>()) {
      m = cv::Mat(v.get<std::vector<int>>());
    } else if (v.is<std::vector<double>>()) {
      m = cv::Mat(v.get<std::vector<double>>());
    }
    if (!m.empty()) {
      switch (depth) {
        case CV_32S:
          m.convertTo(dst, CV_32S);
          break;
        case CV_64F:
          m.convertTo(dst, CV_64F);
          break;
        default:
          m.convertTo(dst, CV_32F);
      }
      ret = true;
    }
  }
  return ret;
}

inline APIData
parse_apidata_from_file(const std::string& file)
{
  APIData ad;
  if (!triton::common::FileExists(file)) {
    fprintf(stderr, "file path not exists [%s]", file.c_str());
  } else {
    std::ifstream ifs(file);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();
    ifs.close();
    JDoc jd;
    jd.Parse(content.c_str());
    if (jd.HasParseError()) {
      fprintf(
          stderr, "json parse error: %s (%lu)",
          rapidjson::GetParseError_En(jd.GetParseError()), jd.GetErrorOffset());
    }
    {
      ad.fromJVal(jd);
    }
  }
  return ad;
}

/////////////// rapidjson helper functions /////////////////////
inline JDoc
parse_jdoc_from_file(const std::string& file)
{
  JDoc jd;
  if (!triton::common::FileExists(file)) {
    fprintf(stderr, "file path not exists [%s]", file.c_str());
    return jd;
  }
  std::ifstream ifs(file);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  jd.Parse(buffer.str().c_str());
  if (jd.HasParseError()) {
    fprintf(
        stderr, "json parse error: %s (%lu)",
        rapidjson::GetParseError_En(jd.GetParseError()), jd.GetErrorOffset());
  }
  return jd;
}

template <typename T>
inline void
parse_2dvec_from_jval(
    const rapidjson::Value& v, std::vector<std::vector<T>>& arrs)
{
  for (unsigned int i = 0; i < v.Size(); i++) {
    auto& vv = v[i];
    std::vector<T> arr;
    for (unsigned int j = 0; j < vv.Size(); j++) {
      arr.push_back(vv[j].Get<T>());
    }
    arrs.emplace_back(arr);
  }
}

template <>
inline void
parse_2dvec_from_jval<std::string>(
    const rapidjson::Value& v, std::vector<std::vector<std::string>>& arrs)
{
  for (unsigned int i = 0; i < v.Size(); i++) {
    auto& vv = v[i];
    std::vector<std::string> arr;
    for (unsigned int j = 0; j < vv.Size(); j++) {
      arr.push_back(vv[j].GetString());
    }
    arrs.emplace_back(arr);
  }
}

}}  // namespace dataelem::alg

#endif  // DATAELEM_COMMON_APIDATA_H_
