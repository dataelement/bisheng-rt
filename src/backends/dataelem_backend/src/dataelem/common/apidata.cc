#include "dataelem/common/apidata.h"

namespace dataelem { namespace alg {

/*- visitor_vad -*/
vout
visitor_vad::process(const std::string& str)
{
  (void)str;
  return vout();
}

vout
visitor_vad::process(const float& f)
{
  (void)f;
  return vout();
}

vout
visitor_vad::process(const double& d)
{
  (void)d;
  return vout();
}

vout
visitor_vad::process(const int& i)
{
  (void)i;
  return vout();
}

vout
visitor_vad::process(const bool& b)
{
  (void)b;
  return vout();
}

vout
visitor_vad::process(const APIData& ad)
{
  return vout(ad);
}

vout
visitor_vad::process(const std::vector<float>& vd)
{
  (void)vd;
  return vout();
}

vout
visitor_vad::process(const std::vector<double>& vd)
{
  (void)vd;
  return vout();
}

vout
visitor_vad::process(const std::vector<int>& vd)
{
  (void)vd;
  return vout();
}

vout
visitor_vad::process(const std::vector<bool>& vd)
{
  (void)vd;
  return vout();
}

vout
visitor_vad::process(const std::vector<std::string>& vs)
{
  (void)vs;
  return vout();
}

vout
visitor_vad::process(const std::vector<APIData>& vad)
{
  return vout(vad);
}

vout
visitor_vad::process(const cv::Mat& m)
{
  (void)m;
  return vout();
}

vout
visitor_vad::process(const std::vector<cv::Mat>& vm)
{
  (void)vm;
  return vout();
}

/*- APIData -*/
APIData::APIData(const JVal& jval)
{
  fromJVal(jval);
}

void
APIData::fromJVal(const JVal& jval)
{
  for (rapidjson::Value::ConstMemberIterator cit = jval.MemberBegin();
       cit != jval.MemberEnd(); ++cit) {
    if (cit->value.IsNull()) {
    } else if (cit->value.IsBool()) {
      add(cit->name.GetString(), cit->value.GetBool());
    } else if (cit->value.IsObject()) {
      APIData ad(jval[cit->name.GetString()]);
      std::vector<APIData> vad = {ad};
      add(cit->name.GetString(), vad);
    } else if (cit->value.IsArray()) {
      // only supports array that bears a single type, number, string or object
      const JVal& jarr = jval[cit->name.GetString()];
      if (jarr.Size() != 0) {
        if (jarr[0].IsDouble()) {
          std::vector<double> vd;
          for (rapidjson::SizeType i = 0; i < jarr.Size(); i++) {
            vd.push_back(jarr[i].GetDouble());
          }
          add(cit->name.GetString(), vd);
        } else if (jarr[0].IsInt()) {
          std::vector<int> vd;
          for (rapidjson::SizeType i = 0; i < jarr.Size(); i++) {
            vd.push_back(jarr[i].GetInt());
          }
          add(cit->name.GetString(), vd);
        } else if (jarr[0].IsBool()) {
          std::vector<bool> vd;
          for (rapidjson::SizeType i = 0; i < jarr.Size(); i++) {
            vd.push_back(jarr[i].GetBool());
          }
          add(cit->name.GetString(), vd);
        } else if (jarr[0].IsString()) {
          std::vector<std::string> vs;
          for (rapidjson::SizeType i = 0; i < jarr.Size(); i++) {
            vs.push_back(jarr[i].GetString());
          }
          add(cit->name.GetString(), vs);
        } else if (jarr[0].IsObject()) {
          std::vector<APIData> vad;
          for (rapidjson::SizeType i = 0; i < jarr.Size(); i++) {
            APIData nad;
            nad.fromJVal(jarr[i]);
            vad.push_back(nad);
          }
          add(cit->name.GetString(), vad);
        } else {
          throw DataConversionException(
              "conversion error: unknown type of array");
        }
      }
    } else if (cit->value.IsString()) {
      add(cit->name.GetString(), cit->value.GetString());
    } else if (cit->value.IsDouble()) {
      add(cit->name.GetString(), cit->value.GetDouble());
    } else if (cit->value.IsInt()) {
      add(cit->name.GetString(), cit->value.GetInt());
    }
  }
}

void
APIData::toJDoc(JDoc& jd) const
{
  visitor_rjson vrj(&jd);
  auto hit = _data.begin();
  while (hit != _data.end()) {
    vrj.set_key((*hit).first);
    mapbox::util::apply_visitor(vrj, (*hit).second);
    ++hit;
  }
}

void
APIData::toJVal(JDoc& jd, JVal& jv) const
{
  visitor_rjson vrj(&jd, &jv);
  auto hit = _data.begin();
  while (hit != _data.end()) {
    vrj.set_key((*hit).first);
    mapbox::util::apply_visitor(vrj, (*hit).second);
    ++hit;
  }
}

}}  // namespace dataelem::alg
