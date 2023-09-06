#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/xpressive/xpressive.hpp>

#include <stdarg.h>
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <string>


namespace dataelem { namespace common {

class StrUtils {
 public:
  static std::vector<std::string> split(const std::string& s, char delim)
  {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      if (!item.empty()) {
        elems.push_back(item);
      }
    }
    return elems;
  }

  static bool iequals(const std::string& a, const std::string& b)
  {
    unsigned int sz = a.size();
    if (b.size() != sz) {
      return false;
    }
    for (unsigned int i = 0; i < sz; ++i) {
      if (std::tolower(a[i]) != std::tolower(b[i])) {
        return false;
      }
    }
    return true;
  }

  static int my_hardware_concurrency()
  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    return std::count(
        std::istream_iterator<std::string>(cpuinfo),
        std::istream_iterator<std::string>(), std::string("processor"));
  }
};

// safe lexical cast
template <typename T>
bool
safe_lexical_cast(const char* str, T& dst)
{
  try {
    dst = boost::lexical_cast<T>(str);
    return true;
  }
  catch (...) {
    return false;
  }
}

template <>
bool safe_lexical_cast(const char* str, bool& dst);

bool vector_join(
    const std::vector<std::string>& vstr, const std::string connector,
    std::string& output);

// use of snprintf， efficient format
std::string Printf(const char* format, ...);
void Appendf(std::string* dst, const char* format, ...);
void Appendv(std::string* dst, const char* format, va_list ap);

// string algo
// part compare: [i][ends,starts]_with[_copy], contains, equals, all
// transform: to_[lower,upper], [replace,erase]_[first]
// category helper: is_[space,alnum,alpha,cntrl,digital,graph,lower,print]
//   [punct,upper,xdigit,any_of], if_from_range
// modify: trim_[left,right,_][_copy]
// lookup: [i]find_[first,lst,nth,head,tail] not self update, no copy functions
// find & update: [i]replace/erase_[first,last,nth,all,head,tail][_copy]
// split & merge: [i]find_all, split, join[_if]
// iter version of split: [find,split]_interator

// regex algo
// grammer: .^?()*+{}[]|
// core class:
//   basic: basic_regrex->[s,c]regrex, match_results->[s,c]match,
//   match/search/replace: regex_match, regrex_search, regrex_replace

// u8 and utf-32 transformation
inline std::u32string
u8_to_u32(const std::string& s)
{
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cv;
  return cv.from_bytes(s);
}

inline std::string
u32_to_u8(const std::u32string& s)
{
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cv;
  return cv.to_bytes(s);
}


inline std::wstring
s2ws(const std::string& str)
{
  if (str.empty()) {
    return L"";
  }
  unsigned len = str.size() + 1;
  setlocale(LC_CTYPE, "en_US.UTF-8");
  std::unique_ptr<wchar_t[]> p(new wchar_t[len]);
  mbstowcs(p.get(), str.c_str(), len);
  std::wstring w_str(p.get());
  return w_str;
}

inline std::string
ws2s(const std::wstring& w_str)
{
  if (w_str.empty()) {
    return "";
  }
  unsigned len = w_str.size() * 4 + 1;
  setlocale(LC_CTYPE, "en_US.UTF-8");
  std::unique_ptr<char[]> p(new char[len]);
  wcstombs(p.get(), w_str.c_str(), len);
  std::string str(p.get());
  return str;
}

template <typename T>
std::basic_string<T>
lowercase(const std::basic_string<T>& s)
{
  std::basic_string<T> s2 = s;
  std::transform(s2.begin(), s2.end(), s2.begin(), tolower);
  return std::move(s2);
}

template <typename T>
std::basic_string<T>
uppercase(const std::basic_string<T>& s)
{
  std::basic_string<T> s2 = s;
  std::transform(s2.begin(), s2.end(), s2.begin(), toupper);
  return std::move(s2);
}

inline void
string_split(
    const std::string& s, std::vector<std::string>& sv, const char flag = ' ')
{
  sv.clear();
  std::istringstream iss(s);
  std::string temp;
  while (std::getline(iss, temp, flag)) {
    sv.push_back(temp);
  }
  return;
}

inline std::string&
ltrim(std::string& str)
{
  auto p = std::find_if(
      str.begin(), str.end(), std::not1(std::ptr_fun<int, int>(std::isspace)));
  str.erase(str.begin(), p);
  return str;
}

inline std::string&
rtrim(std::string& str)
{
  auto p = std::find_if(
      str.rbegin(), str.rend(),
      std::not1(std::ptr_fun<int, int>(std::isspace)));
  str.erase(p.base(), str.end());
  return str;
}

inline std::string&
trim(std::string& str)
{
  ltrim(rtrim(str));
  return str;
}


}}  // namespace dataelem::common
