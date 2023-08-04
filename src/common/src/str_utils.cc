#include "triton/common/str_utils.h"

namespace dataelem { namespace common {

template <>
bool
safe_lexical_cast(const char* str, bool& dst)
{
  std::string lower(str);
  try {
    dst = boost::lexical_cast<bool>(str);
  }
  catch (...) {
    boost::algorithm::to_lower(lower);
    if (lower == "true") {
      dst = true;
      return true;
    } else if (lower == "false") {
      dst = false;
      return true;
    }
    return false;
  }
  return true;
}

bool
vector_join(
    const std::vector<std::string>& vstr, const std::string connector,
    std::string& output)
{
  output = "";
  if (vstr.size() == 0) {
    return true;
  }
  std::stringstream ss;
  std::string buffer;
  auto iter = vstr.cbegin();
  if (!safe_lexical_cast((*iter).c_str(), buffer)) {
    return false;
  }
  ss.str(buffer);
  iter++;
  for (; iter != vstr.cend(); iter++) {
    if (!safe_lexical_cast((*iter).c_str(), buffer)) {
      return false;
    }
    ss << connector << buffer;
  }

  output = ss.str();
  return true;
}


// reference from tensorflow/core/lib/strings/stringprintf.h
void
Appendv(std::string* dst, const char* format, va_list ap)
{
  static const int kSpaceLength = 1024;
  char space[kSpaceLength];

  // It's possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, kSpaceLength, format, backup_ap);
  va_end(backup_ap);

  if (result < kSpaceLength) {
    if (result >= 0) {
      // Normal case -- everything fit.
      dst->append(space, result);
      return;
    }

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  int length = result + 1;
  char* buf = new char[length];

  // Restore the va_list before we use it again
  va_copy(backup_ap, ap);
  result = vsnprintf(buf, length, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) {
    // It fit
    dst->append(buf, result);
  }
  delete[] buf;
}

std::string
Printf(const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  std::string result;
  Appendv(&result, format, ap);
  va_end(ap);
  return result;
}

void
Appendf(std::string* dst, const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  Appendv(dst, format, ap);
  va_end(ap);
}

}};  // namespace dataelem::common
