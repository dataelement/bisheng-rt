#include <assert.h>
#include <stdio.h>

#include "dataelem/common/status.h"

namespace dataelem { namespace alg {

Status::Status(errors::Code code, absl::string_view msg)
{
  assert(code != errors::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
}

void
Status::Update(const Status& new_status)
{
  if (ok()) {
    *this = new_status;
  }
}

void
Status::SlowCopyFrom(const State* src)
{
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string&
Status::empty_string()
{
  static std::string* empty = new std::string;
  return *empty;
}

std::string
Status::ToString() const
{
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case errors::CANCELLED:
        type = "Cancelled";
        break;
      case errors::UNKNOWN:
        type = "Unknown";
        break;
      case errors::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case errors::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case errors::NOT_FOUND:
        type = "Not found";
        break;
      case errors::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case errors::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case errors::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case errors::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case errors::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case errors::ABORTED:
        type = "Aborted";
        break;
      case errors::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case errors::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case errors::INTERNAL:
        type = "Internal";
        break;
      case errors::UNAVAILABLE:
        type = "Unavailable";
        break;
      case errors::DATA_LOSS:
        type = "Data loss";
        break;
      default:
        snprintf(
            tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code()));
        type = tmp;
        break;
    }
    std::string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void
Status::IgnoreError() const
{
  // no-op
}

std::ostream&
operator<<(std::ostream& os, const Status& x)
{
  os << x.ToString();
  return os;
}

}}  // namespace dataelem::alg
