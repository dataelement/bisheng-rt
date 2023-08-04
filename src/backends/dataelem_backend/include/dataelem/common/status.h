#ifndef DATAELEM_COMMON_STATUS_H_
#define DATAELEM_COMMON_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"

namespace dataelem { namespace alg {

struct errors {
  enum Code {
    OK,
    CANCELLED,
    UNKNOWN,
    INVALID_ARGUMENT,
    DEADLINE_EXCEEDED,
    NOT_FOUND,
    ALREADY_EXISTS,
    PERMISSION_DENIED,
    UNAUTHENTICATED,
    RESOURCE_EXHAUSTED,
    FAILED_PRECONDITION,
    ABORTED,
    OUT_OF_RANGE,
    UNIMPLEMENTED,
    INTERNAL,
    UNAVAILABLE,
    DATA_LOSS
  };
};

class Status {
 public:
  /// Create a success status.
  Status() {}
  Status(errors::Code code, absl::string_view msg);
  /// Copy the specified status.
  Status(const Status& s);

  void operator=(const Status& s);

  static Status OK() { return Status(); }
  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == NULL); }

  errors::Code code() const { return ok() ? errors::OK : state_->code; }

  const std::string& error_message() const
  {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
  static const std::string& empty_string();
  struct State {
    errors::Code code;
    std::string msg;
  };
  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
};

inline Status::Status(const Status& s)
    : state_((s.state_ == NULL) ? NULL : new State(*s.state_))
{
}

inline void
Status::operator=(const Status& s)
{
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
}

inline bool
Status::operator==(const Status& x) const
{
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}
inline bool
Status::operator!=(const Status& x) const
{
  return !(*this == x);
}

std::ostream& operator<<(std::ostream& os, const Status& x);

}}  // namespace dataelem::alg

#endif  // DATAELEM_COMMON_STATUS_H_
