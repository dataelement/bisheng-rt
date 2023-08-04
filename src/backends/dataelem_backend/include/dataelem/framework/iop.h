#ifndef DATAELEM_FRAMEWORK_IOP_H_
#define DATAELEM_FRAMEWORK_IOP_H_

#include "dataelem/framework/alg_context.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace dataelem { namespace alg {
class IOp {
 public:
  IOp() = default;
  virtual ~IOp() = default;
  virtual TRITONSERVER_Error* init(triton::backend::BackendModel*) = 0;
  virtual std::string Name() = 0;
  virtual TRITONSERVER_Error* Execute(AlgRunContext* context) = 0;
};

class IApp {
 public:
  IApp() = default;
  virtual ~IApp() = default;
  virtual TRITONSERVER_Error* init(triton::backend::BackendModel*) = 0;
  virtual std::string Name() = 0;
  virtual TRITONSERVER_Error* Execute(
      AlgRunContext* context, std::string* resp) = 0;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_IOP_H_
