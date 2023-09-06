#ifndef DATAELEM_APPS_OCR_APP_H_
#define DATAELEM_APPS_OCR_APP_H_

#include "dataelem/framework/app.h"

namespace dataelem { namespace alg {


class OCRApp : public Application {
 public:
  OCRApp() = default;
  ~OCRApp() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state);

  virtual TRITONSERVER_Error* Execute(AlgRunContext* context, std::string* r);

 protected:
  StringList graph_names_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_APPS_OCR_APP_H_
