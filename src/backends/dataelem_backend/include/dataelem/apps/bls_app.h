#ifndef DATAELEM_APPS_BLS_APP_H_
#define DATAELEM_APPS_BLS_APP_H_

#include "dataelem/framework/app.h"

namespace dataelem { namespace alg {


class BLSApp : public Application {
 public:
  BLSApp() = default;
  ~BLSApp() = default;

  TRITONSERVER_Error* init(triton::backend::BackendModel* model_state)
  {
    Application::init(model_state);
    graph_names_ = {"addsub_python", "addsub_tf"};
    app_name_ = "BLSApp";
    return nullptr;
  };

  virtual TRITONSERVER_Error* Execute(AlgRunContext* context, std::string* r);

 protected:
  StringList graph_names_;
};

}}  // namespace dataelem::alg

#endif  // DATAELEM_APPS_BLS_APP_H_
