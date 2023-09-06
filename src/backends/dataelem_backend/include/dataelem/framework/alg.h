#ifndef DATAELEM_FRAMEWORK_ALG_H_
#define DATAELEM_FRAMEWORK_ALG_H_

#include "dataelem/framework/alg_utils.h"
#include "dataelem/framework/iop.h"

namespace dataelem { namespace alg {

struct StepConfig {
  StepConfig() = default;
  StepConfig(StringList ins, StringList outs)
      : input_names(ins), output_names(outs)
  {
  }

  StringList input_names;
  StringList output_names;
};

class Algorithmer : public IOp {
 public:
  Algorithmer() = default;
  virtual ~Algorithmer() = default;
  virtual TRITONSERVER_Error* init(triton::backend::BackendModel* model_state)
  {
    graph_executor_ =
        std::make_unique<GraphExecutor>(model_state->TritonServer());
    server_ = model_state->TritonServer();
    return UpdateIONames(model_state->ModelConfig());
  };

  virtual std::string Name() { return alg_name_; }
  virtual TRITONSERVER_Error* Execute(AlgRunContext* context) = 0;

 protected:
  virtual TRITONSERVER_Error* GraphExecuateStep(
      AlgRunContext* context, const StringList& input_names,
      const StringList& output_names, const OCTensorList& inputs,
      OCTensorList& outputs);

  virtual TRITONSERVER_Error* GraphExecuate(
      const std::string& graph_name, AlgRunContext* context,
      const StringList& input_names, const StringList& output_names,
      const OCTensorList& inputs,
      std::future<TRITONSERVER_InferenceResponse*>* future);

  virtual TRITONSERVER_Error* DecodeStep(AlgRunContext* context);

 private:
  TRITONSERVER_Error* UpdateIONames(JValue& model_config);

 protected:
  std::string alg_name_;

  StepConfig enc_dec_io_names_;
  StepConfig io_names_;
  StringList optional_inputs_;
  std::unique_ptr<GraphExecutor> graph_executor_;
  StringList graph_names_;
  TRITONSERVER_Server* server_;
};


template <typename T>
void
ignore(T&&)
{
}

}}  // namespace dataelem::alg

#endif  // DATAELEM_FRAMEWORK_ALG_H_