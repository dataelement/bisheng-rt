#ifndef TRT_MODEL_H
#define TRT_MODEL_H

#include "networkContext.h"

#include <string>
#include <iostream>

class Model {

 public:

  Model(samplesCommon::SampleParams& params) : mParams(params) {

  }

  ~Model() {}

  virtual bool build();

  virtual bool loadModel();

  virtual bool saveEngine();

  virtual bool initBuffer() = 0;

  virtual bool initIContext() = 0;

  virtual bool constructNetwork() = 0;

  TrtUniquePtr<IContext> _context;
  samplesCommon::SampleParams mParams;
};

class SampleModel: public Model {

 public:

  SampleModel(samplesCommon::SampleParams& params) : Model(params) {

  }

  ~SampleModel() {}

  virtual bool initBuffer();

  virtual bool initIContext();

  virtual bool infer(int& batch_size,
                     std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs);

  samplesCommon::BufferManager buffers;

};

class SampleModelDynamic : public Model {

 public:

  SampleModelDynamic(samplesCommon::SampleParams& params) : Model(params) {

  }

  ~SampleModelDynamic() {}

  virtual bool initBuffer();

  virtual bool initIContext();

  virtual bool infer(std::vector<void*>& inputs,
                     std::vector<Dims>& dims,
                     std::map<std::string, std::pair<void*, nvinfer1::Dims>>& outputs);

  std::vector<samplesCommon::ManagedBuffer> _inputs_buffers;
  std::vector<samplesCommon::ManagedBuffer> _outputs_buffers;

};

#endif
