#include "sampleTrtLayers.h"

ITensor* Padding(TrtUniquePtr<IContext>& ctx, ITensor* inputs, Dims& prePadding, Dims& postPadding) {
  nvinfer1::IPaddingLayer* layer = ctx->getNetWorkDefine()->addPaddingNd(*inputs, prePadding, postPadding);
  assert(layer);
  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* MaxPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    int kernel_size, int stride, nvinfer1::PaddingMode mode) {
  nvinfer1::IPoolingLayer* layer = ctx->getNetWorkDefine()->addPoolingNd(*inputs, nvinfer1::PoolingType::kMAX,
                                   DimsHW{kernel_size,kernel_size});
  assert(layer);
  layer->setStride(DimsHW{stride, stride});
  layer->setPaddingMode(mode);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* AvgPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    int kernel_size, int stride, nvinfer1::PaddingMode mode) {
  nvinfer1::IPoolingLayer* layer = ctx->getNetWorkDefine()->addPoolingNd(*inputs, nvinfer1::PoolingType::kAVERAGE,
                                   DimsHW{kernel_size,kernel_size});
  assert(layer);
  layer->setStride(DimsHW{stride, stride});
  layer->setPaddingMode(mode);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* AvgPoolingV2(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                      int h_pool_size, int w_pool_size, int h_stride,
                      int w_stride, nvinfer1::PaddingMode mode) {
  nvinfer1::IPoolingLayer* layer = ctx->getNetWorkDefine()->addPoolingNd(*inputs, nvinfer1::PoolingType::kAVERAGE,
                                   DimsHW{h_pool_size, w_pool_size});
  assert(layer);
  layer->setStride(DimsHW{h_stride, w_stride});
  layer->setPaddingMode(mode);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Softmax(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int axis) {
  ISoftMaxLayer* layer = ctx->getNetWorkDefine()->addSoftMax(*inputs);
  assert(layer);
  layer->setAxes(1<<axis);
  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Conv2dTranspose(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int kernel_size,
                         int stride, nvinfer1::PaddingMode mode,
                         ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )) {
  std::string kernel_name = scope.subIScope("kernel").getOpName();
  std::string bias_name = scope.subIScope("bias").getOpName();
  nvinfer1::Weights kernel_weights = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights bias_weights = ctx->getWeightsByName(bias_name);
  nvinfer1::IDeconvolutionLayer* layer = ctx->getNetWorkDefine()->addDeconvolutionNd(*inputs, filters,
                                         Dims{2, {kernel_size, kernel_size}},
                                         kernel_weights, bias_weights);
  assert(layer);
  layer->setStride(DimsHW{stride, stride});
  layer->setPaddingMode(mode);
  nvinfer1::ITensor* outputs = layer->getOutput(0);

  if (activation) {
    outputs = activation(ctx, scope, outputs);
  }
  return outputs;
}

ITensor* Activation(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::ActivationType op) {
  nvinfer1::IActivationLayer* layer = ctx->getNetWorkDefine()->addActivation(*inputs, op);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Subsample(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int factor) {
  if(factor == 1)
    return inputs;
  return MaxPooling(ctx, inputs, 1, factor);
}

ITensor* Concat(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::ITensor*> tensors, int axis) {
  IConcatenationLayer* layer = ctx->getNetWorkDefine()->addConcatenation(tensors.data(), tensors.size());
  assert(layer);
  layer->setAxis(axis);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* ElementWise(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::ITensor*> tensors,
                     nvinfer1::ElementWiseOperation op) {
  nvinfer1::ITensor* outputs = tensors[0];
  for (size_t i = 1; i < tensors.size(); i++) {
    nvinfer1::ITensor* tensor = tensors[i];
    IElementWiseLayer* layer = ctx->getNetWorkDefine()->addElementWise(*outputs, *tensor, op);
    assert(layer);
    outputs = layer->getOutput(0);
  }

  return outputs;
}

ITensor* Resize(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int factor, nvinfer1::ResizeMode mode) {
  nvinfer1::IResizeLayer* layer = ctx->getNetWorkDefine()->addResize(*inputs);
  assert(layer);
  std::vector<float> scales(4);
  scales[0] = 1.0;
  scales[1] = 1.0;
  scales[2] = (float) factor;
  scales[3] = (float) factor;
  nvinfer1::Weights scale_weights = ctx->createTempWeights<float>(scales);

  layer->setResizeMode(mode);
  layer->setScales((float *)scale_weights.values, 4);

  ITensor* outputs = layer->getOutput(0);
  return outputs;
}


ITensor* BatchNorm(TrtUniquePtr<IContext>& ctx,
                   IScope& scope,
                   ITensor* inputs,
                   int axis,
                   float eps) {
  std::string beta_name = scope.subIScope("beta").getOpName();
  std::string gamma_name = scope.subIScope("gamma").getOpName();
  std::string moving_mean_name = scope.subIScope("moving_mean").getOpName();
  std::string moving_variance_name = scope.subIScope("moving_variance").getOpName();
  nvinfer1::Weights beta = ctx->getWeightsByName(beta_name);
  nvinfer1::Weights gamma = ctx->getWeightsByName(gamma_name);
  nvinfer1::Weights moving_mean = ctx->getWeightsByName(moving_mean_name);
  nvinfer1::Weights moving_variance = ctx->getWeightsByName(moving_variance_name);
  int nweight = gamma.count;
  std::vector<float> val1(gamma.count);
  nvinfer1::Weights combined_scale_weights = ctx->createTempWeights<float>(val1);

  std::vector<float> val2(beta.count);
  nvinfer1::Weights combined_bias_weights = ctx->createTempWeights<float>(val2);

  for (size_t i = 0; i < nweight; ++i) {
    float scale = (static_cast<float const*>(gamma.values))[i];
    float bias = (static_cast<float const*>(beta.values))[i];
    float mean = (static_cast<float const*>(moving_mean.values))[i];
    float variance = (static_cast<float const*>(moving_variance.values))[i];
    float& combined_scale_ref = const_cast<float*>(static_cast<float const*>(combined_scale_weights.values))[i];
    float& combined_bias_ref = const_cast<float*>(static_cast<float const*>(combined_bias_weights.values))[i];
    combined_scale_ref = scale / sqrtf(variance + eps);
    combined_bias_ref = bias - mean * combined_scale_ref;
  }

  nvinfer1::IScaleLayer* layer = ctx->getNetWorkDefine()->addScaleNd(*inputs, nvinfer1::ScaleMode::kCHANNEL,
                                 combined_bias_weights, combined_scale_weights,
                                 {}, axis);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Scale(TrtUniquePtr<IContext>& ctx, ITensor* inputs, float scale, float shift, int axis) {
  std::vector<float> val1(1);
  val1[0] = scale;
  nvinfer1::Weights scale_weights = ctx->createTempWeights<float>(val1);

  std::vector<float> val2(1);
  val2[0] = shift;
  nvinfer1::Weights shift_weights = ctx->createTempWeights<float>(val2);

  nvinfer1::IScaleLayer* layer = ctx->getNetWorkDefine()->addScaleNd(*inputs, nvinfer1::ScaleMode::kUNIFORM,
                                 shift_weights, scale_weights, {}, axis);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Transpose(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::Permutation const& perm) {
  nvinfer1::IShuffleLayer* layer = ctx->getNetWorkDefine()->addShuffle(*inputs);
  assert(layer);
  layer->setFirstTranspose(perm);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Reshape(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::Dims dims) {
  nvinfer1::IShuffleLayer* layer = ctx->getNetWorkDefine()->addShuffle(*inputs);
  assert(layer);

  layer->setReshapeDimensions(dims);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* FullyConnected(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                        ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )) {
  std::string kernel_name = scope.subIScope("kernel").getOpName();
  std::string bias_name = scope.subIScope("bias").getOpName();
  nvinfer1::Weights kernel_weights = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights bias_weights = ctx->getWeightsByName(bias_name);
  nvinfer1::IFullyConnectedLayer* layer = ctx->getNetWorkDefine()->addFullyConnected(*inputs, filters,
                                          kernel_weights, bias_weights);
  assert(layer);

  nvinfer1::ITensor* outputs = layer->getOutput(0);

  if (activation) {
    outputs = activation(ctx, scope, outputs);
  }
  return outputs;
}

ITensor* Conv2d(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                int kernel_size, int stride, int dilation_rate,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )) {
  std::string kernel_name = scope.subIScope("kernel").getOpName();
  std::string bias_name = scope.subIScope("bias").getOpName();
  nvinfer1::Weights kernel_weights = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights bias_weights = ctx->getWeightsByName(bias_name);
  nvinfer1::IConvolutionLayer* layer = ctx->getNetWorkDefine()->addConvolutionNd(*inputs, filters,
                                       Dims{2, {kernel_size, kernel_size}}, kernel_weights, bias_weights);
  assert(layer);
  layer->setStride(DimsHW{stride, stride});
  layer->setDilation(DimsHW{dilation_rate, dilation_rate});
  if(stride==1) {
    layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    int kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation_rate - 1);
    int pad_total = kernel_size_effective - 1;
    int pad_beg = pad_total / 2;
    int pad_end = pad_total - pad_beg;
    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);

    nvinfer1::Dims beg_padding;
    beg_padding.nbDims = 2;
    beg_padding.d[0] = pad_beg;
    beg_padding.d[1] = pad_beg;

    nvinfer1::Dims end_padding;
    end_padding.nbDims = 2;
    end_padding.d[0] = pad_end;
    end_padding.d[1] = pad_end;

    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
  }

  nvinfer1::ITensor* outputs = layer->getOutput(0);

  if (activation) {
    outputs = activation(ctx, scope, outputs);
  }
  return outputs;
}

ITensor* Conv2dV2(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                int kernel_size, int h_stride, int w_stride, int dilation_rate,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )) {
  std::string kernel_name = scope.subIScope("kernel").getOpName();
  std::string bias_name = scope.subIScope("bias").getOpName();
  nvinfer1::Weights kernel_weights = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights bias_weights = ctx->getWeightsByName(bias_name);
  nvinfer1::IConvolutionLayer* layer = ctx->getNetWorkDefine()->addConvolutionNd(*inputs, filters,
                                       Dims{2, {kernel_size, kernel_size}}, kernel_weights, bias_weights);
  assert(layer);
  layer->setStride(DimsHW{h_stride, w_stride});
  layer->setDilation(DimsHW{dilation_rate, dilation_rate});
  layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  // if(max(h_stride, w_stride) == 1) {
  //   layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  // } else {
  //   int kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation_rate - 1);
  //   int pad_total = kernel_size_effective - 1;
  //   int pad_beg = pad_total / 2;
  //   int pad_end = pad_total - pad_beg;
  //   layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);

  //   nvinfer1::Dims beg_padding;
  //   beg_padding.nbDims = 2;
  //   beg_padding.d[0] = pad_beg;
  //   beg_padding.d[1] = pad_beg;

  //   nvinfer1::Dims end_padding;
  //   end_padding.nbDims = 2;
  //   end_padding.d[0] = pad_end;
  //   end_padding.d[1] = pad_end;

  //   layer->setPrePadding(beg_padding);
  //   layer->setPostPadding(end_padding);
  // }
  nvinfer1::ITensor* outputs = layer->getOutput(0);

  if (activation) {
    outputs = activation(ctx, scope, outputs);
  }
  return outputs;
}

ITensor* GetTensorInt(TrtUniquePtr<IContext>& ctx, int val, int size) {
  auto network = ctx->getNetWorkDefine();
  std::vector<int> v(size);
  for(int i=0; i<size; i++)
    v[i] = val;
  nvinfer1::Weights wt = ctx->createTempWeights<int>(v);
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = size;
  nvinfer1::ITensor* outputs = network->addConstant(dims, wt)->getOutput(0);
  return outputs;
}

ITensor* Tile(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* repeats) {

  auto similar = [](nvinfer1::Dims dims, int val) {
    nvinfer1::Dims d{dims.nbDims, {}};
    d.nbDims = dims.nbDims;
    for(int i=0; i<dims.nbDims; i++)
      d.d[i] = val;
    return d;
  };
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Dims dims = inputs->getDimensions();
  nvinfer1::Dims starts = similar(dims, 0);
  nvinfer1::Dims strides = similar(dims, 1);
  nvinfer1::Dims sizes = similar(dims, 0);
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  shape = ElementWise(ctx, {shape, repeats}, nvinfer1::ElementWiseOperation::kPROD);

  nvinfer1::ISliceLayer* layer = network->addSlice(*inputs, starts, sizes, strides);
  layer->setInput(2, *shape);
  layer->setMode(nvinfer1::SliceMode::kWRAP);
  assert(layer);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, nvinfer1::Weights wt, DataType& type) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = wt.count;

  nvinfer1::ITensor* outputs;
  if (type == DataType::kFLOAT) {
    outputs = network->addConstant(dims, wt)->getOutput(0);
  } else {
    const float* src_ptr = static_cast<const float*>(wt.values);
    std::vector<half> dst_ptr(wt.count);
    for(int i = 0; i < wt.count; i++)
      dst_ptr[i] = (half)src_ptr[i];
    nvinfer1::Weights wt_convert_half = ctx->createTempWeights<half>(dst_ptr);
    outputs = network->addConstant(dims, wt_convert_half)->getOutput(0);
  }
  return outputs;
}


ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::Weights>& wts, DataType& type) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  for(int i = 0; i < wts.size(); i++) {
    dims.d[0] +=  wts[i].count;
  }

  nvinfer1::ITensor* outputs;
  if (type == DataType::kFLOAT) {
    int num = 0;
    std::vector<float> dst_ptr(dims.d[0]);
    for(int i = 0; i < wts.size(); i++) {
      const float* src_ptr = static_cast<const float*>(wts[i].values);
      for(int j = 0; j < wts[i].count; j++)
        dst_ptr[num++] = (float)src_ptr[j];
    }
    nvinfer1::Weights wt_convert = ctx->createTempWeights<float>(dst_ptr);
    outputs = network->addConstant(dims, wt_convert)->getOutput(0);
  } else {
    int num = 0;
    std::vector<half> dst_ptr(dims.d[0]);
    for(int i = 0; i < wts.size(); i++) {
      const float* src_ptr = static_cast<const float*>(wts[i].values);
      for(int j = 0; j < wts[i].count; j++)
        dst_ptr[num++] = (half)src_ptr[j];
    }
    nvinfer1::Weights wt_convert = ctx->createTempWeights<half>(dst_ptr);
    outputs = network->addConstant(dims, wt_convert)->getOutput(0);
  }
  return outputs;
}

ITensor* Range(TrtUniquePtr<IContext>& ctx, ITensor* length) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = 3;
  nvinfer1::IFillLayer* layer = network->addFill(dims, nvinfer1::FillOperation::kLINSPACE);
  assert(layer);
  layer->setInput(0, *length);

  layer->setAlpha(0);
  layer->setBeta(1);
  layer->setOutputType(0, nvinfer1::DataType::kINT32);

  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* ReshapeDynamic(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* shape) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::IShuffleLayer* layer = network->addShuffle(*inputs);
  assert(layer);

  layer->setInput(1, *shape);

  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* ReduceMean(TrtUniquePtr<IContext>& ctx, ITensor* inputs, uint32_t axis, bool keep_dims) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::IReduceLayer* layer = network->addReduce(*inputs, nvinfer1::ReduceOperation::kAVG, axis, keep_dims);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* GlobalAvgPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs, bool keep_dims) {
  inputs = ReduceMean(ctx, inputs, 1<<2, keep_dims);
  inputs = ReduceMean(ctx, inputs, 1<<3, keep_dims);
  return inputs;
}

ITensor* Unary(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::UnaryOperation op) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::IUnaryLayer* layer = network->addUnary(*inputs, op);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Identity(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::DataType dtype) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::IIdentityLayer* layer = network->addIdentity(*inputs);
  assert(layer);
  layer->setOutputType(0, dtype);
  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* Gather(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int index, int axis) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = 1;

  std::vector<int> v(1);
  v[0] = index;
  nvinfer1::Weights wt_index = ctx->createTempWeights<int>(v);
  ITensor* indice = network->addConstant(dims, wt_index)->getOutput(0);

  nvinfer1::IGatherLayer* layer = network->addGather(*inputs, *indice, axis);
  assert(layer);
  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* LayerNorm(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int axis, float epsilon) {
  auto ScaleChannel = [](TrtUniquePtr<IContext>& ctx, ITensor* inputs,

  nvinfer1::Weights wt_scale, nvinfer1::Weights wt_bias, int axis) {
    auto network = ctx->getNetWorkDefine();
    nvinfer1::IScaleLayer* layer = network->addScaleNd(*inputs, nvinfer1::ScaleMode::kCHANNEL, wt_bias, wt_scale, {}, axis);
    assert(layer);
    nvinfer1::ITensor* outputs = layer->getOutput(0);
    return outputs;
  };
  std::string kernel_name = scope.subIScope("gamma").getOpName();
  std::string bias_name = scope.subIScope("beta").getOpName();
  nvinfer1::Weights wt_scale = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights wt_bias = ctx->getWeightsByName(bias_name);

  auto network = ctx->getNetWorkDefine();
  ITensor* mean = ReduceMean(ctx, inputs, 1<<axis, true);

  ITensor* variance = ElementWise(ctx, {inputs, mean}, nvinfer1::ElementWiseOperation::kSUB);
  variance = ElementWise(ctx, {variance, variance}, nvinfer1::ElementWiseOperation::kPROD);
  variance = ReduceMean(ctx, variance, 1<<axis, true);

  ITensor* norm_x = ElementWise(ctx, {inputs, mean}, nvinfer1::ElementWiseOperation::kSUB);
  ITensor* scale = Scale(ctx, variance, 1.0, epsilon, axis);
  scale = Unary(ctx, Unary(ctx, scale, nvinfer1::UnaryOperation::kSQRT), nvinfer1::UnaryOperation::kRECIP);
  norm_x = ElementWise(ctx, {norm_x, scale}, nvinfer1::ElementWiseOperation::kPROD);
  ITensor* outputs = ScaleChannel(ctx, norm_x, wt_scale, wt_bias, axis);
  return outputs;
}

ITensor* Conv1d(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int kernel_size_w,
                int kernel_size_h, int stride_w, int stride_h, nvinfer1::PaddingMode mode,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )) {
  std::string kernel_name = scope.subIScope("kernel").getOpName();
  std::string bias_name = scope.subIScope("bias").getOpName();
  nvinfer1::Weights wt_kernel = ctx->getWeightsByName(kernel_name);
  nvinfer1::Weights wt_bias = ctx->getWeightsByName(bias_name);
  auto network = ctx->getNetWorkDefine();
  nvinfer1::IConvolutionLayer* layer = network->addConvolutionNd(*inputs, filters,
                                       Dims{2, {kernel_size_w, kernel_size_h}},
                                       wt_kernel, wt_bias);
  assert(layer);
  layer->setStride(DimsHW{stride_w, stride_h});
  layer->setPaddingMode(mode);
  nvinfer1::ITensor* outputs = layer->getOutput(0);

  if (activation) {
    outputs = activation(ctx, scope, outputs);
  }
  return outputs;
}

ITensor* MatrixMultiply(TrtUniquePtr<IContext>& ctx, ITensor* inputs0, ITensor* inputs1, bool is_transpose) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::MatrixOperation op0 = nvinfer1::MatrixOperation::kNONE;
  nvinfer1::MatrixOperation op1 = nvinfer1::MatrixOperation::kNONE;
  if(is_transpose)
    op1 = nvinfer1::MatrixOperation::kTRANSPOSE;
  nvinfer1::IMatrixMultiplyLayer* layer = network->addMatrixMultiply(*inputs0, op0, *inputs1, op1);
  assert(layer);
  ITensor* outputs = layer->getOutput(0);
  return outputs;
}

