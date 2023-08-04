#include "trans_ctc_model.h"

ITensor* buildBlock(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int stride,
                    int& conv_i, int& bn_i, int axis, bool is_projection_shortcut) {
  ITensor* shortcut = inputs;
  IScope bn_scope;
  IScope conv_scope;
  if(bn_i == 0) {
    bn_scope = scope.subIScope("batch_normalization");
  } else {
    bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  }
  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;

  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  if(is_projection_shortcut) {
    conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
    shortcut = Conv2d(ctx, conv_scope, inputs, filters, 1, stride, 1);
    conv_i++;
  }

  conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
  inputs = Conv2d(ctx, conv_scope, inputs, filters, 3, stride, 1);
  conv_i++;

  bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;

  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
  inputs = Conv2d(ctx, conv_scope, inputs, filters, 3, 1, 1);
  conv_i++;
  return ElementWise(ctx, {inputs, shortcut}, nvinfer1::ElementWiseOperation::kSUM);
}

ITensor* blockLayer31(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int blocks, int filters,
                      int stride, int& conv_i, int& bn_i, int axis) {
  ITensor* outputs = buildBlock(ctx, scope, inputs, filters, stride, conv_i, bn_i, axis, true);
  for(int i = 1; i < blocks; i++) {
    outputs = buildBlock(ctx, scope, outputs, filters, 1, conv_i, bn_i, axis, false);
  }
  return outputs;
}

ITensor* buildResNet31(TrtUniquePtr<IContext>& ctx, IScope& backbone_scope, ITensor* inputs, int channel_axis) {
  int filters = 16;
  int conv_i = 0;
  int bn_i = 0;

  IScope bn_scope;
  IScope conv_scope;
  conv_scope = backbone_scope.subIScope("conv2d");
  ITensor* outputs = Conv2d(ctx, conv_scope, inputs, filters, 3, 1, 1);
  conv_i++;
  std::vector<int> block_size = {5, 5, 5};
  std::vector<int> block_strides = {1, 2, 2};

  for(int i = 0; i < 3; i++) {
    outputs = blockLayer31(ctx, backbone_scope, outputs, block_size[i], filters, block_strides[i],
                           conv_i, bn_i, channel_axis);
    filters *= 2;
  }

  bn_scope = backbone_scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  outputs = BatchNorm(ctx, bn_scope, outputs, channel_axis, 1e-5f);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);
  return outputs;
}



ITensor* splitHeads(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int hidden_size, int num_heads) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* C = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 2, 0);

  nvinfer1::ITensor* HEADS = GetTensorInt(ctx, num_heads, 1);
  nvinfer1::ITensor* DEPTH = ElementWise(ctx, {C, HEADS}, nvinfer1::ElementWiseOperation::kDIV);
  inputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, HEADS, DEPTH, W}, 0));

  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 1;
  perm.order[2] = 3;
  perm.order[3] = 2;
  inputs = Transpose(ctx, inputs, perm);

  return inputs;
}

ITensor* combineHeads(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int hidden_size) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 1;
  perm.order[2] = 3;
  perm.order[3] = 2;
  inputs = Transpose(ctx, inputs, perm);

  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 3, 0);
  nvinfer1::ITensor* C = GetTensorInt(ctx, hidden_size, 1);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);
  inputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, C, W, ONE}, 0));

  return inputs;
}

ITensor* selfAttention(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* y, ITensor* bias,
                       SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();
  ITensor* x = y;

  IScope conv_scope = scope.subIScope("attention/self/query");
  ITensor* q = Conv2d(ctx, conv_scope, x, params.hidden_size, 1, 1, 1);
  conv_scope = scope.subIScope("attention/self/key");
  ITensor* k = Conv2d(ctx, conv_scope, y, params.hidden_size, 1, 1, 1);
  conv_scope = scope.subIScope("attention/self/value");
  ITensor* v = Conv2d(ctx, conv_scope, y, params.hidden_size, 1, 1, 1);

  q = splitHeads(ctx, q, params.hidden_size, params.num_heads);
  k = splitHeads(ctx, k, params.hidden_size, params.num_heads);
  v = splitHeads(ctx, v, params.hidden_size, params.num_heads);

  float depth = 1.0 / sqrtf(params.hidden_size / params.num_heads);
  q = Scale(ctx, q, depth, {}, 1);

  ITensor* outputs = MatrixMultiply(ctx, q, k, true);

  nvinfer1::ITensor* shape = network->addShape(*outputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* HEADS = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* DEPTH = Gather(ctx, shape, 3, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);

  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 1;
  perm.order[2] = 3;
  perm.order[3] = 2;
  bias = Transpose(ctx, bias, perm);
  bias = Tile(ctx, bias, Concat(ctx, {ONE, HEADS, W, ONE}, 0));
  outputs = ElementWise(ctx, {outputs, bias}, nvinfer1::ElementWiseOperation::kSUM);

  outputs = Softmax(ctx, outputs, 3);

  outputs = MatrixMultiply(ctx, outputs, v, false);
  outputs = combineHeads(ctx, outputs, params.hidden_size);
  conv_scope = scope.subIScope("attention/self/dense");
  outputs = Conv2d(ctx, conv_scope, outputs, params.hidden_size, 1, 1, 1);
  return outputs;
}

ITensor* feedForwardNetWork(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                            ITensor* inputs_padding, SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();

  IScope conv_scope = scope.subIScope("ffn/dense");
  inputs = Conv2d(ctx, conv_scope, inputs, params.filter_size, 1, 1, 1);

  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  conv_scope = scope.subIScope("ffn/dense_1");
  inputs = Conv2d(ctx, conv_scope, inputs, params.hidden_size, 1, 1, 1);

  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* C = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);
  inputs_padding = Tile(ctx, inputs_padding, Concat(ctx, {ONE, C, ONE, ONE}, 0));
  inputs = ElementWise(ctx, {inputs, inputs_padding}, nvinfer1::ElementWiseOperation::kPROD);
  return inputs;
}

ITensor* encoderStack(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* encoder_inputs, ITensor* attention_bias,
                      ITensor* inputs_padding, SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();

  for(int i = 0; i < params.num_layer; i++) {
    IScope layer_scope = scope.subIScope("layer_" + to_string(i));

    ITensor* inputs = encoder_inputs;

    IScope conv_scope = layer_scope.subIScope("attention/self/conv1d");
    encoder_inputs = Conv1d(ctx, conv_scope, encoder_inputs, params.hidden_size, 3, 1, 1, 1, nvinfer1::PaddingMode::kSAME_UPPER);

    encoder_inputs = Activation(ctx, encoder_inputs, nvinfer1::ActivationType::kRELU);

    IScope ln_scope = layer_scope.subIScope("attention/self/LayerNorm");
    encoder_inputs = LayerNorm(ctx, ln_scope, encoder_inputs, params.channel_axis);

    conv_scope = layer_scope.subIScope("attention/self/conv1d_1");
    encoder_inputs = Conv1d(ctx, conv_scope, encoder_inputs, params.hidden_size, 3, 1, 1, 1, nvinfer1::PaddingMode::kSAME_UPPER);

    encoder_inputs = Activation(ctx, encoder_inputs, nvinfer1::ActivationType::kRELU);

    ln_scope = layer_scope.subIScope("attention/self/LayerNorm_1");
    encoder_inputs = LayerNorm(ctx, ln_scope, encoder_inputs, params.channel_axis);

    encoder_inputs = ElementWise(ctx, {encoder_inputs, inputs}, nvinfer1::ElementWiseOperation::kSUM);

    inputs = encoder_inputs;
    ln_scope = layer_scope.subIScope("attention/self/LayerNorm_2");
    encoder_inputs = LayerNorm(ctx, ln_scope, encoder_inputs, params.channel_axis);

    encoder_inputs = selfAttention(ctx, layer_scope, encoder_inputs, attention_bias, params);

    encoder_inputs = ElementWise(ctx, {encoder_inputs, inputs}, nvinfer1::ElementWiseOperation::kSUM);
    inputs = encoder_inputs;

    ln_scope = layer_scope.subIScope("ffn/LayerNorm");
    encoder_inputs = LayerNorm(ctx, ln_scope, encoder_inputs, params.channel_axis);

    encoder_inputs = feedForwardNetWork(ctx, layer_scope, encoder_inputs, inputs_padding, params);

    encoder_inputs = ElementWise(ctx, {encoder_inputs, inputs}, nvinfer1::ElementWiseOperation::kSUM);
  }

  IScope ln_scope = scope.subIScope("LayerNorm");
  return LayerNorm(ctx, ln_scope, encoder_inputs, params.channel_axis);
}

ITensor* getInputsPadding(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* inputs_shape,
                          SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* L = Gather(ctx, inputs_shape, 1, 1);
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);

  nvinfer1::ITensor* DIV = GetTensorInt(ctx, params.downsample, 1);
  DIV = ReshapeDynamic(ctx, DIV, Concat(ctx, {ONE, ONE}, 0));
  DIV = Tile(ctx, DIV, Concat(ctx, {B, ONE}, 0));
  L = ElementWise(ctx, {L, DIV}, nvinfer1::ElementWiseOperation::kDIV);
  L = Tile(ctx, L, Concat(ctx, {ONE, W}, 0));

  nvinfer1::ITensor* indexs = Range(ctx, W);
  indexs = ReshapeDynamic(ctx, indexs, Concat(ctx, {ONE, W}, 0));
  indexs = Tile(ctx, indexs, Concat(ctx, {B, ONE}, 0));

  nvinfer1::ITensor* outputs = ElementWise(ctx, {indexs, L}, nvinfer1::ElementWiseOperation::kLESS);
  outputs = Identity(ctx, outputs, nvinfer1::DataType::kFLOAT);
  //B,W->B,1,W,1
  outputs = ReshapeDynamic(ctx, outputs, Concat(ctx, {B, ONE, W, ONE}, 0));
  return outputs;
}

ITensor* getPaddingBias(TrtUniquePtr<IContext>& ctx, ITensor* inputs, SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();
  float NEG_INF = params.fp16 ? -5e4 : -1e9;
  inputs = Scale(ctx, inputs, -1.0, 1.0, params.channel_axis);
  inputs = Scale(ctx, inputs, NEG_INF, 0.0, params.channel_axis);
  return inputs;
}

ITensor* getPositionEncoding(TrtUniquePtr<IContext>& ctx, ITensor* inputs) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* C = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* position = Range(ctx, W);

  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);
  position = ReshapeDynamic(ctx, position, Concat(ctx, {ONE, W}, 0));
  position = Identity(ctx, position, nvinfer1::DataType::kFLOAT);

  int hidden_size = 512;
  int num_timescales = hidden_size / 2;
  float min_timescale = 1.0;
  float max_timescale = 10000.0;

  float log_timescale_increment = log(max_timescale / min_timescale) / (1.0 * num_timescales - 1);
  std::vector<float> v_inv_timescales(num_timescales);
  for(int i=0; i<num_timescales; i++)
    v_inv_timescales[i] = 1.0 * min_timescale * exp(-1.0 * i * log_timescale_increment);

  nvinfer1::Weights wt_timescales = ctx->createTempWeights<float>(v_inv_timescales);
  nvinfer1::Dims dims;
  dims.nbDims = 2;
  dims.d[0] = num_timescales;
  dims.d[1] = 1;
  nvinfer1::ITensor* inv_timescales = network->addConstant(dims, wt_timescales)->getOutput(0);
  nvinfer1::ITensor* scaled_time = ElementWise(ctx, {inv_timescales, position}, nvinfer1::ElementWiseOperation::kPROD);
  nvinfer1::ITensor* signal = Concat(ctx, {Unary(ctx, scaled_time, nvinfer1::UnaryOperation::kSIN),
                                     Unary(ctx, scaled_time, nvinfer1::UnaryOperation::kCOS)
                                          }, 0);

  shape = network->addShape(*signal)->getOutput(0);

  signal = ReshapeDynamic(ctx, signal, Concat(ctx, {ONE, shape, ONE}, 0));
  signal = Tile(ctx, signal, Concat(ctx, {B, ONE, ONE, ONE}, 0));
  return signal;
}

ITensor* buildTransEncode(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, ITensor* inputs_shape,
                          SampleTrtTransCtcParams& params) {
  auto network = ctx->getNetWorkDefine();

  ITensor* inputs_padding;
  ITensor* attention_bias;
  ITensor* signal;

  inputs_padding = getInputsPadding(ctx, inputs, inputs_shape, params);
  attention_bias = getPaddingBias(ctx, inputs_padding, params);
  signal = getPositionEncoding(ctx, inputs);

  signal = Scale(ctx, signal, -1.0, 0.0, params.channel_axis);
  inputs = ElementWise(ctx, {inputs, signal}, nvinfer1::ElementWiseOperation::kSUB);
  inputs = encoderStack(ctx, scope, inputs, attention_bias, inputs_padding, params);
  return inputs;
}

void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& inputs,
                  SampleTrtTransCtcParams& params,
                  std::vector<ITensor*>& outputs_vec) {
  assert(inputs[0]);
  assert(inputs[1]);

  IScope pre_scope;
  IScope backbone_scope = pre_scope.subIScope("backbone");
  ITensor* outputs = buildResNet31(ctx, backbone_scope, inputs[0], params.channel_axis);

  IScope conv_scope = pre_scope.subIScope("ocr_transformer/post_conv");
  outputs = Conv2d(ctx, conv_scope, outputs, 512, 1, 1, 1);

  IScope bn_scope = pre_scope.subIScope("ocr_transformer/post_bn");
  outputs = BatchNorm(ctx, bn_scope, outputs, params.channel_axis, 1e-3f);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);

  //BxCxHxW
  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 2;
  perm.order[2] = 1;
  perm.order[3] = 3;
  outputs = Transpose(ctx, outputs, perm);

  //B,H,C,W -> B,HxC,W
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*outputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* H = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* C = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 3, 0);

  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);

  C = ElementWise(ctx, {H, C}, nvinfer1::ElementWiseOperation::kPROD);
  outputs = ReshapeDynamic(ctx, outputs, Concat(ctx, {B, C, W, ONE}, 0));

  conv_scope = pre_scope.subIScope("ocr_transformer/dense");
  outputs = Conv2d(ctx, conv_scope, outputs, 512, 1, 1, 1);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);

  outputs = buildTransEncode(ctx, pre_scope, outputs, inputs[1], params);

  conv_scope = pre_scope.subIScope("ocr_ctc/logits");
  outputs = Conv2d(ctx, conv_scope, outputs, 6410, 1, 1, 1);

  shape = network->addShape(*outputs)->getOutput(0);
  B = Gather(ctx, shape, 0, 0);
  C = Gather(ctx, shape, 1, 0);
  W = Gather(ctx, shape, 2, 0);
  outputs = ReshapeDynamic(ctx, outputs, Concat(ctx, {B, C, W}, 0));

  outputs_vec.push_back(outputs);
}

