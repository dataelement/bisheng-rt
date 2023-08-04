#include "transformer_model.h"

ITensor* bottleneckBlock(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int stride,
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
    shortcut = Conv2d(ctx, conv_scope, inputs, 4 * filters, 1, stride, 1);
    conv_i++;
  }

  conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
  inputs = Conv2d(ctx, conv_scope, inputs, filters, 1, 1, 1);
  conv_i++;

  bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
  inputs = Conv2d(ctx, conv_scope, inputs, filters, 3, stride, 1);
  conv_i++;

  bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  conv_scope = scope.subIScope("conv2d_"+std::to_string(conv_i));
  inputs = Conv2d(ctx, conv_scope, inputs, 4 * filters, 1, 1, 1);
  conv_i++;
  return ElementWise(ctx, {inputs, shortcut}, nvinfer1::ElementWiseOperation::kSUM);
}

ITensor* blockLayer50(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int blocks, int filters,
                      int stride, int& conv_i, int& bn_i, int axis) {
  ITensor* outputs = bottleneckBlock(ctx, scope, inputs, filters, stride, conv_i, bn_i, axis, true);
  for(int i = 1; i < blocks; i++) {
    outputs = bottleneckBlock(ctx, scope, outputs, filters, 1, conv_i, bn_i, axis, false);
  }

  return outputs;
}

ITensor* buildResNet50(TrtUniquePtr<IContext>& ctx, IScope& backbone_scope, ITensor* inputs,
                       SampleTrtTransformerParams& params) {
  int filters = 64;
  int conv_stride = 2;
  int kernel_size = 7;
  int dilation = 1;
  int conv_i = 0;
  int bn_i = 0;

  IScope bn_scope;
  IScope conv_scope;
  conv_scope = backbone_scope.subIScope("conv2d");
  ITensor* outputs = Conv2d(ctx, conv_scope, inputs, filters, kernel_size, conv_stride, dilation);
  conv_i++;

  std::vector<int> block_size = {3, 4, 6, 3};
  std::vector<int> block_strides = {1, 2, 2, 1};

  for(int i = 0; i < 4; i++) {
    outputs = blockLayer50(ctx, backbone_scope, outputs, block_size[i], filters, block_strides[i],
                           conv_i, bn_i, params.channel_axis);
    filters *= 2;

  }

  bn_scope = backbone_scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  outputs = BatchNorm(ctx, bn_scope, outputs, params.channel_axis, 1e-5f);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);
  return outputs;
}

ITensor* convBnLayer(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                     int num_filter, int filter_size, int h_stride,
                     int w_stride, int& conv_i, int& bn_i, int axis,
                     std::string activation="relu") {
  IScope bn_scope;
  IScope conv_scope;
  if(conv_i == 0) {
    conv_scope = scope.subIScope("conv2d");
  } else {
    conv_scope = scope.subIScope("conv2d_" + std::to_string(conv_i));
  }
  if(bn_i == 0) {
    bn_scope = scope.subIScope("batch_normalization");
  } else {
    bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  }
  inputs = Conv2dV2(
    ctx, conv_scope, inputs, num_filter, filter_size, h_stride, w_stride, 1);
  conv_i++;

  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;

  if (activation == "relu") {
    inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);
  }
  return inputs;
}

ITensor* convBnLayerNew(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                     int num_filter, int filter_size, int h_stride,
                     int w_stride, int& conv_i, int& bn_i, int axis) {
  inputs = AvgPoolingV2(ctx, inputs, h_stride, w_stride, h_stride, w_stride);
  IScope bn_scope;
  IScope conv_scope;
  if(conv_i == 0) {
    conv_scope = scope.subIScope("conv2d");
  } else {
    conv_scope = scope.subIScope("conv2d_" + std::to_string(conv_i));
  }
  if(bn_i == 0) {
    bn_scope = scope.subIScope("batch_normalization");
  } else {
    bn_scope = scope.subIScope("batch_normalization_" + std::to_string(bn_i));
  }
  inputs = Conv2dV2(
    ctx, conv_scope, inputs, num_filter, filter_size, 1, 1, 1);
  conv_i++;

  inputs = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  bn_i++;

  return inputs;
}

ITensor* shortCut(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                     int num_filter, int h_stride, int w_stride, int& conv_i,
                     int& bn_i, int axis, bool if_first) {
  nvinfer1::Dims dims = inputs->getDimensions();
  int channel = dims.d[1];
  if (channel != num_filter || h_stride != 1) {
    if (if_first) {
      return convBnLayer(
        ctx, scope, inputs, num_filter, 1, h_stride, w_stride, conv_i, bn_i, axis);
    } else {
      return convBnLayerNew(
        ctx, scope, inputs, num_filter, 1, h_stride, w_stride, conv_i, bn_i, axis);
    }
  } else if (if_first) {
    return convBnLayer(
        ctx, scope, inputs, num_filter, 1, h_stride, w_stride, conv_i, bn_i, axis);

  } else {
    return inputs;
  }
}

ITensor* basicBlock(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                     int num_filter, int h_stride, int w_stride, int& conv_i,
                     int& bn_i, int axis, bool if_first) {
  ITensor* outputs = convBnLayer(
    ctx, scope, inputs, num_filter, 3, h_stride, w_stride, conv_i, bn_i, axis);
  outputs = convBnLayer(
    ctx, scope, outputs, num_filter, 3, 1, 1, conv_i, bn_i, axis, "none");
  ITensor* shortcut = shortCut(
    ctx, scope, inputs, num_filter, h_stride, w_stride, conv_i, bn_i, axis, if_first);
  outputs = ElementWise(ctx, {outputs, shortcut}, nvinfer1::ElementWiseOperation::kSUM);
  return Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);
}

ITensor* buildResNetvd(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                       SampleTrtTransformerParams& params) {
  int conv_i = 0;
  int bn_i = 0;

  // module_0
  ITensor* outputs = convBnLayer(
    ctx, scope, inputs, 32, 3, 1, 1, conv_i, bn_i, params.channel_axis);
  outputs = convBnLayer(
    ctx, scope, outputs, 32, 3, 1, 1, conv_i, bn_i, params.channel_axis);
  outputs = convBnLayer(
    ctx, scope, outputs, 64, 3, 1, 1, conv_i, bn_i, params.channel_axis);
  outputs = MaxPooling(ctx, outputs, 3, 2);

  // module_1
  std::vector<int> block_sizes = {3, 4, 6, 3};
  std::vector<int> num_filters = {64, 128, 256, 512};
  for (int block_id = 0; block_id < block_sizes.size(); block_id++) {
    for (int i = 0; i < block_sizes[block_id]; i++) {
      int h_stride = 1;
      int w_stride = 1;
      if (i == 0 && block_id != 0) {
        h_stride = 2;
        w_stride = 1;
      }
      bool if_first = ((block_id == 0) && (i == 0));
      outputs = basicBlock(ctx, scope, outputs, num_filters[block_id],
                            h_stride, w_stride, conv_i, bn_i,
                            params.channel_axis, if_first);
    }
  }
  outputs = MaxPooling(ctx, outputs, 2, 2);
  return outputs;
}

ITensor* getInputsPadding(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* inputs_shape,
                          SampleTrtTransformerParams& params) {
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

  //B,W->B,1,W
  outputs = ReshapeDynamic(ctx, outputs, Concat(ctx, {B, ONE, W}, 0));

  return outputs;
}

ITensor* getAttBias(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* inputs_padding) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);

  ITensor* outputs = Tile(ctx, inputs_padding, Concat(ctx, {ONE, W, ONE}, 0));
  return outputs;
}

ITensor* getFFNBias(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* inputs_padding) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* H = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);

  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 2;
  perm.order[2] = 1;
  ITensor* outputs  = Transpose(ctx, inputs_padding, perm);
  outputs = Tile(ctx, outputs, Concat(ctx, {ONE, ONE, H}, 0));
  return outputs;
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

ITensor* buildTransEncodeOp(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, ITensor* inputs_shape,
                            SampleTrtTransformerParams& params) {
  auto network = ctx->getNetWorkDefine();
  ITensor* inputs_padding = getInputsPadding(ctx, inputs, inputs_shape, params);
  ITensor* attention_bias = getAttBias(ctx, inputs, inputs_padding);;
  ITensor* ffn_padding = getFFNBias(ctx, inputs, inputs_padding);
  ITensor* signal = getPositionEncoding(ctx, inputs);

  signal = Scale(ctx, signal, -1.0, 0.0, params.channel_axis);
  inputs = ElementWise(ctx, {inputs, signal}, nvinfer1::ElementWiseOperation::kSUB);

  nvinfer1::Permutation perm;
  perm.order[0] = 0;
  perm.order[1] = 2;
  perm.order[2] = 1;
  perm.order[3] = 3;
  inputs = Transpose(ctx, inputs, perm);

  ITensor* shape = network->addShape(*inputs)->getOutput(0);
  ITensor* B = Gather(ctx, shape, 0, 0);
  ITensor* L = Gather(ctx, shape, 1, 0);
  ITensor* H = Gather(ctx, shape, 2, 0);
  inputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, L, H}, 0));
  DataType type = params.fp16 ? DataType::kHALF : DataType::kFLOAT;
  for(int i = 0; i < params.num_layer; i++) {
    IScope attention_scope = scope.subIScope("layer_"+to_string(i)+"/attention/self");
    IScope ffn_scope = scope.subIScope("layer_"+to_string(i)+"/ffn");
    std::vector<IScope> scopes = {attention_scope.subIScope("conv1d/kernel"),
                                  attention_scope.subIScope("conv1d/bias"),
                                  attention_scope.subIScope("LayerNorm/gamma"),
                                  attention_scope.subIScope("LayerNorm/beta"),
                                  attention_scope.subIScope("conv1d_1/kernel"),
                                  attention_scope.subIScope("conv1d_1/bias"),
                                  attention_scope.subIScope("LayerNorm_1/gamma"),
                                  attention_scope.subIScope("LayerNorm_1/beta"),
                                  attention_scope.subIScope("LayerNorm_2/gamma"),
                                  attention_scope.subIScope("LayerNorm_2/beta"),
                                  attention_scope.subIScope("query/kernel"),
                                  attention_scope.subIScope("key/kernel"),
                                  attention_scope.subIScope("value/kernel"),
                                  attention_scope.subIScope("dense/kernel"),
                                  ffn_scope.subIScope("LayerNorm/gamma"),
                                  ffn_scope.subIScope("LayerNorm/beta"),
                                  ffn_scope.subIScope("dense/kernel"),
                                  ffn_scope.subIScope("dense/bias"),
                                  ffn_scope.subIScope("dense_1/kernel"),
                                  ffn_scope.subIScope("dense_1/bias"),
                                  scope.subIScope("LayerNorm/gamma"),
                                  scope.subIScope("LayerNorm/beta")
                                 };
    std::vector<ITensor*> inputs_list = {inputs, attention_bias, ffn_padding};
    for (unsigned int j = 0; j < scopes.size(); j++) {
      ITensor* weight = GetConstTensor(ctx, ctx->getWeightsByName(scopes[j].getOpName()), type);
      inputs_list.push_back(weight);
    }
    auto creator = getPluginRegistry()->getPluginCreator("TransformerEncodePluginDynamic", "1");
    DataType type_id = type;
    int isLastLayer = int(i==params.num_layer-1);
    std::vector<PluginField> layerFields;
    layerFields.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32, 1));
    layerFields.emplace_back(PluginField("hidden_size", &params.hidden_size, PluginFieldType::kINT32, 1));
    layerFields.emplace_back(PluginField("num_heads", &params.num_heads, PluginFieldType::kINT32, 1));
    layerFields.emplace_back(PluginField("isLastLayer", &isLastLayer, PluginFieldType::kINT32, 1));
    PluginFieldCollection pluginEncodeData;
    pluginEncodeData.nbFields = layerFields.size();
    pluginEncodeData.fields = layerFields.data();
    //create the plugin object using the layerName and the plugin meta data
    IPluginV2* plugin = creator->createPlugin("TransformerEncodePluginDynamic", &pluginEncodeData);
    //add the plugin to the TensorRT network using the network API
    auto transEncode = network->addPluginV2(&inputs_list[0], 25, *plugin);
    inputs = transEncode->getOutput(0);
    plugin->destroy(); // Destroy the plugin object
  }
  ITensor *outputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, L, H}, 0));

  return outputs;
}

ITensor* getExtendSeqLen(TrtUniquePtr<IContext>& ctx, ITensor* inputs_shape, SampleTrtTransformerParams& params) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs_shape)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);
  nvinfer1::ITensor* FIVE = GetTensorInt(ctx, 5, 1);

  nvinfer1::ITensor* L = Gather(ctx, inputs_shape, 1, 1);
  nvinfer1::ITensor* DIV = GetTensorInt(ctx, params.downsample, 1);
  DIV = ReshapeDynamic(ctx, DIV, Concat(ctx, {ONE, ONE}, 0));
  DIV = Tile(ctx, DIV, Concat(ctx, {B, ONE}, 0));
  L = ElementWise(ctx, {L, DIV}, nvinfer1::ElementWiseOperation::kDIV);

  L = Tile(ctx, L, Concat(ctx, {ONE, FIVE}, 0));
  L = ReshapeDynamic(ctx, L, Concat(ctx, {B, FIVE}, 0));

  return L;
}

ITensor* getExtendMemory(TrtUniquePtr<IContext>& ctx, ITensor* inputs) {
  auto network = ctx->getNetWorkDefine();
  nvinfer1::ITensor* shape = network->addShape(*inputs)->getOutput(0);
  nvinfer1::ITensor* B = Gather(ctx, shape, 0, 0);
  nvinfer1::ITensor* W = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* C = Gather(ctx, shape, 2, 0);
  nvinfer1::ITensor* ONE = GetTensorInt(ctx, 1, 1);
  nvinfer1::ITensor* FIVE = GetTensorInt(ctx, 5, 1);

  inputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, ONE, W, C}, 0));
  inputs = Tile(ctx, inputs, Concat(ctx, {ONE, FIVE, ONE, ONE}, 0));
  B = ElementWise(ctx, {B, FIVE}, nvinfer1::ElementWiseOperation::kPROD);
  inputs = ReshapeDynamic(ctx, inputs, Concat(ctx, {B, W, C}, 0));
  return inputs;
}

std::vector<ITensor*> buildTransDecodeOp(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
    ITensor* inputs_shape, SampleTrtTransformerParams& params) {
  auto network = ctx->getNetWorkDefine();
  ITensor* shape = network->addShape(*inputs)->getOutput(0);
  ITensor* B = Gather(ctx, shape, 0, 0);
  ITensor* L = Gather(ctx, shape, 1, 0);
  nvinfer1::ITensor* FIVE = GetTensorInt(ctx, 5, 1);

  ITensor* extended_memory_sequence_length = getExtendSeqLen(ctx, inputs_shape, params);
  ITensor* extended_memory = getExtendMemory(ctx, inputs);
  DataType type = params.fp16 ? DataType::kHALF : DataType::kFLOAT;
  IScope decoder_scope = scope.subIScope("Transformer/decoder_stack");
  std::vector<std::vector<nvinfer1::Weights>> weights_list(18);
  for(int i = 0; i < params.num_layer; i++) {
    IScope self_attention_scope = decoder_scope.subIScope("layer_"+to_string(i)+"/self_attention");
    IScope encdec_attention_scope = decoder_scope.subIScope("layer_"+to_string(i)+"/encdec_attention");
    IScope ffn_scope = decoder_scope.subIScope("layer_"+to_string(i)+"/ffn");
std:
    vector<IScope> scope_list = {self_attention_scope.subIScope("layer_normalization/layer_norm_scale"),
                                 self_attention_scope.subIScope("layer_normalization/layer_norm_bias"),
                                 self_attention_scope.subIScope("self_attention/q/kernel"),
                                 self_attention_scope.subIScope("self_attention/k/kernel"),
                                 self_attention_scope.subIScope("self_attention/v/kernel"),
                                 self_attention_scope.subIScope("self_attention/output_transform/kernel"),
                                 encdec_attention_scope.subIScope("layer_normalization/layer_norm_scale"),
                                 encdec_attention_scope.subIScope("layer_normalization/layer_norm_bias"),
                                 encdec_attention_scope.subIScope("attention/q/kernel"),
                                 encdec_attention_scope.subIScope("attention/k/kernel"),
                                 encdec_attention_scope.subIScope("attention/v/kernel"),
                                 encdec_attention_scope.subIScope("attention/output_transform/kernel"),
                                 ffn_scope.subIScope("layer_normalization/layer_norm_scale"),
                                 ffn_scope.subIScope("layer_normalization/layer_norm_bias"),
                                 ffn_scope.subIScope("feed_foward_network/filter_layer/kernel"),
                                 ffn_scope.subIScope("feed_foward_network/filter_layer/bias"),
                                 ffn_scope.subIScope("feed_foward_network/output_layer/kernel"),
                                 ffn_scope.subIScope("feed_foward_network/output_layer/bias")
                                };
    for (unsigned j = 0; j < scope_list.size(); j++) {
      nvinfer1::Weights weight = ctx->getWeightsByName(scope_list[j].getOpName());
      weights_list[j].push_back(weight);
    }
  }
  std::vector<ITensor*> inputs_list = {extended_memory, extended_memory_sequence_length};
  for (unsigned j = 0; j < weights_list.size(); j++) {
    ITensor* t = GetConstTensor(ctx, weights_list[j], type);
    inputs_list.push_back(t);
  }
  std::vector<IScope> scope_list = {decoder_scope.subIScope("layer_normalization/layer_norm_scale"),
                                    decoder_scope.subIScope("layer_normalization/layer_norm_bias"),
                                    scope.subIScope("Transformer/embedding_shared_weights/embedding_and_softmax/weights")
                                   };
  for (unsigned j = 0; j < scope_list.size(); j++) {
    nvinfer1::Weights weight = ctx->getWeightsByName(scope_list[j].getOpName());
    ITensor* t = GetConstTensor(ctx, weight, type);
    inputs_list.push_back(t);
  }
  auto creator = getPluginRegistry()->getPluginCreator("TransformerDecodePluginDynamic", "1");
  DataType type_id = type;
  std::vector<PluginField> layerFields;
  layerFields.emplace_back(PluginField("type_id", &type_id, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("hidden_size", &params.hidden_size, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("num_heads", &params.num_heads, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("beam_width", &params.beam_width, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("vocab_size", &params.vocab_size, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("start_id", &params.start_id, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("end_id", &params.end_id, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("num_layer", &params.num_layer, PluginFieldType::kINT32, 1));
  PluginFieldCollection pluginDecodeData;
  pluginDecodeData.nbFields = layerFields.size();
  pluginDecodeData.fields = layerFields.data();

  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("TransformerDecodePluginDynamic", &pluginDecodeData);

  //add the plugin to the TensorRT network using the network API
//    auto transDecode = network->addPluginV2(inputs_list, 23, *plugin);
  auto transDecode = network->addPluginV2(&inputs_list[0], 23, *plugin);
  ITensor* output_ids = transDecode->getOutput(0);
  ITensor* parent_ids = transDecode->getOutput(1);
  ITensor* sequence_length = transDecode->getOutput(2);
  plugin->destroy(); // Destroy the plugin object

  std::vector<ITensor*> outputs;
  outputs.push_back(output_ids);
  outputs.push_back(parent_ids);
  outputs.push_back(sequence_length);

  return outputs;
}

void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& inputs,
                  SampleTrtTransformerParams& params,
                  std::vector<ITensor*>& outputs_vec) {
  assert(inputs[0]);
  assert(inputs[1]);

  IScope pre_scope;
  nvinfer1::Permutation perm_input;
  perm_input.order[0] = 0;
  perm_input.order[1] = 3;
  perm_input.order[2] = 1;
  perm_input.order[3] = 2;
  ITensor* inputs_trans = Transpose(ctx, inputs[0], perm_input);

  ITensor* outputs;
  if (params.resnet_vd) {
    outputs = buildResNetvd(ctx, pre_scope, inputs_trans, params);
  } else {
    IScope backbone_scope = pre_scope.subIScope("resnet_model");
    outputs = buildResNet50(ctx, backbone_scope, inputs_trans, params);
  }

  IScope conv_scope = pre_scope.subIScope("ocr_transformer/backbone/post_conv");
  outputs = Conv2d(ctx, conv_scope, outputs, 512, 1, 1, 1);

  IScope bn_scope = pre_scope.subIScope("post_bn");
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

  conv_scope = pre_scope.subIScope("ocr_transformer/backbone/dense");
  outputs = Conv2d(ctx, conv_scope, outputs, 512, 1, 1, 1);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);

  outputs = buildTransEncodeOp(ctx, pre_scope, outputs, inputs[1], params);

  outputs_vec = buildTransDecodeOp(ctx, pre_scope, outputs, inputs[1], params);
}

