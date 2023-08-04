#include <cuda_runtime_api.h>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "sampleTrtLayers.h"
#include "NvInfer.h"
#include "netUtils.h"

ITensor* buildBlock(TrtUniquePtr<IContext>& ctx, std::string name, ITensor* inputs, int filters, int stride,
                    int axis, bool is_projection_shortcut) {
  ITensor* shortcut;
  IScope bn_scope(name + "_preact_bn");
  ITensor* preact = BatchNorm(ctx, bn_scope, inputs, axis, 1e-5f);
  preact = Activation(ctx, preact, nvinfer1::ActivationType::kRELU);

  if(is_projection_shortcut) {
    IScope scope_shortcut(name + "_0_conv");
    shortcut = Conv2d(ctx, scope_shortcut, preact, 4 * filters, 1, stride, 1);
  } else {
    shortcut = Subsample(ctx, inputs, stride);
  }

  IScope scope_conv1(name + "_1_conv");
  IScope scope_bn1(name + "_1_bn");
  inputs = Conv2d(ctx, scope_conv1, preact, filters, 1, 1, 1);
  inputs = BatchNorm(ctx, scope_bn1, inputs, axis, 1e-5f);
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  IScope scope_conv2(name + "_2_conv");
  IScope scope_bn2(name + "_2_bn");
  inputs = Conv2d(ctx, scope_conv2, inputs, filters, 3, stride, 1);
  inputs = BatchNorm(ctx, scope_bn2, inputs, axis, 1e-5f);
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);

  IScope scope_conv3(name + "_3_conv");
  inputs = Conv2d(ctx, scope_conv3, inputs, 4 * filters, 1, 1, 1);
  inputs = ElementWise(ctx, {inputs, shortcut}, nvinfer1::ElementWiseOperation::kSUM);
  return inputs;
}

ITensor* blockLayer(TrtUniquePtr<IContext>& ctx, std::string name, ITensor* inputs, int blocks, int filters,
                    int stride, int axis) {
  inputs = buildBlock(ctx, name + "_block1", inputs, filters, 1, axis, true);
  for(int i = 2; i < blocks; i++) {
    inputs = buildBlock(ctx, name + "_block" + std::to_string(i), inputs, filters, 1, axis, false);
  }
  inputs = buildBlock(ctx, name + "_block" + std::to_string(blocks), inputs, filters, stride, axis, false);
  return inputs;
}

void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& input_tensors,
                  std::vector<ITensor*>& output_tensors) {
  assert(input_tensors[0]);
  IScope pre_scope;
  nvinfer1::Permutation perm_pre{0, 3, 1, 2};
  ITensor* inputs = Transpose(ctx, input_tensors[0], perm_pre);

  int channel_axis = 1;
  int filters = 64;
  IScope scope_conv("conv1_conv");
  ITensor* outputs = Conv2d(ctx, scope_conv, inputs, filters, 7, 2, 1);
  DimsHW d1(1, 1), d2(1, 1);
  outputs = Padding(ctx, outputs, d1, d2);
  outputs = MaxPooling(ctx, outputs, 3, 2, nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
  std::vector<int> block_size = {3, 4, 6, 3};
  std::vector<int> block_strides = {2, 2, 2, 1};
  for (int i = 2; i <=5; i++) {
    std::string name = "conv" + std::to_string(i);
    outputs = blockLayer(ctx, name, outputs, block_size[i - 2], filters, block_strides[i - 2], channel_axis);
    filters *= 2;
  }
  IScope bn_scope("post_bn");
  outputs = BatchNorm(ctx, bn_scope, outputs, channel_axis, 1e-5f);
  outputs = Activation(ctx, outputs, nvinfer1::ActivationType::kRELU);
  outputs = GlobalAvgPooling(ctx, outputs, true);
  IScope fc_scope("dense");
  outputs = FullyConnected(ctx, fc_scope, outputs, 2);
  outputs = Softmax(ctx, outputs, 1);
  output_tensors.push_back(outputs);
}

