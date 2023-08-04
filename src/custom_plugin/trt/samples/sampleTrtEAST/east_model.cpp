#include "east_model.h"

ITensor* meanImageSubtraction(TrtUniquePtr<IContext>& ctx, ITensor* inputs, std::vector<float> means) {
  int n = means.size();
  std::vector<float> val1(n);
  for(int i=0; i<n; i++)
    val1[i] = 1.0;
  nvinfer1::Weights scale_weights = ctx->createTempWeights<float>(val1);

  std::vector<float> val2(n);
  for(int i=0; i<n; i++)
    val2[i] = -means[i];
  nvinfer1::Weights shift_weights = ctx->createTempWeights<float>(val2);

  nvinfer1::IScaleLayer* layer = ctx->getNetWorkDefine()->addScaleNd(*inputs, nvinfer1::ScaleMode::kCHANNEL,
                                 shift_weights, scale_weights, {}, 1);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* nonlin(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs) {
  IScope bn = scope.subIScope("BatchNorm");
  inputs = BatchNorm(ctx, bn, inputs);
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);
  return inputs;
}

ITensor* sigmoid(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs) {
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kSIGMOID);
  return inputs;
}

ITensor* norm(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs) {
  IScope bn = scope.subIScope("BatchNorm");
  inputs = BatchNorm(ctx, bn, inputs);
  return inputs;
}

ITensor* bottleneck(TrtUniquePtr<IContext>& ctx,
                    IScope& scope, ITensor* inputs,
                    int depth, int depth_bottleneck, int stride, int rate) {
  nvinfer1::Weights wt_bias{DataType::kFLOAT, nullptr, 0};
  nvinfer1::Dims dims = inputs->getDimensions();
  int depth_in = dims.d[1];
  ITensor* shortcut = nullptr;
  if(depth==depth_in) {
    shortcut = Subsample(ctx, inputs, stride);
  } else {
    IScope scope_shortcut = scope.subIScope("shortcut");
    shortcut = Conv2d(ctx, scope_shortcut, inputs, depth, 1, stride, 1, norm);
  }
  IScope scope_conv1 = scope.subIScope("conv1");
  ITensor* residual = Conv2d(ctx, scope_conv1, inputs, depth_bottleneck, 1, 1, 1, nonlin);
  IScope scope_conv2 = scope.subIScope("conv2");
  residual = Conv2d(ctx, scope_conv2, residual, depth_bottleneck, 3, stride, rate, nonlin);
  IScope scope_conv3 = scope.subIScope("conv3");
  residual = Conv2d(ctx, scope_conv3, residual, depth, 1, 1, 1, norm);
  ITensor* outputs = Activation(ctx, ElementWise(ctx, {shortcut, residual}, nvinfer1::ElementWiseOperation::kSUM),
                                nvinfer1::ActivationType::kRELU);
  return outputs;
}

ITensor* fusionASPP(TrtUniquePtr<IContext>& ctx,
                    IScope& scope, ITensor* inputs, int num) {
  IScope scope_conv1 = scope.subIScope("Conv");
  ITensor* ASPP1 = Conv2d(ctx, scope_conv1, inputs, num, 1, 1, 1, nonlin);

  IScope scope_conv11 = scope.subIScope("Conv_1");
  ITensor* ASPP2 = Conv2d(ctx, scope_conv11, inputs, num, 3, 1, 3, nonlin);

  IScope scope_conv2 = scope.subIScope("Conv_2");
  ITensor* ASPP3 = Conv2d(ctx, scope_conv2, inputs, num, 3, 1, 6, nonlin);

  IScope scope_conv3 = scope.subIScope("Conv_3");
  ITensor* ASPP4 = Conv2d(ctx, scope_conv3, inputs, num, 3, 1, 9, nonlin);

  IScope scope_conv4 = scope.subIScope("Conv_4");
  ITensor* ASPP5 = Conv2d(ctx, scope_conv4, inputs, num, 3, 1, 12, nonlin);

  IScope scope_conv5 = scope.subIScope("Conv_5");
  ITensor* ASPP6 = Conv2d(ctx, scope_conv5, inputs, num, 3, 1, 15, nonlin);

  IScope scope_conv6 = scope.subIScope("Conv_6");
  ITensor* ASPP7 = Conv2d(ctx, scope_conv6, inputs, num, 3, 1, 18, nonlin);

  ITensor* outputs = Concat(ctx, {ASPP1,ASPP2,ASPP3,ASPP4,ASPP5,ASPP6,ASPP7}, 1);
  IScope scope_conv7 = scope.subIScope("Conv_7");
  outputs = Conv2d(ctx, scope_conv7, outputs, num, 1, 1, 1, nonlin);

  return outputs;
}

ITensor* fusionFeature(TrtUniquePtr<IContext>& ctx,
                       IScope& scope, std::vector<ITensor*>& feat_F) {
  std::vector<ITensor*> feat_G(3, nullptr);
  std::vector<ITensor*> feat_H(3, nullptr);
  std::vector<int> num_outputs = {256, 64, 32};
  nvinfer1::Weights wt_bias{DataType::kFLOAT, nullptr, 0};

  feat_H[0] = fusionASPP(ctx, scope, feat_F[0], num_outputs[0]);
  //双线性插值
  feat_G[0] = Resize(ctx, feat_H[0], 2, nvinfer1::ResizeMode::kLINEAR);
  feat_H[1] = Concat(ctx, {feat_G[0], feat_F[1]}, 1);

  IScope scope_conv8 = scope.subIScope("Conv_8");
  feat_H[1] = Conv2d(ctx, scope_conv8, feat_H[1], num_outputs[1], 1, 1, 1, nonlin);

  IScope scope_conv9 = scope.subIScope("Conv_9");
  feat_H[1] = Conv2d(ctx, scope_conv9, feat_H[1], num_outputs[1], 3, 1, 1, nonlin);
  feat_G[1] = Resize(ctx, feat_H[1], 2, nvinfer1::ResizeMode::kLINEAR);

  IScope scope_conv10 = scope.subIScope("Conv_10");
  feat_H[2] = Concat(ctx, {feat_G[1], feat_F[2]}, 1);
  feat_H[2] = Conv2d(ctx, scope_conv10, feat_H[2], num_outputs[2], 1, 1, 1, nonlin);

  IScope scope_conv11 = scope.subIScope("Conv_11");
  feat_H[2] = Conv2d(ctx, scope_conv11, feat_H[2], num_outputs[2], 3, 1, 1, nonlin);

  IScope scope_conv12 = scope.subIScope("Conv_12");
  feat_G[2] = Conv2d(ctx, scope_conv12, feat_H[2], num_outputs[2], 3, 1, 1, nonlin);
  return feat_G[2];
}


void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& input_tensors,
                  std::vector<ITensor*>& output_tensors) {
  assert(input_tensors[0]);
  IScope pre_scope;
  nvinfer1::Permutation perm_pre{0, 3, 1, 2};
  ITensor* inputs = Transpose(ctx, input_tensors[0], perm_pre);
  inputs = meanImageSubtraction(ctx, inputs, {123.68, 116.78, 103.94});
  std::vector<ITensor*> feat_F(3, nullptr);

  IScope backbone_scope("resnet_v1_50");
  IScope conv1 = backbone_scope.subIScope("conv1");
  ITensor* outputs = Conv2d(ctx, conv1,inputs, 64, 7, 2, 1, nonlin);
  outputs = MaxPooling(ctx, outputs, 3, 2);
  feat_F[2] = outputs;

  IScope block11 = backbone_scope.subIScope("block1/unit_1/bottleneck_v1");
  outputs = bottleneck(ctx, block11, outputs, 256, 64, 1, 1);
  IScope block12 = backbone_scope.subIScope("block1/unit_2/bottleneck_v1");
  outputs = bottleneck(ctx, block12, outputs, 256, 64, 1, 1);
  IScope block13 = backbone_scope.subIScope("block1/unit_3/bottleneck_v1");
  outputs = bottleneck(ctx, block13, outputs, 256, 64, 2, 1);
  feat_F[1] = outputs;

  IScope block21 = backbone_scope.subIScope("block2/unit_1/bottleneck_v1");
  outputs = bottleneck(ctx, block21, outputs, 512, 128, 1, 1);
  IScope block22 = backbone_scope.subIScope("block2/unit_2/bottleneck_v1");
  outputs = bottleneck(ctx, block22, outputs, 512, 128, 1, 1);
  IScope block23 = backbone_scope.subIScope("block2/unit_3/bottleneck_v1");
  outputs = bottleneck(ctx, block23, outputs, 512, 128, 1, 1);
  IScope block24 = backbone_scope.subIScope("block2/unit_4/bottleneck_v1");
  outputs = bottleneck(ctx, block24,outputs, 512, 128, 2, 1);

  IScope block31 = backbone_scope.subIScope("block3/unit_1/bottleneck_v1");
  outputs = bottleneck(ctx, block31, outputs, 1024, 256, 1, 1);
  IScope block32 = backbone_scope.subIScope("block3/unit_2/bottleneck_v1");
  outputs = bottleneck(ctx, block32, outputs, 1024, 256, 1, 1);
  IScope block33 = backbone_scope.subIScope("block3/unit_3/bottleneck_v1");
  outputs = bottleneck(ctx, block33, outputs, 1024, 256, 1, 1);
  IScope block34 = backbone_scope.subIScope("block3/unit_4/bottleneck_v1");
  outputs = bottleneck(ctx, block34, outputs, 1024, 256, 1, 1);
  IScope block35 = backbone_scope.subIScope("block3/unit_5/bottleneck_v1");
  outputs = bottleneck(ctx, block35, outputs, 1024, 256, 1, 1);
  IScope block36 = backbone_scope.subIScope("block3/unit_6/bottleneck_v1");
  outputs = bottleneck(ctx, block36, outputs, 1024, 256, 1, 1);

  IScope block41 = backbone_scope.subIScope("block4/unit_1/bottleneck_v1");
  outputs = bottleneck(ctx, block41, outputs, 2048, 512, 1, 2);
  IScope block42 = backbone_scope.subIScope("block4/unit_2/bottleneck_v1");
  outputs = bottleneck(ctx, block42, outputs, 2048, 512, 1, 2);
  IScope block43 = backbone_scope.subIScope("block4/unit_3/bottleneck_v1");
  outputs = bottleneck(ctx, block43, outputs, 2048, 512, 1, 2);
  feat_F[0] = outputs;


  IScope scope_fusion("feature_fusion");
  outputs = fusionFeature(ctx, scope_fusion, feat_F);

  IScope scope_conv13 = scope_fusion.subIScope("Conv_13");
  ITensor* F_score = Conv2d(ctx, scope_conv13, outputs, 1, 1, 1, 1, sigmoid);

  IScope scope_conv14 = scope_fusion.subIScope("Conv_14");
  ITensor* geo_map = Conv2d(ctx, scope_conv14, outputs, 4, 1, 1, 1, sigmoid);
  geo_map = Scale(ctx, geo_map, 896.0, 0.0);

  IScope scope_conv15 = scope_fusion.subIScope("Conv_15");
  ITensor* angle_map = Conv2d(ctx, scope_conv15, outputs, 1, 1, 1, 1, sigmoid);
  angle_map = Scale(ctx, angle_map, 3.141592653589793*0.5, -0.5*0.5*3.141592653589793);

  ITensor* F_geometry = Concat(ctx, {geo_map, angle_map}, 1);

  IScope scope_conv16 = scope_fusion.subIScope("Conv_16");
  ITensor* F_cos_map = Conv2d(ctx, scope_conv16, outputs, 1, 1, 1, 1, sigmoid);

  IScope scope_conv17 = scope_fusion.subIScope("Conv_17");
  ITensor* F_sin_map = Conv2d(ctx, scope_conv17, outputs, 1, 1, 1, 1, sigmoid);

  nvinfer1::Permutation perm{0, 2, 3, 1};
  F_score = Transpose(ctx, F_score, perm);
  F_geometry = Transpose(ctx, F_geometry, perm);
  F_cos_map = Transpose(ctx, F_cos_map, perm);
  F_sin_map = Transpose(ctx, F_sin_map, perm);

  output_tensors.push_back(F_score);
  output_tensors.push_back(F_geometry);
  output_tensors.push_back(F_cos_map);
  output_tensors.push_back(F_sin_map);
}

