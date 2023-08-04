#include "maskrcnn_model.h"

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

ITensor* relu(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs) {
  inputs = Activation(ctx, inputs, nvinfer1::ActivationType::kRELU);
  return inputs;
}

ITensor* norm(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs) {
  IScope bn = scope.subIScope("BatchNorm");
  inputs = BatchNorm(ctx, bn, inputs);
  return inputs;
}

ITensor* bottleneck(TrtUniquePtr<IContext>& ctx, IScope scope, ITensor* inputs,
                    int ch_out, int stride, int rate,
                    int axis) {
  nvinfer1::Weights wt_bias{DataType::kFLOAT, nullptr, 0};
  ITensor* shortcut_in = inputs;
  IScope scope_conv1 = scope.subIScope("conv1");
  ITensor* outputs = Conv2d(ctx, scope_conv1, inputs, ch_out, 1, 1, rate, nonlin);

  IScope scope_conv2 = scope.subIScope("conv2");
  if (stride == 2) {
    outputs = Conv2d(ctx, scope_conv2, outputs,ch_out, 3, 2, rate, nonlin);
  } else {
    outputs = Conv2d(ctx, scope_conv2, outputs, ch_out, 3, stride, rate, nonlin);
  }

  IScope scope_conv3 = scope.subIScope("conv3");
  outputs = Conv2d(ctx, scope_conv3, outputs, ch_out * 4, 1, 1, rate, norm);
  int depth_in = shortcut_in->getDimensions().d[1];
  ITensor* shortcut_out = nullptr;

  if (depth_in != ch_out * 4) {
    IScope scope_con4 = scope.subIScope("convshortcut");
    shortcut_out = Conv2d(ctx, scope_con4, shortcut_in, ch_out * 4, 1, stride, rate, norm);
  } else {
    shortcut_out = shortcut_in;
  }

  outputs = Activation(ctx, ElementWise(ctx, {outputs, shortcut_out},
                                        nvinfer1::ElementWiseOperation::kSUM),
                       nvinfer1::ActivationType::kRELU);
  return outputs;
}

ITensor* blockLayer(TrtUniquePtr<IContext>& ctx, IScope scope, ITensor* inputs, int blocks, int filters,
                    int stride, int axis) {
  for(int i=0; i<blocks; i++) {
    IScope block_scope = scope.subIScope("block" + std::to_string(i));
    stride = i == 0 ? stride : 1;
    inputs = bottleneck(ctx, block_scope, inputs, filters, stride, 1, axis);
  }
  return inputs;
}

ITensor* buildResNet101(TrtUniquePtr<IContext>& ctx,
                        IScope scope_backbone,
                        ITensor* inputs,
                        int channel_axis,
                        std::vector<ITensor*>& outputs_list) {
  int filters = 64;
  int conv_stride = 2;
  int kernel_size = 7;
  IScope scope_conv = scope_backbone.subIScope("conv0");
  ITensor* outputs = Conv2d(ctx, scope_conv, inputs, filters, kernel_size, conv_stride, 1, nonlin);
  DimsHW d1(1, 1), d2(0, 0);
  outputs = Padding(ctx, outputs, d1, d2);
  outputs = MaxPooling(ctx, outputs, 3, 2, nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP);
  std::vector<int> block_size = {3, 4, 23, 3};
  std::vector<int> block_strides = {1, 2, 2, 2};
  std::vector<int> block_filters = {64, 128, 256, 512};
  IScope scope_group0("group0");
  ITensor* outputs0 = blockLayer(ctx, scope_group0, outputs, block_size[0], block_filters[0],
                                 block_strides[0], channel_axis);
  IScope scope_group1("group1");
  ITensor* outputs1 = blockLayer(ctx, scope_group1, outputs0, block_size[1], block_filters[1],
                                 block_strides[1], channel_axis);
  IScope scope_group2("group2");
  ITensor* outputs2 = blockLayer(ctx, scope_group2, outputs1, block_size[2], block_filters[2],
                                 block_strides[2], channel_axis);
  IScope scope_group3("group3");
  ITensor* outputs3 = blockLayer(ctx, scope_group3, outputs2, block_size[3], block_filters[3],
                                 block_strides[3], channel_axis);
  outputs_list.push_back(outputs0);
  outputs_list.push_back(outputs1);
  outputs_list.push_back(outputs2);
  outputs_list.push_back(outputs3);
  return outputs3;
}

ITensor* buildFPN(TrtUniquePtr<IContext>& ctx,
                  IScope scope,
                  std::vector<ITensor*>& inputs_list,
                  int channel_axis,
                  std::vector<ITensor*>& outputs_list) {
  IScope scope_fpn = scope.subIScope("fpn");
  int num_channel = 256;
  std::vector<ITensor*> lat_2345;
  for (unsigned int i = 0; i < inputs_list.size(); i++) {
    IScope scope_conv1 = scope_fpn.subIScope("lateral_1x1_c" + std::to_string(i + 2));
    ITensor* lat = Conv2d(ctx, scope_conv1, inputs_list[i], num_channel, 1, 1, 1);
    lat_2345.push_back(lat);
  }
  assert(lat_2345.size() == 4);
  std::vector<ITensor*> lat_sum_5432;
  int idx;

  for (int i = lat_2345.size() - 1; i >= 0; i--) {
    if (i == lat_2345.size() - 1) {
      lat_sum_5432.push_back(lat_2345[i]);
    } else {
      idx = lat_sum_5432.size() - 1;
      ITensor* lat_up = Resize(ctx, lat_sum_5432[idx], 2, nvinfer1::ResizeMode::kNEAREST);
      ITensor* lat_sum = ElementWise(ctx, {lat_2345[i], lat_up}, nvinfer1::ElementWiseOperation::kSUM);
      lat_sum_5432.push_back(lat_sum);
    }
  }
  assert(lat_sum_5432.size() == 4);
  for (int i = lat_sum_5432.size() - 1; i >= 0; i--) {
    IScope scope_conv2 = scope_fpn.subIScope("posthoc_3x3_p" + std::to_string(5 - i));
    ITensor* lat = Conv2d(ctx, scope_conv2, lat_sum_5432[i], num_channel, 3, 1, 1);
    outputs_list.push_back(lat);
  }
  idx = outputs_list.size() - 1;
  nvinfer1::ITensor* outputs = MaxPooling(ctx, outputs_list[idx], 1, 2);
  outputs_list.push_back(outputs);
  outputs = outputs_list[0];
  assert(outputs_list.size() == 5);
  return outputs;
}

ITensor* rpnHead(TrtUniquePtr<IContext>& ctx,
                 IScope scope,
                 std::vector<ITensor*>& inputs_list,
                 std::vector<ITensor*>& outputs_list) {
  IScope scope_rpn = scope.subIScope("rpn");
  int num_channel = 256;
  int num_anchors = 9;
  nvinfer1::Permutation perm{0, 2, 3, 1};
  std::vector<ITensor*> logits_label_list, logits_box_list;
  ITensor* logits_label = nullptr;
  ITensor* logits_box = nullptr;
  for (unsigned int i = 0; i < inputs_list.size(); i++) {
    IScope scope_conv = scope_rpn.subIScope("conv0");
    ITensor* inputs = Conv2d(ctx, scope_conv, inputs_list[i], num_channel, 3, 1, 1, relu);
    IScope scope_class = scope_rpn.subIScope("class");
    logits_label = Conv2d(ctx, scope_class, inputs, num_anchors, 1, 1, 1);
    IScope scope_box = scope_rpn.subIScope("box");
    logits_box = Conv2d(ctx, scope_box, inputs, num_anchors * 4, 1, 1, 1);

    logits_label = Transpose(ctx, logits_label, perm);
    logits_box = Transpose(ctx, logits_box, perm);
    nvinfer1::Dims sh0 = logits_label->getDimensions();
    Dims3 sh1(1, sh0.d[1] * sh0.d[2]* num_anchors, 1);
    Dims3 sh2(1, sh0.d[1] * sh0.d[2]* num_anchors, 4);
    logits_label = Reshape(ctx, logits_label, sh1);
    logits_box = Reshape(ctx, logits_box, sh2);
    logits_label_list.push_back(logits_label);
    logits_box_list.push_back(logits_box);
  }
  logits_box = Concat(ctx, logits_box_list, 1);
  logits_label = Concat(ctx, logits_label_list, 1);
  Dims3 sh3(-1, 1, 1);
  Dims3 sh4(-1, 4, 1);
  logits_label = Reshape(ctx, logits_label, sh3);
  logits_box = Reshape(ctx, logits_box, sh4);
  outputs_list.push_back(logits_box);
  outputs_list.push_back(logits_label);
  return outputs_list[1];
}

ITensor* scaleMeans(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    std::vector<float> means, int channel_axis) {
  nvinfer1::Weights scale_weights, shift_weights;
  int n = means.size();
  scale_weights = ctx->createTempWeights<float>(std::vector<float>(n, 1.0));
  for(int i=0; i<n; i++)
    means[i] = -means[i];
  shift_weights = ctx->createTempWeights<float>(means);
  nvinfer1::IScaleLayer* layer = ctx->getNetWorkDefine()->addScaleNd(*inputs, nvinfer1::ScaleMode::kCHANNEL, shift_weights,
                                 scale_weights, {}, channel_axis);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* scaleStds(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                   std::vector<float> means, int channel_axis) {
  nvinfer1::Weights scale_weights, shift_weights;
  int n = means.size();
  for(int i=0; i<n; i++)
    means[i] =  1. / means[i];
  scale_weights = ctx->createTempWeights<float>(means);
  shift_weights = ctx->createTempWeights<float>(std::vector<float>(n, 0.));
  nvinfer1::IScaleLayer* layer = ctx->getNetWorkDefine()->addScaleNd(*inputs, nvinfer1::ScaleMode::kCHANNEL, shift_weights,
                                 scale_weights, {}, channel_axis);
  assert(layer);
  nvinfer1::ITensor* outputs = layer->getOutput(0);
  return outputs;
}

ITensor* proposalLayer(TrtUniquePtr<IContext>& ctx, std::vector<ITensor*>& inputs,
                       SampleTrtMaskRCNNParams& mParams) {
  auto creator = getPluginRegistry()->getPluginCreator("OcrProposalLayer_TRT", "1");
  int prenms_topk = mParams.PRENMS_TOPK;
  int keep_topk = mParams.KEEP_TOPK;
  float iou_threshold = mParams.PROPOSAL_NMS_THRESH;
  int max_side = mParams.MAX_SIDE;
  std::vector<PluginField> layerFields;
  layerFields.emplace_back(PluginField("prenms_topk", &prenms_topk, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("keep_topk", &keep_topk, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("iou_threshold", &iou_threshold, PluginFieldType::kFLOAT32, 1));
  layerFields.emplace_back(PluginField("max_side", &max_side, PluginFieldType::kINT32, 1));
  PluginFieldCollection pluginData;
  pluginData.nbFields = layerFields.size();
  pluginData.fields = layerFields.data();
  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("OcrProposalLayer_TRT", &pluginData);
  //add the plugin to the TensorRT network using the network API
  ITensor* inputs_list[] = {inputs[1], inputs[0]};
  auto ocr_proposalLayer = ctx->getNetWorkDefine()->addPluginV2(inputs_list, 2, *plugin);
  ITensor* outputs = ocr_proposalLayer->getOutput(0);
  plugin->destroy(); // Destroy the plugin object
  return outputs;
}

ITensor* decodeBox(TrtUniquePtr<IContext>& ctx, ITensor* proposal, ITensor* box_logits,
                   int& stage, SampleTrtMaskRCNNParams& mParams) {
  int max_side = mParams.MAX_SIDE;
  auto creator = getPluginRegistry()->getPluginCreator("OcrDecodeBoxLayer_TRT", "1");
  std::vector<PluginField> layerFields;
  layerFields.emplace_back(PluginField("cascade_stage", &stage, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("max_side", &max_side, PluginFieldType::kINT32, 1));
  PluginFieldCollection pluginData;
  pluginData.nbFields = layerFields.size();
  pluginData.fields = layerFields.data();
  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("OcrDecodeBoxLayer_TRT", &pluginData);
  //add the plugin to the TensorRT network using the network API
  ITensor* inputs_list[] = {proposal, box_logits};
  auto ocr_proposalLayer = ctx->getNetWorkDefine()->addPluginV2(inputs_list, 2, *plugin);
  ITensor* outputs = ocr_proposalLayer->getOutput(0);
  plugin->destroy(); // Destroy the plugin object
  return outputs;
}

ITensor* frcnnOutput(TrtUniquePtr<IContext>& ctx, ITensor* proposal, ITensor* box_logits,
                     ITensor* box_score, ITensor* box_cos, ITensor* box_sin, SampleTrtMaskRCNNParams& mParams) {
  auto creator = getPluginRegistry()->getPluginCreator("OcrDetectionLayer_TRT", "1");
  int num_classes = 2;
  int keep_topk = mParams.RESULTS_PER_IM;
  float score_threshold = mParams.RESULT_SCORE_THRESH;
  float iou_threshold = mParams.FRCNN_NMS_THRESH;
  int max_side = mParams.MAX_SIDE;
  std::vector<PluginField> layerFields;
  layerFields.emplace_back(PluginField("num_classes", &num_classes, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("keep_topk", &keep_topk, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("score_threshold", &score_threshold, PluginFieldType::kFLOAT32, 1));
  layerFields.emplace_back(PluginField("iou_threshold", &iou_threshold, PluginFieldType::kFLOAT32, 1));
  layerFields.emplace_back(PluginField("max_side", &max_side, PluginFieldType::kINT32, 1));
  PluginFieldCollection pluginData;
  pluginData.nbFields = layerFields.size();
  pluginData.fields = layerFields.data();
  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("OcrProposalLayer_TRT", &pluginData);
  //add the plugin to the TensorRT network using the network API
  ITensor* inputs_list[] = {proposal, box_score, box_logits, box_cos, box_sin};
  auto ocr_proposalLayer = ctx->getNetWorkDefine()->addPluginV2(inputs_list, 5, *plugin);
  ITensor* outputs = ocr_proposalLayer->getOutput(0);
  plugin->destroy(); // Destroy the plugin object
  return outputs;
}

ITensor* sliceDetections(TrtUniquePtr<IContext>& ctx, ITensor* inputs) {
  auto creator = getPluginRegistry()->getPluginCreator("OcrSpecialSlice_TRT", "1");
  std::vector<PluginField> layerFields;
  PluginFieldCollection pluginData;
  pluginData.nbFields = layerFields.size();
  pluginData.fields = layerFields.data();
  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("OcrProposalLayer_TRT", &pluginData);
  //add the plugin to the TensorRT network using the network API
  ITensor* inputs_list[] = {inputs};
  auto ocr_proposalLayer = ctx->getNetWorkDefine()->addPluginV2(inputs_list, 1, *plugin);
  ITensor* outputs = ocr_proposalLayer->getOutput(0);
  plugin->destroy(); // Destroy the plugin object
  return outputs;
}

ITensor* roiAlign(TrtUniquePtr<IContext>& ctx,
                  IScope& scope,
                  std::vector<ITensor*> features,
                  ITensor* proposal,
                  int pooled_size,
                  int padding) {
  assert(int(features.size()) == 5);
  auto creator = getPluginRegistry()->getPluginCreator("OcrPyramidROIAlign_TRT", "1");
  std::vector<PluginField> layerFields;
  layerFields.emplace_back(PluginField("pooled_size", &pooled_size, PluginFieldType::kINT32, 1));
  layerFields.emplace_back(PluginField("padding", &padding, PluginFieldType::kINT32, 1));
  PluginFieldCollection pluginData;
  pluginData.nbFields = layerFields.size();
  pluginData.fields = layerFields.data();
  //create the plugin object using the layerName and the plugin meta data
  IPluginV2* plugin = creator->createPlugin("OcrPyramidROIAlign_TRT", &pluginData);
//    add the plugin to the TensorRT network using the network API
  for (unsigned int i = 0; i < features.size(); i++) {
    Dims dims = features[i]->getDimensions();
    assert(dims.d[0] == 1);
    Dims3 sq(dims.d[1], dims.d[2], dims.d[3]);
    features[i] = Reshape(ctx, features[i], sq);
  }
  ITensor* inputs_list[] = {proposal, features[0], features[1], features[2], features[3]};
  auto ocr_proposalLayer = ctx->getNetWorkDefine()->addPluginV2(inputs_list, 5, *plugin);
  ITensor* outputs = ocr_proposalLayer->getOutput(0);
  plugin->destroy(); // Destroy the plugin object
  return outputs;
}

ITensor* rcnnHead(TrtUniquePtr<IContext>& ctx,
                  IScope& scope,
                  ITensor* inputs) {
  IScope rcnn_scope = scope.subIScope("head");
  int hidden_dims = 1024;
  IScope f6_scope = rcnn_scope.subIScope("fc6");
  inputs = FullyConnected(ctx, f6_scope, inputs, hidden_dims, relu);

  IScope f7_scope = rcnn_scope.subIScope("fc7");
  inputs = FullyConnected(ctx, f7_scope, inputs, hidden_dims, relu);
  return inputs;
}

ITensor* rcnnLayer(TrtUniquePtr<IContext>& ctx,
                   IScope& scope,
                   ITensor* inputs,
                   std::vector<ITensor*>& outputs) {
  IScope rcnn_scope = scope.subIScope("outputs");
  int num_class = 2;
  IScope class_scope = rcnn_scope.subIScope("class");
  ITensor* classification = FullyConnected(ctx, class_scope, inputs, num_class);
  classification = Softmax(ctx, classification, 1);

  IScope box_scope = rcnn_scope.subIScope("box");
  ITensor* box_regression = FullyConnected(ctx, box_scope, inputs, 4);

  IScope cos_scope = rcnn_scope.subIScope("cos");
  ITensor* box_cos = FullyConnected(ctx, cos_scope, inputs, 1);

  IScope sin_scope = rcnn_scope.subIScope("sin");
  ITensor* box_sin = FullyConnected(ctx, sin_scope, inputs, 1);

  outputs.push_back(classification);
  outputs.push_back(box_regression);
  outputs.push_back(box_cos);
  outputs.push_back(box_sin);
  return classification;
}

ITensor* maskLayer(TrtUniquePtr<IContext>& ctx,
                   IScope& scope,
                   ITensor* inputs) {
  IScope mask_scope = scope.subIScope("maskrcnn");
  int num_convs = 4;
  int num_category = 1;
  int hidden_dims = 256;
  for (unsigned int i = 0; i < num_convs; i++) {
    IScope conv_scope = mask_scope.subIScope("fcn" + to_string(i));
    inputs = Conv2d(ctx, conv_scope, inputs, hidden_dims, 3, 1, 1, relu);
  }
  IScope deconv_scope = mask_scope.subIScope("deconv/conv2d_transpose");
  inputs = Conv2dTranspose(ctx, deconv_scope, inputs, hidden_dims, 2, 2,
                           nvinfer1::PaddingMode::kSAME_UPPER, relu);
  IScope conv2_scope = mask_scope.subIScope("conv");
  inputs = Conv2d(ctx, conv2_scope, inputs, 1, 1, 1, 1, sigmoid);
  return inputs;
}

void buildMaskRcnn(TrtUniquePtr<IContext>& context,
                   std::vector<ITensor*>& input_tensors,
                   SampleTrtMaskRCNNParams& mParams,
                   std::vector<ITensor*>& output_tensors) {
  int channel_axis = mParams.channel_axis;
  nvinfer1::Permutation perm{0, 3, 1, 2};
  IScope root_scope;
  ITensor* inputs = Transpose(context, input_tensors[0], perm);
  ITensor* means = scaleMeans(context, inputs, {103.53, 116.28, 123.675}, 1);
  ITensor* stds = scaleStds(context, means, {57.375, 57.12, 58.395}, 1);
  std::vector<nvinfer1::ITensor*> blocks, features;
  inputs = buildResNet101(context, root_scope, stds, channel_axis, blocks);

  inputs = buildFPN(context, root_scope, blocks, channel_axis, features);
  //rpn head
  std::vector<nvinfer1::ITensor*> rpn_logits;
  inputs = rpnHead(context, root_scope, features,  rpn_logits);
  //rpn proposal
  ITensor* proposal = proposalLayer(context, rpn_logits, mParams);
  //cascade 0
  inputs = roiAlign(context, root_scope, features, proposal, 14, 1);
  inputs = AvgPooling(context, inputs, 2, 2);
  IScope cascade_rcnn_stage1_scope("cascade_rcnn_stage1");
  inputs = rcnnHead(context, cascade_rcnn_stage1_scope, inputs);
  std::vector<ITensor*> rcnnOutputs1;
  inputs = rcnnLayer(context, cascade_rcnn_stage1_scope, inputs, rcnnOutputs1);
  int cascade_stage = 0;
  proposal = decodeBox(context, proposal, rcnnOutputs1[1], cascade_stage, mParams);

  //cascade 1
  inputs = roiAlign(context, root_scope, features, proposal, 14, 1);
  inputs = AvgPooling(context, inputs, 2, 2);
  IScope cascade_rcnn_stage2_scope("cascade_rcnn_stage2");
  inputs = rcnnHead(context, cascade_rcnn_stage2_scope, inputs);
  std::vector<ITensor*> rcnnOutputs2;
  inputs = rcnnLayer(context, cascade_rcnn_stage2_scope, inputs, rcnnOutputs2);
  cascade_stage = 1;
  proposal = decodeBox(context, proposal, rcnnOutputs2[1], cascade_stage, mParams);

  //cascade 2
  inputs = roiAlign(context, root_scope, features, proposal, 14, 1);
  inputs = AvgPooling(context, inputs, 2, 2);
  IScope cascade_rcnn_stage3_scope("cascade_rcnn_stage3");
  inputs = rcnnHead(context, cascade_rcnn_stage3_scope, inputs);
  std::vector<ITensor*> rcnnOutputs3;
  inputs = rcnnLayer(context, cascade_rcnn_stage3_scope, inputs, rcnnOutputs3);
  ITensor* score_cascade = ElementWise(context, {rcnnOutputs1[0], rcnnOutputs2[0], rcnnOutputs3[0]}, nvinfer1::ElementWiseOperation::kSUM);
  score_cascade = scaleStds(context, score_cascade, {3.0, 3.0}, 1);

  //frcnn output
  ITensor* detections = frcnnOutput(context, proposal, rcnnOutputs3[1], score_cascade, rcnnOutputs3[2], rcnnOutputs3[3], mParams);
  //slice
  inputs = sliceDetections(context, detections);
  //mask rcnn
  inputs = roiAlign(context, root_scope, features, inputs, 28, 1);
  inputs = AvgPooling(context, inputs, 2, 2);
  ITensor* mask = maskLayer(context, root_scope, inputs);
  output_tensors.push_back(detections);
  output_tensors.push_back(mask);
}


