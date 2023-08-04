/*
 * Copyright (c) 2021, 4pd CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MASKRCNN_MODEL_H
#define MASKRCNN_MODEL_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "sampleTrtLayers.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <typeinfo>

// params
struct SampleTrtMaskRCNNParams : public samplesCommon::SampleParams {
  int PRENMS_TOPK;
  int KEEP_TOPK;
  int RESULTS_PER_IM;
  float PROPOSAL_NMS_THRESH;
  float RESULT_SCORE_THRESH;
  float FRCNN_NMS_THRESH;
  int MAX_SIDE;
  int inputH;
  int inputW;
  int channel_axis;
};

ITensor* bottleneck(TrtUniquePtr<IContext>& ctx, IScope scope, ITensor* inputs,
                    int ch_out, int stride, int rate,
                    int axis);

ITensor* blockLayer(TrtUniquePtr<IContext>& ctx, IScope scope, ITensor* inputs, int blocks, int filters,
                    int stride, int axis);

ITensor* buildResNet101(TrtUniquePtr<IContext>& ctx,
                        IScope scope_backbone,
                        ITensor* inputs,
                        int channel_axis,
                        std::vector<ITensor*>& outputs_list);

ITensor* buildFPN(TrtUniquePtr<IContext>& ctx,
                  IScope scope,
                  std::vector<ITensor*>& inputs_list,
                  int channel_axis,
                  std::vector<ITensor*>& outputs_list);

ITensor* rpnHead(TrtUniquePtr<IContext>& ctx,
                 IScope scope,
                 std::vector<ITensor*>& inputs_list,
                 std::vector<ITensor*>& outputs_list);

ITensor* scaleMeans(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    std::vector<float> means, int channel_axis);

ITensor* scaleStds(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                   std::vector<float> means, int channel_axis);

ITensor* proposalLayer(TrtUniquePtr<IContext>& ctx, std::vector<ITensor*>& inputs,
                       SampleTrtMaskRCNNParams& mParams);

ITensor* decodeBox(TrtUniquePtr<IContext>& ctx, ITensor* proposal, ITensor* box_logits,
                   int& stage, SampleTrtMaskRCNNParams& mParams);

ITensor* frcnnOutput(TrtUniquePtr<IContext>& ctx, ITensor* proposal, ITensor* box_logits,
                     ITensor* box_score, ITensor* box_cos, ITensor* box_sin, SampleTrtMaskRCNNParams& mParams);

ITensor* sliceDetections(TrtUniquePtr<IContext>& ctx, ITensor* inputs);

ITensor* roiAlign(TrtUniquePtr<IContext>& ctx,
                  IScope& scope,
                  std::vector<ITensor*> features,
                  ITensor* proposal,
                  int pooled_size,
                  int padding);

ITensor* rcnnHead(TrtUniquePtr<IContext>& ctx,
                  IScope& scope,
                  ITensor* inputs);

ITensor* rcnnLayer(TrtUniquePtr<IContext>& ctx,
                   IScope& scope,
                   ITensor* inputs,
                   std::vector<ITensor*>& outputs);

ITensor* maskLayer(TrtUniquePtr<IContext>& ctx,
                   IScope& scope,
                   ITensor* inputs);

void buildMaskRcnn(TrtUniquePtr<IContext>& context,
                   std::vector<ITensor*>& input_tensors,
                   SampleTrtMaskRCNNParams& mParams,
                   std::vector<ITensor*>& output_tensors);

#endif
