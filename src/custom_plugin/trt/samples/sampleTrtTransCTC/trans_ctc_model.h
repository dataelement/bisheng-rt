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

#ifndef TRANSCTC_MODEL_H
#define TRANSCTC_MODEL_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleTrtLayers.h"
#include "netUtils.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>

// params
struct SampleTrtTransCtcParams : public samplesCommon::SampleParams {
  int hidden_size;
  int filter_size;
  int num_heads;
  int num_layer;
  int downsample;
  int channel_axis;
};

ITensor* buildBlock(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int stride,
                    int& conv_i, int& bn_i, int axis, bool is_projection_shortcut);

ITensor* blockLayer31(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int blocks, int filters,
                      int stride, int& conv_i, int& bn_i, int axis);

ITensor* buildResNet31(TrtUniquePtr<IContext>& ctx, IScope& backbone_scope, ITensor* inputs, int channel_axis);

ITensor* splitHeads(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int hidden_size, int num_heads);

ITensor* combineHeads(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int hidden_size);

ITensor* selfAttention(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* y, ITensor* bias,
                       SampleTrtTransCtcParams& params);

ITensor* feedForwardNetWork(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
                            ITensor* inputs_padding, SampleTrtTransCtcParams& params);

ITensor* encoderStack(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* encoder_inputs, ITensor* attention_bias,
                      ITensor* inputs_padding, SampleTrtTransCtcParams& params);

ITensor* getInputsPadding(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* inputs_shape,
                          SampleTrtTransCtcParams& params);

ITensor* getPaddingBias(TrtUniquePtr<IContext>& ctx, ITensor* inputs, SampleTrtTransCtcParams& params);

ITensor* getPositionEncoding(TrtUniquePtr<IContext>& ctx, ITensor* inputs);

ITensor* buildTransEncode(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, ITensor* inputs_shape,
                          SampleTrtTransCtcParams& params);

void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& inputs,
                  SampleTrtTransCtcParams& params,
                  std::vector<ITensor*>& outputs_vec);

#endif

