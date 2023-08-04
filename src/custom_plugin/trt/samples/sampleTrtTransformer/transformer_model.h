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

#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include "argsParser.h"
#include "sampleTrtLayers.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "netUtils.h"
#include <cuda_runtime_api.h>
#include <fstream>

// params
struct SampleTrtTransformerParams : public samplesCommon::SampleParams {
  int hidden_size;
  int num_heads;
  int beam_width;
  int vocab_size;
  int start_id;
  int end_id;
  int num_layer;
  int downsample;
  int channel_axis;
  bool resnet_vd;
};

ITensor* buildResNet50(TrtUniquePtr<IContext>& ctx, IScope& backbone_scope, ITensor* inputs,
                       SampleTrtTransformerParams& params);

ITensor* buildTransEncodeOp(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, ITensor* inputs_shape,
                            SampleTrtTransformerParams& params);

std::vector<ITensor*> buildTransDecodeOp(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs,
    ITensor* inputs_shape, SampleTrtTransformerParams& params);

void buildNetwork(TrtUniquePtr<IContext>& ctx, std::vector<ITensor*>& inputs,
                  SampleTrtTransformerParams& params, std::vector<ITensor*>& outputs_vec);

#endif