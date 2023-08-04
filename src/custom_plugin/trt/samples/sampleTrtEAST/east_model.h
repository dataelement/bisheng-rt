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

#ifndef EAST_MODEL_H
#define EAST_MODEL_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInferPlugin.h"
#include "sampleTrtLayers.h"
#include "NvInfer.h"
#include "netUtils.h"
#include <cuda_runtime_api.h>

ITensor* meanImageSubtraction(TrtUniquePtr<IContext>& ctx, ITensor* inputs, std::vector<float> means);

ITensor* bottleneck(TrtUniquePtr<IContext>& ctx,
                    IScope& scope, ITensor* inputs,
                    int depth, int depth_bottleneck, int stride, int rate);

ITensor* fusionASPP(TrtUniquePtr<IContext>& ctx,
                    IScope& scope, ITensor* inputs, int num);

ITensor* fusionFeature(TrtUniquePtr<IContext>& ctx,
                       IScope& scope, std::vector<ITensor*>& feat_F);

void buildNetwork(TrtUniquePtr<IContext>& ctx,
                  std::vector<ITensor*>& inputs,
                  std::vector<ITensor*>& outputs_vec);

#endif
