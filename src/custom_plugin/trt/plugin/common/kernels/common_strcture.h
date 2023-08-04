/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef COMMON_STRCTURE_H
#define COMMON_STRCTURE_H

template<typename T>
struct DenseWeight {
  const T* kernel;
  const T* bias;
};

template<typename T>
struct LayerNormWeight {
  const T* gamma;
  const T* beta;
};

template<typename T>
struct AttentionWeight {
  DenseWeight<T> query_weight;
  DenseWeight<T> key_weight;
  DenseWeight<T> value_weight;
  DenseWeight<T> attention_output_weight;
};

template<typename T>
struct FFNWeight {
  DenseWeight<T> intermediate_weight;
  DenseWeight<T> output_weight;
};

#endif