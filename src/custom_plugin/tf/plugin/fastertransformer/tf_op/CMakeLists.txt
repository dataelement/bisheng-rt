# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8)

set(tf_ocr_encode_files
  ocr_transformer_op.cc
  ocr_transformer_op.cu.cc
  ../cuda/ocr_open_attention.cu
  ../cuda/cuda_kernels.cu
)

set(tf_ocr_decoding_files
  ocr_decoding_op.cc
  ocr_decoding_op.cu.cc
  ../cuda/ocr_open_decoder.cu
  ../cuda/cuda_kernels.cu
  ../cuda/ocr_open_attention.cu
  ../cuda/decoding_kernel_check.cpp
)

set(tf_beam_search_files
  beam_search_op.cc
  beam_search_op.cu
)

# if(BUILD_TF)
#   add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# endif()
add_definitions(-DGOOGLE_CUDA=1)

add_library(tf_ocr_encode_op SHARED ${tf_ocr_encode_files})
target_link_libraries(tf_ocr_encode_op PRIVATE -lcublas -lcudart ${TF_LIB_DEPS})

add_library(tf_ocr_decoding_op SHARED ${tf_ocr_decoding_files})
target_link_libraries(tf_ocr_decoding_op PRIVATE -lcublas -lcudart ${TF_LIB_DEPS})

if (TF_VER STREQUAL "2") 
  set(tf_fastertransformer_op_files ${tf_ocr_encode_files} ${tf_ocr_decoding_files} ${tf_beam_search_files})
else()
  set(tf_fastertransformer_op_files ${tf_ocr_encode_files} ${tf_ocr_decoding_files})
endif()

add_library(tf_fastertransformer_op SHARED ${tf_fastertransformer_op_files})
target_link_libraries(tf_fastertransformer_op PRIVATE -lcublas -lcudart ${TF_LIB_DEPS})

set_property(TARGET tf_fastertransformer_op PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET tf_fastertransformer_op PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
