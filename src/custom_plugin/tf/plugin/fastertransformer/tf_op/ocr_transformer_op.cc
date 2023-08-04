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
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/ocr_transformer_op.h"
#include "fastertransformer/tf_op/common_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/logging.h"
#include <cuda_fp16.h>
namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("OcrTransformer")
    .Input("inputs: T")
    .Input("att_mask: T")
    .Input("ffn_mask: T")
    .Input("att_conv1d0_kernel: T")
    .Input("att_conv1d0_bias: T")
    .Input("att_layernorm0_gamma: T")
    .Input("att_layernorm0_beta: T")
    .Input("att_conv1d1_kernel: T")
    .Input("att_conv1d1_bias: T")
    .Input("att_layernorm1_gamma: T")
    .Input("att_layernorm1_beta: T")
    .Input("att_layernorm2_gamma: T")
    .Input("att_layernorm2_beta: T")
    .Input("att_query_kernel: T")
    .Input("att_key_kernel: T")
    .Input("att_value_kernel: T")
    .Input("att_dense_kernel: T")
    .Input("ffn_layernorm0_gamma: T")
    .Input("ffn_layernorm0_beta: T")
    .Input("ffn_dense0_kernel: T")
    .Input("ffn_dense0_bias: T")
    .Input("ffn_dense1_kernel: T")
    .Input("ffn_dense1_bias: T")
    .Input("final_layernorm_gamma: T")
    .Input("final_layernorm_beta: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("is_last_layer: bool")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int head_num, size_per_head;
      bool is_last_layer;
      c->GetAttr("head_num", &head_num);
      c->GetAttr("size_per_head", &size_per_head);
      c->GetAttr("is_last_layer", &is_last_layer);
      int rank = c->Rank(c->input(0));
      if (rank != 3)
      {
        return errors::InvalidArgument("[@OcrTransformer::ShapeInference] "
                                       "invalid rank (from_tensor@input[0]): ",
                                       rank,
                                       ", should be 3");
      }

      // calculate batch size
      shape_inference::DimensionHandle from_len_dim;
      shape_inference::DimensionHandle batch_dim;
      shape_inference::ShapeHandle input0;

      batch_dim = c->Dim(c->input(0), 0);
      from_len_dim = c->Dim(c->input(0), 1);
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input0));

      c->set_output(0, c->MakeShape({batch_dim, from_len_dim, head_num * size_per_head}));
      return Status::OK();
    });

template <typename Device, typename T>
class OcrTransformerOp : public CommonOp<T>
{
public:
  explicit OcrTransformerOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES_OK(context, context->GetAttr("is_last_layer", &is_last_layer_));
  }

  void Compute(OpKernelContext *context) override
  {
    int rank = (int)context->input(0).dims();
    if (rank != 3)
    {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("[@OcrTransformer::Compute] "
                                          "invalid rank (from_tensor@input[0]): ",
                                          rank,
                                          ", should be 3"));
    }

    auto batch_size_ = (int)context->input(0).dim_size(0);
    auto from_seq_len_ = (int)context->input(0).dim_size(1);

    VLOG(2) << "[@OcrTransformer::Compute] getting batch size: "
            << batch_size_ << "\n";

    typedef OcrEncoderTransformerTraits<traits_::OpType, cuda::OpenMultiHeadAttentionOcr> EncoderTraits_;
    OcrEncoderTransformer<EncoderTraits_> *encoder_transformer_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      encoder_transformer_ = new OcrEncoderTransformer<EncoderTraits_>(allocator_,
                                                                        batch_size_,
                                                                        from_seq_len_,
                                                                        head_num_,
                                                                        size_per_head_,
                                                                        is_last_layer_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }

    OP_REQUIRES(context, context->num_inputs() == 25, errors::InvalidArgument("Less input arguments"));

    EncoderInitParamOCR<DataType_> param; //init param here
    param.cublas_handle = this->get_cublas_handler();
    this->get_tensor(context, 0, &param.inputs);
    this->get_tensor(context, 1, &param.att_mask);
    this->get_tensor(context, 2, &param.ffn_mask);
    this->get_tensor(context, 3, &param.self_conv1d0.kernel);
    this->get_tensor(context, 4, &param.self_conv1d0.bias);
    this->get_tensor(context, 5, &param.self_layernorm0.gamma);
    this->get_tensor(context, 6, &param.self_layernorm0.beta);
    this->get_tensor(context, 7, &param.self_conv1d1.kernel);
    this->get_tensor(context, 8, &param.self_conv1d1.bias);
    this->get_tensor(context, 9, &param.self_layernorm1.gamma);
    this->get_tensor(context, 10, &param.self_layernorm1.beta);
    this->get_tensor(context, 11, &param.self_layernorm2.gamma);
    this->get_tensor(context, 12, &param.self_layernorm2.beta);
    this->get_tensor(context, 13, &param.self_attention.query_weight.kernel);
    this->get_tensor(context, 14, &param.self_attention.key_weight.kernel);
    this->get_tensor(context, 15, &param.self_attention.value_weight.kernel);
    this->get_tensor(context, 16, &param.self_attention.attention_output_weight.kernel);
    this->get_tensor(context, 17, &param.ffn_layernorm.gamma);
    this->get_tensor(context, 18, &param.ffn_layernorm.beta);
    this->get_tensor(context, 19, &param.ffn.intermediate_weight.kernel);
    this->get_tensor(context, 20, &param.ffn.intermediate_weight.bias);
    this->get_tensor(context, 21, &param.ffn.output_weight.kernel);
    this->get_tensor(context, 22, &param.ffn.output_weight.bias);
    this->get_tensor(context, 23, &param.layernorm_final.gamma);
    this->get_tensor(context, 24, &param.layernorm_final.beta);

    // std::cout<<"att mask:";
    // for(int i=0; i<context->input(1).dims(); i++)
    //   std::cout<<(int)context->input(1).dim_size(i)<<" ";
    // std::cout<<std::endl;

    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {batch_size_, from_seq_len_, head_num_ * size_per_head_}, &output));

    param.transformer_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    OP_REQUIRES_OK(
        context,
        functor::OcrTransformerOpFunctor<Device, T>::Compute(
            context,
            param,
            encoder_transformer_));
    delete encoder_transformer_;
  }

private:
  int head_num_, size_per_head_;
  bool is_last_layer_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("OcrTransformer").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OcrTransformerOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow
