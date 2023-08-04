#ifndef TRT_SAMPLE_LAYERS_H
#define TRT_SAMPLE_LAYERS_H

#include <memory>
#include "NvInfer.h"
#include "common.h"
#include "networkContext.h"
#include "scope.h"

ITensor* Padding(TrtUniquePtr<IContext>& ctx, ITensor* inputs, Dims& prePadding, Dims& postPadding);

ITensor* MaxPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    int kernel_size, int stride, nvinfer1::PaddingMode mode=nvinfer1::PaddingMode::kSAME_UPPER);

ITensor* AvgPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                    int kernel_size, int stride, nvinfer1::PaddingMode mode=nvinfer1::PaddingMode::kSAME_UPPER);

ITensor* AvgPoolingV2(TrtUniquePtr<IContext>& ctx, ITensor* inputs,
                      int h_pool_size, int w_pool_size, int h_stride,
                      int w_stride, nvinfer1::PaddingMode mode=nvinfer1::PaddingMode::kSAME_UPPER);

ITensor* Softmax(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int axis);

ITensor* Conv2dTranspose(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int kernel_size,
                         int stride, nvinfer1::PaddingMode mode,
                         ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* ) = nullptr);

ITensor* Activation(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::ActivationType op);

ITensor* Subsample(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int factor);

ITensor* Concat(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::ITensor*> tensors, int axis);

ITensor* ElementWise(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::ITensor*> tensors,
                     nvinfer1::ElementWiseOperation op);

ITensor* Resize(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int factor, nvinfer1::ResizeMode mode);

ITensor* BatchNorm(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int axis=1, float eps=1e-5f);

ITensor* Scale(TrtUniquePtr<IContext>& ctx, ITensor* inputs, float scale, float shift, int axis=1);

ITensor* Transpose(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::Permutation const& perm);

ITensor* Reshape(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::Dims dims);

ITensor* FullyConnected(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                        ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* ) = nullptr);

ITensor* Conv2d(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                int kernel_size, int stride, int dilation,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* ) = nullptr);

ITensor* Conv2dV2(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters,
                int kernel_size, int h_stride, int w_stride, int dilation,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* ) = nullptr);

ITensor* GetTensorInt(TrtUniquePtr<IContext>& ctx, int val, int size);

ITensor* Tile(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* repeats);

ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, nvinfer1::Weights wt, DataType& type);

ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, std::vector<nvinfer1::Weights>& wts, DataType& type);

ITensor* Range(TrtUniquePtr<IContext>& ctx, ITensor* length);

ITensor* ReshapeDynamic(TrtUniquePtr<IContext>& ctx, ITensor* inputs, ITensor* shape);

ITensor* ReduceMean(TrtUniquePtr<IContext>& ctx, ITensor* inputs, uint32_t axis, bool keep_dims);

ITensor* GlobalAvgPooling(TrtUniquePtr<IContext>& ctx, ITensor* inputs, bool keep_dims);

ITensor* Unary(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::UnaryOperation op);

ITensor* Identity(TrtUniquePtr<IContext>& ctx, ITensor* inputs, nvinfer1::DataType dtype);

ITensor* Gather(TrtUniquePtr<IContext>& ctx, ITensor* inputs, int index, int axis);

ITensor* LayerNorm(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int axis, float epsilon=1e-6);

ITensor* Conv1d(TrtUniquePtr<IContext>& ctx, IScope& scope, ITensor* inputs, int filters, int kernel_size_w,
                int kernel_size_h, int stride_w, int stride_h, nvinfer1::PaddingMode mode,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope&, ITensor* )=nullptr);

ITensor* MatrixMultiply(TrtUniquePtr<IContext>& ctx, ITensor* inputs0, ITensor* inputs1, bool is_transpose);

#endif
