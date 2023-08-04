# layers模块说明


## Conv2d

函数定义：
```
ITensor* Conv2d(TrtUniquePtr<IContext>& ctx, 
                IScope& scope, 
                ITensor* inputs, 
                int filters,
                int kernel_size, 
                int stride, 
                int dilation_rate,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope& , ITensor* )=nullptr)
```

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* filters：卷积核个数
* kernel_size：卷积核大小
* stride：步长
* dilation_rate：空洞卷积膨胀比例
* activate：指定激活函数，默认为NULL


## Conv2dTranspose

函数定义：

```
ITensor* Conv2dTranspose(TrtUniquePtr<IContext>& ctx, 
                         IScope& scope, 
                         ITensor* inputs, 
                         int filters, 
                         int kernel_size,
                         int stride, 
                         nvinfer1::PaddingMode mode,
                         ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope& , ITensor* )=nullptr)
```

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* filters：kernel个数
* kernel_size：kernel大小
* stride：步长
* mode：pading方式，SAME_UPPER对应TF中SAME
* activate：指定激活函数，默认为NULL

## Conv1d

函数定义：

```
ITensor* Conv1d(TrtUniquePtr<IContext>& ctx, 
                IScope& scope, 
                ITensor* inputs, 
                int filters, 
                int kernel_size_w,
                int kernel_size_h, 
                int stride_w, 
                int stride_h, 
                nvinfer1::PaddingMode mode,
                ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope& , ITensor* )=nullptr)
```

对齐Python中Conv1d，输入为NCHW形式，W维度步长和kernel设定为1。

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* filters：kernel个数
* kernel_size_w：w方向kernel大小
* kernel_size_h：h方向kernel大小
* stride_w：h方向kernel步长
* stride_h：h方向kernel步长
* mode：pading方式，kSAME_UPPER对应TF中SAME
* activate：指定激活函数，默认为NULL

## MaxPooling

函数定义：
```

ITensor* MaxPooling(TrtUniquePtr<IContext>& ctx, 
                    ITensor* inputs,
                    int kernel_size, 
                    int stride, 
                    nvinfer1::PaddingMode mode=nvinfer1::PaddingMode::kSAME_UPPER)
```
参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* kernel_size：kernel大小
* stride：kernel步长
* mode：pading方式，默认kSAME_UPPER对应Tensorflow中SAME，EXPLICIT_ROUND_UP对应Tensorflow中VALID

## AvgPooling

函数定义：
```
ITensor* AvgPooling(TrtUniquePtr<IContext>& ctx, 
                    ITensor* inputs,
                    int kernel_size, 
                    int stride, 
                    nvinfer1::PaddingMode paddingMode=nvinfer1::PaddingMode::kSAME_UPPER)
```

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* kernel_size：kernel大小
* stride：kernel步长
* mode：pading方式，默认kSAME_UPPER对应Tensorflow中SAME，EXPLICIT_ROUND_UP对应Tensorflow中VALID

## GlobalAvgPooling

函数定义：
```
ITensor* GlobalAvgPooling(TrtUniquePtr<IContext>& ctx, 
                          ITensor* inputs, 
                          bool keep_dims)
```

支持NCHW格式输入，对H和W维度全局平均池化操作。

参数说明：
* ctx：IContext上下文
* inputs：输入
* keep_dims：是否保留求均值值的维度，保留的话维度变为1

## BatchNorm

函数定义：
```
ITensor* BatchNorm(TrtUniquePtr<IContext>& ctx, 
                   IScope& scope, 
                   ITensor* inputs, 
                   int axis=1, 
                   float eps=1e-5f)
```

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* kernel_size：kernel大小
* axis：指定维度
* eps：eps参数，默认1e-5

## LayerNorm

函数定义：
```
ITensor* LayerNorm(TrtUniquePtr<IContext>& ctx, 
                   IScope& scope, 
                   ITensor* inputs, 
                   int axis, 
                   float epsilon=1e-6)
```

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* axis：指定维度
* epsilon：默认1e-6

## FullyConnected

函数定义：
```
ITensor* FullyConnected(TrtUniquePtr<IContext>& ctx, 
                        IScope& scope, 
                        ITensor* inputs, 
                        int filters,
                        ITensor* (* activation) (TrtUniquePtr<IContext>&, IScope& , ITensor* )=nullptr)
```

全链接层，输入维度必须大约3维度，会将CHW维度进行合并然后做矩阵乘法。

参数说明：
* ctx：IContext上下文
* scope：指定layer的Scope
* inputs：输入
* filters：kernel个数
* activation：激活函数
                   
## Transpose

函数定义：
```
ITensor* Transpose(TrtUniquePtr<IContext>& ctx, 
                   ITensor* inputs, 
                   nvinfer1::Permutation const& perm)
```
参数说明：
* ctx：IContext上下文
* inputs：输入
* perm：转换次序

## Concat

函数定义：
```
ITensor* Concat(TrtUniquePtr<IContext>& ctx, 
                std::vector<nvinfer1::ITensor*> tensors, 
                int axis)
```
参数说明：
* ctx：IContext上下文
* tensors：要合并的tensor向量
* axis：合并维度

## Padding

函数定义：
```
ITensor* Padding(TrtUniquePtr<IContext>& ctx, 
                 ITensor* inputs, 
                 Dims& prePadding, 
                 Dims& postPadding)
```
注意tensor4pd中设定分别为h方向上padding和w方向上padding。
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* prePadding：上左方向padding
* postPadding：右下方向padding

## Reshape

函数定义：
```
ITensor* Reshape(TrtUniquePtr<IContext>& ctx, 
                 ITensor* inputs, 
                 nvinfer1::Dims dims)
```
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* dims：Reshpae的大小

## Softmax
函数定义：
```
ITensor* Softmax(TrtUniquePtr<IContext>& ctx, 
                 ITensor* inputs, 
                 int axis)
```
参数说明：
* ctx：IContext上下文
* inputs：输入
* axis：指定进行softmax的维度

## Subsample
函数定义：
```
ITensor* Subsample(TrtUniquePtr<IContext>& ctx, 
                   ITensor* inputs, 
                   int factor)
```
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* factor：采样步长 

## Activation
函数定义：
```
ITensor* Activation(TrtUniquePtr<IContext>& ctx, 
                    ITensor* inputs, 
                    nvinfer1::ActivationType op)
```
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* op：指定激活函数类型

## Resize
函数定义：
```
ITensor* Resize(TrtUniquePtr<IContext>& ctx, 
                ITensor* inputs, 
                int factor, 
                nvinfer1::ResizeMode mode)
```
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* factor：指定Resize的缩放因子
* mode：Resize模式


## Scale
函数定义：
```
ITensor* Scale(TrtUniquePtr<IContext>& ctx, 
               ITensor* inputs, 
               float scale, 
               float shift, 
               int axis=1)
```
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* scale：scale因子 
* shift：shift因子
* axis：指定作用维度

## ElementWise
函数定义：
```
ITensor* ElementWise(TrtUniquePtr<IContext>& ctx, 
                     std::vector<nvinfer1::ITensor*> tensors,
                     nvinfer1::ElementWiseOperation binary_op)
```
对输入tensor列表按元素进行算术运算。
参数说明：
* ctx：IContext上下文
* tensor：输入tensor
* op：指定元素运算符

## MatrixMultiply
```
ITensor* MatrixMultiply(TrtUniquePtr<IContext>& ctx, 
                        ITensor* inputs0, 
                        ITensor* inputs1,
                        bool is_transpose)
```

矩阵乘法运算，对应tf.matmul操作。
参数说明：
* ctx：IContext上下文
* inputs0：输入tenosr0
* inputs1：输入tenosr1
* is_transpose：tensor1是否进行转置

## GetTensorInt
函数定义：
```
ITensor* GetTensorInt(TrtUniquePtr<IContext>& ctx, 
                      int val, 
                      int size)
```
创建一个size大小的Tensor将其值填充为val
参数说明：
* ctx：IContext上下文
* val：常数int
* size：大小


## Tile
函数定义：
```
ITensor* Tile(TrtUniquePtr<IContext>& ctx, 
              ITensor* inputs, 
              ITensor* repeats)
```

将Tensor按维度进行复制。

参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* repeats：指定维度复制的倍数


## GetConstTensor
函数定义：
```
ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, 
                        nvinfer1::Weights wt, 
                        DataType& type)
```
将nvinfer1::weights转化为tensor。

参数说明：
* ctx：IContext上下文
* wt：weights
* type：要转换的数据类型


## GetConstTensor
函数定义：
```
ITensor* GetConstTensor(TrtUniquePtr<IContext>& ctx, 
                        std::vector<nvinfer1::Weights>& wts, 
                        DataType& type)
```
将一个Weights表表进行拼接然后转化为tensor。

参数说明：
* ctx：IContext上下文
* wt：weights列表
* type：要转换的数据类型


## Range
函数定义：
```
ITensor* Range(TrtUniquePtr<IContext>& ctx, 
               ITensor* length)
```
类似np.range，创建一个一维度的tensor
 
参数说明：
* ctx：IContext上下文
* length：长度length


## ReshapeDynamic
函数定义：
```
ITensor* ReshapeDynamic(TrtUniquePtr<IContext>& ctx, 
                        ITensor* inputs, 
                        ITensor* shape)
```
创建一个size大小的Tensor将其值填充为val

参数说明：
* ctx：IContext上下文
* inputs：输入 
* shape：新的shape类型为tensor


## ReduceMean
函数定义：
```
ITensor* ReduceMean(TrtUniquePtr<IContext>& ctx, 
                    ITensor* inputs, 
                    uint32_t axis, 
                    bool keep_dims)
```
对axis维度进行求均值，返回tensor。

参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* axis：指定维度
* keep_dims：是否保留求均值值的维度，保留的话维度变为1


## Unary
函数定义：
```
ITensor* Unary(TrtUniquePtr<IContext>& ctx, 
               ITensor* inputs, 
               nvinfer1::UnaryOperation op)
```
对输入tensor进行一元算术运算。
参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* size：大小

## Identity
函数定义：
```
ITensor* Identity(TrtUniquePtr<IContext>& ctx, 
                  ITensor* inputs, 
                  nvinfer1::DataType dtype)
```
对输入转换数据类型。

参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* dtype：输出类型 

## Gather
函数定义：
```
ITensor* Gather(TrtUniquePtr<IContext>& ctx, 
                ITensor* inputs, 
                int index,
                int axis)
```
对tensor指定维度操作聚集为一个新的tensor。

参数说明：
* ctx：IContext上下文
* inputs：输入tensor
* index：指定维度索引
* axis：维度
