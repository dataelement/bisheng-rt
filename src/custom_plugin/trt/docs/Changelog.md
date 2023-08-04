本项目基于官方TensorRT(https://github.com/NVIDIA/TensorRT)基础上进行开发

version: v22.08
time: 2022.10.20
1. 基于nvidia的官方TensorRT 22.08版本，TensorRT 8.4.2.4(cuda11.7 cudnn8.5)

version: v22.04
time: 2022.09.20
1. 基于nvidia的官方TensorRT 22.04版本，TensorRT 8.2.4.2(cuda11.6 cudnn8.4)

version: v1.0.3
time: 2022.03.01
1. transformer trt转换时vocab_size设置错误，需要把起始符号和结束符号考虑进去

version: v1.0.2-rc0
time: 2021.12.09
1. maskRCNNKernels.cu，OcrKeepTopKGather<256>，A100、3080卡系列如果线程设置成256，会出现内存越界，怀疑是cub::BlockRadixSort内部出现了内存越界，OcrKeepTopKGather<768>，A100、3080卡正常
2. 提供mrcnn和transformer自动转化tensorrt脚本

version: v1.0.2
time: 2021.12.06
1. maskrcnn trt模型修复bug，group/block/conv3的激活函数用错了，应该用BN，之前手误写成BN+Relu。但是奇怪的是2080、1080、p4、t4（Tensorrt7.0.0.1）等显卡分数没有影响，A100（Tensorrt7.2.1.6）有影响，2080可能是内部layer fuse时规避了

version: v1.0.1
time: 2021.09.14
1. transformer支持resnet_vd

version: v1.0.0
time: 2021.02.04
1. 重构trt-lib，抽离出model、scope、context、trtlayers公共接口
2. 计算力架构支持60、61、70、75、80、86，cuda支持10.0、11.1
3. 代码规范化，跟google代码对齐
4. 更新model、scope、context、trtlayers文档，east、mrcnn、transformer、trans-ctc使用说明
5. 增加手写、印刷二分类模型

version: v0.2
time: 2020.12.08
1. samples增加sampleTrtTransCTC、sampleTrtTransformer

version: v0.1
time: 2020.10.20
1. 去掉parse不相关的部分
2. plugin保留maskrcnn、east自定义op
3. samples增加sampleTrtEAST、sampleUffEAST、sampleUffMaskRCNN、sampleUffConvert
4. 支持fp32、fp16、int8
5. maskrcnn支持1056、1600、2048、2560四种size