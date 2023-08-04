# Ocr build tensorflow classification model with tensor-rt

## 使用说明

1. Install tensorflow

    ```
    pip install tensorflow-gpu==1.14.0
    ```

2.  将tensorflow权重转化成trt格式

    ```
    cd python
    CUDA_VISIBLE_DEVICES=0 python transform_weights.py
    ```

3.  运行preprocess.py脚本得到运行数据。

4.  配置好trt_classification.cpp中的算法参数SampleClassificationParams，然后进行编译。

5.  运行sample_trt_classification bin文件

    如果模型文件存在的话，直接load模型文件进行预估，否则建立trt模型进行预估

    To run the sample in FP32 mode:
    ```
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_classification 0
    ```

    To run the sample in FP16 mode:
    ```
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_classification 0 --fp16
    ```

6.  运行后处理

    ```
    cd python
    python postprocess.py
    ```
