# Ocr build tensorflow trans_ctc model with tensor-rt

## 使用说明

1. Install tensorflow

    ```
    pip install tensorflow-gpu==1.14.0
    ```

2.  将tensorflow权重转化成trt格式

    ```
    cd python
    python saved_model.py
    ```

3.  配置好trt_trans_ctc_dy.cpp中的算法参数SampleTrtTransCtcParams，然后进行编译。


4.  运行sample_trt_trans_ctc bin文件

    如果模型文件存在的话，直接load模型文件进行预估，否则建立trt模型进行预估

    To run the sample in FP32 mode:
    ```
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_trans_ctc 0
    ```

    To run the sample in FP16 mode:
    ```
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_trans_ctc 0 --fp16
    ```

5.  运行后处理

    ```
    cd python
    python trans_ctc_process.py
    ```
