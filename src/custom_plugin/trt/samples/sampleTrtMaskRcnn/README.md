# Ocr MasK R-CNN tensorflow 转 trt

## 自定义cuda op算子

This sample makes use of TensorRT plugins to run the Mask R-CNN model. `.pb` model needs to be preprocessed and converted to the UFF model with the help of GraphSurgeon and the UFF utility.

- `ResizeNearest` - Nearest neighbor interpolation for resizing features. This works for the FPN (Feature Pyramid Network) module.

- `OcrProposalLayer` - Generate the first stage's proposals based on anchors and RPN's (Region Proposal Network) outputs (scores, bbox_deltas).

- `OcrPyramidROIAlign` - Crop and resize the feature of ROIs (first stage's proposals) from the corresponding feature layer.

- `OcrDetectionLayer` - Refine the first stage's proposals to produce final detections.

- `OcrSpecialSlice` - A workaround plugin to slice detection output [y1, x1, y2, x2, class_id, score] to [y1, x1, y2 , x2] for data with more than one index dimensions (batch_idx, proposal_idx, detections(y1, x1, y2, x2)).

- `ocrDecodeBox` - According to proposals and bbox_logits, produce detections.

## 使用说明

1.  Install tensorflow

    ```
    pip install tensorflow-gpu==1.14.0
    ```

2.  将tensorflow权重转化成trt格式.
	```
	cd sampleTrtMaskRCNN/python
	CUDA_VISIBLE_DEVICES=0 python convert_mrcnn_weights.py
	```

4.  配置好sampleTrtMaskRcnn.cpp中算法参数SampleTrtMaskRCNNParams，编译sampleTrtTrtMaskRcnn.cpp


5.  运行sample_trt_mask_rcnn bin文件

	如果模型文件存在的话，直接load模型文件进行预估，否则解析并建立trt模型进行预估

	To run the sample in FP32 mode:
	```
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_mask_rcnn 0
	```

	To run the sample in FP16 mode:
	```
	CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH ./sample_trt_mask_rcnn 0 --fp16
	```

6.  运行后处理

    ```
    cd python
    python maskrcnn_postprocess.py
    ```






