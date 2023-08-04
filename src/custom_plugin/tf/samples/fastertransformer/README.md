# fastertransformer

## 自定义cuda op算子

This sample makes use of tensorflow plugins to run the transformer model.

- `OcrTransformer` - transormer encode op.

- `Decoding` - transormer decode op.

- `GatherTree` - transormer gathertree op.


## 使用说明

1.  backbone支持resnet50、resnet_vd，通过参数is_resnet_vd配置，is_resnet_vd=false对应resnet50，is_resnet_vd=true对应resnet_vd

2.  支持FP16和FP32，通过参数FP16配置

3.  图片预处理，调用ocr_transformer.py里面的images_preprocess函数，将图片转化为bin和shape文件，供模型预估

4.  fastertransformer模型预估，配置好参数is_resnet_vd、FP16、图片路径src_dir、原始模型路径model_path
	```
	python ocr_fastertransformer.py
	```

5.  将原始transformer savedmodel模型转化成fastertransformer pb模型
	```
	python transformer_model_convert.py
	```


