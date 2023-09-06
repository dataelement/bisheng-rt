# atc --model=ppocr_angle_cls_v1.onnx --framework=5 --output=ppocr_angle_cls_v1 --soc_version=Ascend310P3 --input_format=NCHW --input_shape="x:32,3,48,192"
# atc --model=ppocr_angle_cls_v1.onnx --framework=5 --output=ppocr_angle_cls_dy --soc_version=Ascend310P3 --input_shape="x:-1,3,48,192" --dynamic_dims="1;4;8;16;32" --input_format=ND

# atc --model=det_r34_vd_db_500_opset11_v1.onnx --framework=5 --output=det_r34_vd_db --soc_version=Ascend310P3 --input_format=NCHW --input_shape="x:1,3,960,960"
# atc --model=det_r34_vd_db_500_opset11_v1.onnx --framework=5 --output=det_r34_vd_db00 --soc_version=Ascend310P3 --input_format=NCHW --input_shape="x:1,3,480,960"
# atc --model=det_r34_vd_db_500_opset11_v1.onnx --framework=5 --output=det_r34_vd_db_dy --soc_version=Ascend310P3 --input_shape="x:1,3,-1,-1" --dynamic_dims="480,960;960,480;960,960" --input_format=ND

# atc --model=rec_res34_bilstm_new_opset16.onnx --framework=5 --output=rec_res34_bilstm --soc_version=Ascend310P3 --input_format=NCHW --input_shape="x:32,3,32,600"
# atc --model=rec_res34_bilstm_new_opset16.onnx --framework=5 --output=rec_res34_bilstm_dy --soc_version=Ascend310P3 --input_format=ND --input_shape="x:-1,3,32,-1" --dynamic_dims="1,200;1,400;1,600;4,200;4,400;4,600;8,200;8,400;8,600;16,200;16,400;16,600;32,200;32,400;32,600"

# atc --model=ocr_maskrcnn_freeze.pb --framework=3 --output=ocr_maskrcnn_freeze --soc_version=Ascend310P3 --input_format=NHWC --input_shape="image:1,960,960,3"
atc --model=ocr_transformer.pb --framework=3 --output=ocr_transformer --soc_version=Ascend310P3 --input_format=NHWC --input_shape="image:1,32,800,1;image_shape:1,2"