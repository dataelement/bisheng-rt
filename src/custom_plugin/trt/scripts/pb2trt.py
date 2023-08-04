import os
import json
import struct
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def save_mrcnn_weights(variables):
  dic = {}
  for var in variables:
    value = tensor_util.MakeNdarray(var.attr["value"].tensor)
    if len(value.shape) == 0:
      continue
    if str(value.dtype) != "float32":
      continue
    if "conv" not in var.name and "W" not in var.name and "b" not in var.name:
      continue

    print(str(var.name) + " " + str(value.shape) + " " + str(value.dtype))
    if 'W' in str(var.name):
      new_var_name = str(var.name).replace('W', 'kernel:0')
      dic[new_var_name] = value
    elif str(var.name)[-1] == "b":
      new_var_name = str(var.name)[:-1] + 'bias:0'
      dic[new_var_name] = value
    elif 'bn/mean/EMA' in  str(var.name):
      new_var_name = str(var.name).replace('bn/mean/EMA', 'BatchNorm/moving_mean:0')
      dic[new_var_name] = value
    elif 'bn/variance/EMA' in  str(var.name):
      new_var_name = str(var.name).replace('bn/variance/EMA', 'BatchNorm/moving_variance:0')
      dic[new_var_name] = value
    elif 'bn/gamma' in  str(var.name):
      new_var_name = str(var.name).replace('bn/gamma', 'BatchNorm/gamma:0')
      dic[new_var_name] = value
    elif 'bn/beta' in  str(var.name):
      new_var_name = str(var.name).replace('bn/beta', 'BatchNorm/beta:0')
      dic[new_var_name] = value
    else:
      dic[str(var.name)+":0"] = value
    if str(var.name) == 'maskrcnn/deconv/W':
      new_var_name = 'maskrcnn/deconv/conv2d_transpose/kernel:0'
      dic[new_var_name] = value
    if str(var.name) == 'maskrcnn/deconv/b':
      new_var_name = 'maskrcnn/deconv/conv2d_transpose/bias:0'
      dic[new_var_name] = value
  return dic

def float2str(data):
  return struct.pack('!f', data).hex()

def str2bytes(str_data):
  return bytes(str_data, encoding = 'utf-8')

def gen_mrcnn_trt_weights(dic, dst_name):
  f_write = open(dst_name, 'wb')
  f_write.write(str2bytes('\n'))
  lines = []
  for name, weights in dic.items():
    if len(weights.shape) == 4:
      weights = np.transpose(weights, [3,2,0,1])
    if len(weights.shape) == 2:
      weights = np.transpose(weights, [1, 0])

    size = 1
    for s in weights.shape:
      size *= s
    line = name + ' ' + str(0) + ' ' + str(size)
    weights = weights.reshape([-1])
    for i in range(size):
      line += ' ' + float2str(weights[i])
    lines.append(line+'\n')
  f_write.write(str2bytes(str(len(lines)) + '\n'))
  for line in lines:
    f_write.write(str2bytes(line))
  f_write.close()


def convert_maskrcnn_model(src_name, dst_name):
  with open(src_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    with sess.graph.as_default():
      tf.import_graph_def(graph_def, name="")

  save_variables = [n for n in graph_def.node if n.op=="Const"]
  variables_dic = save_mrcnn_weights(save_variables)

  gen_mrcnn_trt_weights(variables_dic, dst_name)
  sess.close()

DIC_MAP = {}
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d/kernel'] = 'layer_0/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d/bias'] = 'layer_0/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d_1/kernel'] = 'layer_0/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/conv1d_1/bias'] = 'layer_0/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/q/kernel'] = 'layer_0/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/k/kernel'] = 'layer_0/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/v/kernel'] = 'layer_0/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/self_attention/output_transform/kernel'] = 'layer_0/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/filter_layer/kernel'] = 'layer_0/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/filter_layer/bias'] = 'layer_0/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/output_layer/kernel'] = 'layer_0/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/feed_foward_network/output_layer/bias'] = 'layer_0/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/layer_normalization/layer_norm_scale'] = 'layer_0/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_0/ffn/layer_normalization/layer_norm_bias'] = 'layer_0/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/layer_normalization_1/layer_norm_scale'] = 'layer_0/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_0/self_attention/layer_normalization_1/layer_norm_bias'] = 'layer_0/attention/self/LayerNorm_2/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d/kernel'] = 'layer_1/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d/bias'] = 'layer_1/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d_1/kernel'] = 'layer_1/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/conv1d_1/bias'] = 'layer_1/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/q/kernel'] = 'layer_1/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/k/kernel'] = 'layer_1/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/v/kernel'] = 'layer_1/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/self_attention/output_transform/kernel'] = 'layer_1/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/filter_layer/kernel'] = 'layer_1/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/filter_layer/bias'] = 'layer_1/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/output_layer/kernel'] = 'layer_1/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/feed_foward_network/output_layer/bias'] = 'layer_1/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/layer_normalization/layer_norm_scale'] = 'layer_1/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_1/ffn/layer_normalization/layer_norm_bias'] = 'layer_1/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/layer_normalization/layer_norm_scale'] = 'layer_1/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_1/self_attention/layer_normalization/layer_norm_bias'] = 'layer_1/attention/self/LayerNorm_2/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d/kernel'] = 'layer_2/attention/self/conv1d/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d/bias'] = 'layer_2/attention/self/conv1d/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d_1/kernel'] = 'layer_2/attention/self/conv1d_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/conv1d_1/bias'] = 'layer_2/attention/self/conv1d_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/q/kernel'] = 'layer_2/attention/self/query/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/k/kernel'] = 'layer_2/attention/self/key/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/v/kernel'] = 'layer_2/attention/self/value/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/self_attention/output_transform/kernel'] = 'layer_2/attention/self/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/filter_layer/kernel'] = 'layer_2/ffn/dense/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/filter_layer/bias'] = 'layer_2/ffn/dense/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/output_layer/kernel'] = 'layer_2/ffn/dense_1/kernel:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/feed_foward_network/output_layer/bias'] = 'layer_2/ffn/dense_1/bias:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/layer_normalization/layer_norm_scale'] = 'layer_2/ffn/LayerNorm/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_2/ffn/layer_normalization/layer_norm_bias'] = 'layer_2/ffn/LayerNorm/beta:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/layer_normalization/layer_norm_scale'] = 'layer_2/attention/self/LayerNorm_2/gamma:0'
DIC_MAP['Transformer/encoder_stack/layer_2/self_attention/layer_normalization/layer_norm_bias'] = 'layer_2/attention/self/LayerNorm_2/beta:0'

def save_transformer_weights(variables):
  dic = {}
  for var in variables:
    value = tensor_util.MakeNdarray(var.attr["value"].tensor)
    if len(value.shape) == 0 or value.dtype == np.object_:
      continue
    if str(value.dtype) != "float32":
      continue
    print(str(var.name) + " " + str(value.shape) + " " + str(value.dtype))
    if str(var.name) == 'Transformer/encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_scale':
      dic['layer_0/attention/self/LayerNorm/gamma:0'] = value
      dic['layer_0/attention/self/LayerNorm_1/gamma:0'] = value
      dic['layer_1/attention/self/LayerNorm/gamma:0'] = value
      dic['layer_1/attention/self/LayerNorm_1/gamma:0'] = value
      dic['layer_2/attention/self/LayerNorm/gamma:0'] = value
      dic['layer_2/attention/self/LayerNorm_1/gamma:0'] = value
      dic['LayerNorm/gamma:0'] = value
    elif str(var.name) == 'Transformer/encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_bias':
      dic['layer_0/attention/self/LayerNorm/beta:0'] = value
      dic['layer_0/attention/self/LayerNorm_1/beta:0'] = value
      dic['layer_1/attention/self/LayerNorm/beta:0'] = value
      dic['layer_1/attention/self/LayerNorm_1/beta:0'] = value
      dic['layer_2/attention/self/LayerNorm/beta:0'] = value
      dic['layer_2/attention/self/LayerNorm_1/beta:0'] = value
      dic['LayerNorm/beta:0'] = value
    else:
      if var.name in DIC_MAP:
        dic[DIC_MAP[var.name]] = value
      else:
        dic[var.name+":0"] = value
  return dic

def gen_transformer_trt_weights(variables_dic, dst_name, ignore=None):
  f_write = open(dst_name, 'wb')
  f_write.write(str2bytes('\n'))
  lines = []
  for name, weights in variables_dic.items():
    if ignore == None or ignore not in name:
      # print(name, weights.shape)
      if 'conv2d' in name and 'kernel' in name:
        weights = np.transpose(weights, [3,2,0,1])
      elif 'post_conv/kernel' in name:
        weights = np.transpose(weights, [3,2,0,1])
      elif 'conv1d' in name and 'kernel' in name:
        weights = np.transpose(weights, [2,1,0])
      elif 'ocr_ctc/logits/kernel' in name:
        weights = np.transpose(weights, [1,0])
      elif ('dense' in name or 'query' in name or 'key' in name or 'value' in name) and 'kernel' in name:
        weights = np.transpose(weights, [1,0])

    size = 1
    for s in weights.shape:
      size *= s
    line = name + ' ' + str(0) + ' ' + str(size)
    weights = weights.reshape([-1])
    for i in range(size):
      line += ' ' + float2str(weights[i])
    lines.append(line+'\n')
  f_write.write(str2bytes(str(len(lines)) + '\n'))
  for line in lines:
    f_write.write(str2bytes(line))
  f_write.close()

def convert_transformer_model(src_name, dst_name):
  with open(src_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    with sess.graph.as_default():
      tf.import_graph_def(graph_def, name="mrcnn")

  save_variables = [n for n in graph_def.node if n.op=="Const"]
  variables_dic = save_transformer_weights(save_variables)

  gen_transformer_trt_weights(variables_dic, dst_name, ignore='layer_')
  sess.close()

def gen_det_wts_from_pb():
  names = ["det_acnm", "det_acno", "det_dxje", "det_pzrq", "det_xxje",  "general_text_det_mrcnn_v1.0", "mrcnn_cn_eng_Insv1", "mrcnn-v5.1", "std_checkbox"]
  src_dir = "/home/liuqingjie/models/SDK2.x/pb/OCR-DETECTION-MODELS"
  dst_dir = "/home/liuqingjie/models/SDK2.x/wts/OCR-DETECTION-MODELS"
  for name in names:
    src_name = os.path.join(src_dir, name+"_tf", "1", "model.graphdef")
    if not os.path.exists(os.path.join(dst_dir, name, "1")):
      os.makedirs(os.path.join(dst_dir, name, "1"), exist_ok=True)
    dst_name = os.path.join(dst_dir, name, "1", "model.wts")
    convert_maskrcnn_model(src_name, dst_name)

def gen_rec_wts_from_pb():
  names0 = ["recog_acnm", "recog_acno", "recog_dxje", "recog_pzrq", "recog_xxje", "transformer-rare-v1.3", "transInsEnglish"]
  names1 = ["transformer-blank-v0.2", "transformer-hand-v1.16", "transformer-v2.8-gamma"]
  src_dir = "/home/liuqingjie/models/SDK2.x/pb/OCR-RECOGNITION-MODELS"
  dst_dir = "/home/liuqingjie/models/SDK2.x/wts/OCR-RECOGNITION-MODELS"
  for name in names0:
    src_name = os.path.join(src_dir, name+"_tf", "1", "model.graphdef")
    if not os.path.exists(os.path.join(dst_dir, name, "1")):
      os.makedirs(os.path.join(dst_dir, name, "1"), exist_ok=True)
    dst_name = os.path.join(dst_dir, name, "1", "model.wts")
    convert_transformer_model(src_name, dst_name)
  for name in names1:
    src_name = os.path.join(src_dir, name+"_tf", "1", "model.pb")
    if not os.path.exists(os.path.join(dst_dir, name, "1")):
      os.makedirs(os.path.join(dst_dir, name, "1"), exist_ok=True)
    dst_name = os.path.join(dst_dir, name, "1", "model.wts")
    convert_transformer_model(src_name, dst_name)

def gen_det_configs():
  names = ["det_acnm", "det_acno", "det_dxje", "det_pzrq", "det_xxje",  "general_text_det_mrcnn_v1.0", "mrcnn_cn_eng_Insv1", "mrcnn-v5.1", "std_checkbox"]
  config_file = "mrcnn_config.json"
  dst_dir = "det_configs"
  ComputeCapability = "ComputeCapability8.6"
  src_wts_dir = "/home/liuqingjie/models/SDK2.x/wts/OCR-DETECTION-MODELS"
  dst_wts_dir = "/home/liuqingjie/models/SDK2.x/trt/"+ComputeCapability+"/OCR-DETECTION-MODELS"
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
  
  dic = {}
  with open(config_file, "r") as f:
    dic = json.load(f)
  for name in names:
    dic["weightsFile"] = os.path.join(src_wts_dir, name, "1", "model.wts")
    dic["engineFile"] = os.path.join(dst_wts_dir, name+"_trt", "1", "model.plan")
    with open(os.path.join(dst_dir, name+".json"), "w") as f:
      f.write(json.dumps(dic, ensure_ascii=False))

def gen_rec_configs():
  #"transInsEnglish"暂时不支持
  names = ["recog_acnm", "recog_acno", "recog_dxje", "recog_pzrq", "recog_xxje", "transformer-blank-v0.2", "transformer-hand-v1.16", "transformer-rare-v1.3", "transformer-v2.8-gamma"]
  config_file = "transformer_config.json"
  dst_dir = "rec_configs"
  ComputeCapability = "ComputeCapability8.6"
  src_wts_dir = "/home/liuqingjie/models/SDK2.x/wts/OCR-RECOGNITION-MODELS"
  dst_wts_dir = "/home/liuqingjie/models/SDK2.x/trt/"+ComputeCapability+"/OCR-RECOGNITION-MODELS"
  if not os.path.exists(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
  
  dic = {}
  with open(config_file, "r") as f:
    dic = json.load(f)
  for name in names:
    dic["weightsFile"] = os.path.join(src_wts_dir, name, "1", "model.wts")
    dic["engineFile"] = os.path.join(dst_wts_dir, name+"_trt", "1", "model.plan")
    with open(os.path.join(dst_dir, name+".json"), "w") as f:
      f.write(json.dumps(dic, ensure_ascii=False))

def gen_det_trt_from_wts():
  names = ["det_acnm", "det_acno", "det_dxje", "det_pzrq", "det_xxje",  "general_text_det_mrcnn_v1.0", "mrcnn_cn_eng_Insv1", "mrcnn-v5.1", "std_checkbox"]
  config_dir = "det_configs"
  device_id = "3"
  for name in names:
    config_name = os.path.join(config_dir, name+".json")

    dic = {}
    with open(config_name, "r") as f:
      dic = json.load(f)
      dst_dir = os.path.dirname(dic["engineFile"])
      if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    cmd_str = "./sample_trt_mask_rcnn " + device_id + " " + config_name
    os.system(cmd_str)
    print("gen trt model success:", name)

def gen_rec_trt_from_wts():
   #"transInsEnglish"暂时不支持
  names = ["recog_acnm", "recog_acno", "recog_dxje", "recog_pzrq", "recog_xxje", "transformer-blank-v0.2", "transformer-hand-v1.16", "transformer-rare-v1.3", "transformer-v2.8-gamma"]
  config_dir = "rec_configs"
  device_id = "3"
  for name in names:
    config_name = os.path.join(config_dir, name+".json")

    dic = {}
    with open(config_name, "r") as f:
      dic = json.load(f)
      dst_dir = os.path.dirname(dic["engineFile"])
      if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    cmd_str = "./sample_trt_transformer " + device_id + " " + config_name
    os.system(cmd_str)
    print("gen trt model success:", name)

if __name__ == '__main__':
  gen_det_configs()
  gen_rec_configs()
  gen_det_wts_from_pb()
  gen_rec_wts_from_pb()
  gen_det_trt_from_wts()
  gen_rec_trt_from_wts()