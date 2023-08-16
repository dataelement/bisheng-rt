import os

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2, graph_pb2, types_pb2
from tensorflow.tools.graph_transforms import TransformGraph

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        if model_path.endswith('pb'):
            with open(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(model_path, 'r') as pf:
                text_format.Parse(pf.read(), graph_def)
        tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=graph)
        return sess


def add_cast_fp32_to_fp16_node(node_in_name, graph_def, node_out_name=None):
    new_node = graph_def.node.add()
    new_node.op = 'Cast'
    new_node.name = f'{node_in_name}/Castfp16tofp32' if node_out_name is None else node_out_name
    new_node.input.extend([node_in_name])

    new_node.attr['SrcT'].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
    new_node.attr['DstT'].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    new_node.attr['Truncate'].CopyFrom(attr_value_pb2.AttrValue(b=True))


def add_cast_fp16_to_fp32_node(node_in_name, graph_def, node_out_name=None):
    new_node = graph_def.node.add()
    new_node.op = 'Cast'
    new_node.name = f'{node_in_name}/Castfp16tofp32' if node_out_name is None else node_out_name
    new_node.input.extend([node_in_name])

    new_node.attr['SrcT'].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))
    new_node.attr['DstT'].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))

    new_node.attr['Truncate'].CopyFrom(attr_value_pb2.AttrValue(b=True))
    return new_node.name


def rewrite_batch_norm_node(op_name, node, graph_def, target_type='fp16'):
    """
    Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for
    gradient calculation (See here: https://www .tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
    """
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    new_node = graph_def.node.add()
    new_node.op = 'FusedBatchNormV2' if op_name == 'FusedBatchNorm' else op_name
    new_node.name = node.name
    new_node.input.extend(node.input)

    new_node.attr['U'].CopyFrom(
        attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))

    for attr in list(node.attr.keys()):
        if attr in ['T']:
            node.attr[attr].type = dtype
        new_node.attr[attr].CopyFrom(node.attr[attr])


def rewrite_crop_and_resize_node(node, graph_def, target_type='fp16'):
    """
    Rewrite CropAndResize, input 1: box, only support float
    """

    dtype = types_pb2.DT_HALF

    new_node = graph_def.node.add()
    new_node.op = node.op
    new_node.name = node.name + '_fp16'
    new_node.input.extend(node.input)

    for attr in list(node.attr.keys()):
        if attr in ['T']:
            node.attr[attr].type = dtype
        new_node.attr[attr].CopyFrom(node.attr[attr])

    # add cast node for input 1
    node_name = add_cast_fp16_to_fp32_node(new_node.input[1], graph_def)
    new_node.input[1] = node_name

    node_name = add_cast_fp32_to_fp16_node(new_node.name, graph_def, node.name)


def rewrite_node_only_fp32(node, graph_def, target_type='fp16'):
    """
    Rewrite CropAndResize, input 1: box, only support float
    """

    dtype = types_pb2.DT_HALF

    new_node = graph_def.node.add()
    new_node.op = node.op
    new_node.name = node.name + '_fp32'
    new_node.input.extend(node.input)

    for attr in list(node.attr.keys()):
        new_node.attr[attr].CopyFrom(node.attr[attr])

    # add cast node for input 0
    node_name = add_cast_fp16_to_fp32_node(new_node.input[0], graph_def)
    new_node.input[0] = node_name

    node_name = add_cast_fp32_to_fp16_node(new_node.name, graph_def, node.name)


def convert_graph_to_fp16(model_path,
                          save_path,
                          name,
                          as_text=False,
                          target_type='fp16',
                          input_name=None,
                          output_names=None,
                          keep_fp32_node_name=None):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)
    for node in source_graph_def.node:

        if node.op in [
                'FusedBatchNorm', 'FusedBatchNormV3', 'FusedBatchNormV2'
        ]:
            rewrite_batch_norm_node(node.op, node, target_graph_def)
            continue

        if node.op in ['CropAndResize']:
            rewrite_crop_and_resize_node(node, target_graph_def)
            continue

        if node.op in ['Exp', 'Log', 'Sqrt']:
            rewrite_node_only_fp32(node, target_graph_def)
            continue

        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name
        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())

        # keep batch norm params node
        if ('BatchNorm' in node.name) or ('batch_normalization' in node.name):
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        fused_batch_norm_fp32_nodes = [
            'bn/gamma', 'bn/beta', 'bn/mean/EMA', 'bn/variance/EMA'
        ]

        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp32
            if node.name in keep_fp32_node_name:
                new_node.attr[attr].CopyFrom(node.attr[attr])
                continue

            is_fused_bn_node = False
            for n in fused_batch_norm_fp32_nodes:
                if n in node.name:
                    is_fused_bn_node = True
                    break

            if is_fused_bn_node:
                new_node.attr[attr].CopyFrom(node.attr[attr])
                continue

            if node.op in ['Const', 'Placeholder'] and attr == 'dtype':
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    node.attr[attr].type = dtype

                new_node.attr[attr].CopyFrom(node.attr[attr])

            elif node.op in ['Cast'] and attr in ['DstT', 'SrcT']:
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    node.attr[attr].type = dtype

                new_node.attr[attr].CopyFrom(node.attr[attr])

            elif node.op in ['GatherV2', 'GatherNd'] and attr in ['Tparams']:
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    node.attr[attr].type = dtype

                new_node.attr[attr].CopyFrom(node.attr[attr])

            elif node.op in ['NonMaxSuppressionV3'
                             ] and attr in ['T_threshold']:
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    node.attr[attr].type = dtype

                new_node.attr[attr].CopyFrom(node.attr[attr])

            elif attr == 'T':
                if node.attr[attr].type == types_pb2.DT_FLOAT:
                    # modify node dtype
                    node.attr[attr].type = dtype

                new_node.attr[attr].CopyFrom(node.attr[attr])

            elif attr == 'value':
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    # if float_val exists
                    if tensor.float_val:
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        new_node.attr[attr].tensor.CopyFrom(
                            tf.make_tensor_proto(float_val, dtype=dtype))
                        continue

                    # if tensor content exists
                    if tensor.tensor_content:
                        tensor_shape = [
                            x.size for x in tensor.tensor_shape.dim
                        ]
                        tensor_weights = tf.make_ndarray(tensor)
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights,
                                                    tensor_shape)
                        tensor_proto = tf.make_tensor_proto(tensor_weights,
                                                            dtype=dtype)
                        new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                        continue
                else:
                    new_node.attr[attr].CopyFrom(node.attr[attr])
            else:
                new_node.attr[attr].CopyFrom(node.attr[attr])

    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ['strip_unused_nodes', 'sort_by_execution_order']
        target_graph_def = TransformGraph(target_graph_def, input_name,
                                          output_names, transforms)
    node_set = []
    for node in target_graph_def.node:
        print('-' * 30)
        print('node:', node.op, node.name)
        for input_node in node.input:
            print('input:', input_node)

        attrs = list(node.attr.keys())
        for attr in attrs:
            if attr == 'value':
                tensor = node.attr[attr].tensor
                info = f'tensor:{tensor.dtype}'
            else:
                info = node.attr[attr]
            print('attr:', attr, info)

        node_set.append(node.op)

    print('node_set', list(set(node_set)))

    # write graph_def to model
    tf.io.write_graph(target_graph_def,
                      logdir=save_path,
                      name=name,
                      as_text=as_text)
    print('Converting done ...')


def test():

    keep_fp32_node_name = []

    sig = {
        'inputs': ['image:0'],
        'outputs': [
            'output/boxes:0',
            'output/scores:0',
            'output/boxes_cos:0',
            'output/boxes_sin:0',
            'output/masks:0',
            'output/labels:0',
        ],
    }

    model_path = '/home/hanfeng/models/layout_mrcnn/freeze.pb'
    save_path = '/home/hanfeng/models/layout_mrcnn/'
    name = 'freeze_fp16.pb'
    as_text = False
    target_type = 'fp16'
    convert_graph_to_fp16(model_path,
                          save_path,
                          name,
                          as_text=as_text,
                          target_type=target_type,
                          input_name=sig['inputs'],
                          output_names=sig['outputs'],
                          keep_fp32_node_name=keep_fp32_node_name)

    # test loading
    sess = load_graph(save_path + '/' + name)


test()
