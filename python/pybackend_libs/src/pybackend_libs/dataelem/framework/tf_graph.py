# import json
import os
from typing import Any, List

import tensorflow as tf


class TFGraph(object):
    def load_pb(self, sig, variable_scope, device, model_path):
        with open(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=device)
        tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                  gpu_options=gpu_options)

        self.sess = tf.Session(config=tfconfig)
        with self.sess.graph.as_default():
            tf.import_graph_def(graph_def, name=variable_scope)

        self.ys = [
            tf.get_default_graph().get_tensor_by_name(
                os.path.join(variable_scope, n)) for n in sig['outputs']
        ]

        self.xs = [
            tf.get_default_graph().get_tensor_by_name(
                os.path.join(variable_scope, n)) for n in sig['inputs']
        ]

    def __init__(self, sig, device, **kwargs):
        model_path = kwargs.get('model_path')
        model_file = os.path.join(model_path, 'model.graphdef')
        if not os.path.exists(model_file):
            raise Exception(f'{model_file} not exists')

        variable_scope = kwargs.get('variable_scope', 'unknown')
        # model_dir = os.path.dirname(model_path)
        # sig = json.loads(os.path.join(model_dir, "sig.json"))
        self.load_pb(sig, variable_scope, device, model_file)

    def run(self, inputs: List[Any]) -> List[Any]:
        assert len(inputs) == len(self.xs)
        inputs_dict = dict((x, k) for x, k in zip(self.xs, inputs))
        outputs = self.sess.run(self.ys, feed_dict=inputs_dict)
        return outputs
