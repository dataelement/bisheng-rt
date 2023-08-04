# -*- coding: utf-8 -*-
# File: conv2d.py
import tensorflow as tf
import numpy as np

def shape2d(a):
    """
    Ensure a 2D shape.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a):
    """
    Ensuer a 4D shape, to use with 4D symbolic functions.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 4. if ``a`` is a int, return ``[1, a, a, 1]``
            or ``[1, 1, a, a]`` depending on data_format.
    """
    s2d = shape2d(a)
    return [1, 1] + s2d


def Conv2D(name,
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_first',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    Similar to `tf.layers.Conv2D`, but with some differences:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group convolution.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    with tf.variable_scope(name):
        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

        layer = tf.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        return ret


def Conv2DTranspose(name,
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_first',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    with tf.variable_scope(name):
        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

        # Our own implementation, to avoid Keras bugs. https://github.com/tensorflow/tensorflow/issues/25946
        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Unsupported arguments due to Keras bug in TensorFlow 1.13"
        shape_dyn = tf.shape(inputs)
        strides2d = shape2d(strides)
        channels_in = inputs.shape[1]
        out_shape_dyn = tf.stack(
            [shape_dyn[0], filters,
             shape_dyn[2] * strides2d[0],
             shape_dyn[3] * strides2d[1]])
        out_shape3_sta = [filters,
                          None if inputs.shape[2] is None else inputs.shape[2] * strides2d[0],
                          None if inputs.shape[3] is None else inputs.shape[3] * strides2d[1]]

        kernel_shape = shape2d(kernel_size)
        W = tf.get_variable('kernel', kernel_shape + [filters, channels_in], initializer=kernel_initializer)
        if use_bias:
            b = tf.get_variable('bias', [filters], initializer=bias_initializer)
        conv = tf.nn.conv2d_transpose(
            inputs, W, out_shape_dyn,
            shape4d(strides),
            padding=padding.upper(),
            data_format='NCHW')
        conv.set_shape(tf.TensorShape([None] + out_shape3_sta))

        ret = tf.nn.bias_add(conv, b, data_format='NCHW') if use_bias else conv
        if activation is not None:
            ret = activation(ret)

        return ret


# def Conv2DTranspose(name,
#         inputs,
#         filters,
#         kernel_size,
#         strides=(1, 1),
#         padding='same',
#         data_format='channels_first',
#         activation=None,
#         use_bias=True,
#         kernel_initializer=None,
#         bias_initializer=tf.zeros_initializer(),
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None):
#     """
#     A wrapper around `tf.layers.Conv2DTranspose`.
#     Some differences to maintain backward-compatibility:

#     1. Default kernel initializer is variance_scaling_initializer(2.0).
#     2. Default padding is 'same'

#     Variable Names:

#     * ``W``: weights
#     * ``b``: bias
#     """
#     with tf.variable_scope(name):
#         layer = tf.keras.layers.Conv2DTranspose(
#                 filters, kernel_size, strides=strides, padding=padding,
#                 data_format=data_format, activation=activation, use_bias=use_bias,
#                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
#                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
#                 activity_regularizer=activity_regularizer)
#         ret = layer.apply(inputs)
#         return ret


def MaxPooling(name, inputs, pool_size, strides=None, padding='valid', data_format='channels_first'):
    """
    Same as `tf.layers.MaxPooling2D`. Default strides is equal to pool_size.
    """
    with tf.variable_scope(name):
        if strides is None:
            strides = pool_size
        layer = tf.layers.MaxPooling2D(pool_size, strides, padding=padding, data_format=data_format)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        return ret


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


def FullyConnected(name,
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    with tf.variable_scope(name):
        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

        inputs = batch_flatten(inputs)

        layer = tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        return ret


def DynamicLazyAxis(shape, idx):
    return lambda: shape[idx]


def StaticLazyAxis(dim):
    return lambda: dim


class StaticDynamicShape(object):
    def __init__(self, tensor):
        assert isinstance(tensor, tf.Tensor), tensor
        ndims = tensor.shape.ndims
        self.static = tensor.shape.as_list()
        if tensor.shape.is_fully_defined():
            self.dynamic = self.static[:]
        else:
            dynamic = tf.shape(tensor)
            self.dynamic = [DynamicLazyAxis(dynamic, k) for k in range(ndims)]

        for k in range(ndims):
            if self.static[k] is not None:
                self.dynamic[k] = StaticLazyAxis(self.static[k])

    def apply(self, axis, f):
        if self.static[axis] is not None:
            try:
                st = f(self.static[axis])
                self.static[axis] = st
                self.dynamic[axis] = StaticLazyAxis(st)
                return
            except TypeError:
                pass
        self.static[axis] = None
        dyn = self.dynamic[axis]
        self.dynamic[axis] = lambda: f(dyn())

    def get_static(self):
        return self.static

    @property
    def ndims(self):
        return len(self.static)

    def get_dynamic(self, axis=None):
        if axis is None:
            return [self.dynamic[k]() for k in range(self.ndims)]
        return self.dynamic[axis]()


def FixedUnPooling(name, x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.

    Args:
        x (tf.Tensor): a 4D image tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.

    Returns:
        tf.Tensor: a 4D image tensor.
    """
    with tf.variable_scope(name):
        shape = shape2d(shape)

        output_shape = StaticDynamicShape(x)
        output_shape.apply(2, lambda x: x * shape[0])
        output_shape.apply(3, lambda x: x * shape[1])

        # check unpool_mat
        if unpool_mat is None:
            mat = np.zeros(shape, dtype='float32')
            mat[0][0] = 1
            unpool_mat = tf.constant(mat, name='unpool_mat')
        elif isinstance(unpool_mat, np.ndarray):
            unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
        assert unpool_mat.shape.as_list() == list(shape)

        # perform a tensor-matrix kronecker product
        x = tf.expand_dims(x, -1)       # bchwx1
        mat = tf.expand_dims(unpool_mat, 0)  # 1xshxsw
        ret = tf.tensordot(x, mat, axes=1)  # bxcxhxwxshxsw

        ret = tf.transpose(ret, [0, 1, 2, 4, 3, 5])

        shape3_dyn = [output_shape.get_dynamic(k) for k in range(1, 4)]
        ret = tf.reshape(ret, tf.stack([-1] + shape3_dyn))

        ret.set_shape(tf.TensorShape(output_shape.get_static()))
        return ret


def get_data_format(data_format, keras_mode=True):
    if keras_mode:
        dic = {'NCHW': 'channels_first', 'NHWC': 'channels_last'}
    else:
        dic = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
    ret = dic.get(data_format, data_format)
    if ret not in dic.values():
        raise ValueError("Unknown data_format: {}".format(data_format))
    return ret


def BatchNorm(name, inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='channels_first'):

    # parse shapes
    with tf.variable_scope(name):
        data_format = get_data_format(data_format, keras_mode=False)
        shape = inputs.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4], ndims

        if axis is None:
            if ndims == 2:
                axis = 1
            else:
                axis = 1 if data_format == 'NCHW' else 3
        assert axis in [1, 3], axis
        num_chan = shape[axis]

        freeze_bn_backward = False
        # Use the builtin layer for anything except for sync-bn
        tf_args = dict(
            axis=axis,
            momentum=momentum, epsilon=epsilon,
            center=center, scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            # https://github.com/tensorflow/tensorflow/issues/10857#issuecomment-410185429
            fused=(ndims == 4 and axis in [1, 3] and not freeze_bn_backward),
            _reuse=tf.get_variable_scope().reuse)
        tf_args['virtual_batch_size'] = virtual_batch_size
        use_fp16 = inputs.dtype == tf.float16
        if use_fp16:
            # non-fused does not support fp16; fused does not support all layouts.
            # we made our best guess here
            tf_args['fused'] = True
        layer = tf.layers.BatchNormalization(**tf_args)
        ret = layer.apply(inputs, training=False, scope=tf.get_variable_scope())

        # Add EMA variables to the correct collection
        if True:
            for v in layer.non_trainable_variables:
                if isinstance(v, tf.Variable):
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
        return ret


