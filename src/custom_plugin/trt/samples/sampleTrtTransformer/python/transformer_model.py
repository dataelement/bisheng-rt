import tensorflow as tf
import numpy as np
import math
import six
import os
import time
from datetime import datetime
from position import SinusoidalPositionEncoder
import resnet_vd
np.set_printoptions(threshold=1e6, suppress=True)

initializer_range = 0.02

def create_initializer(initializer_range=0.02, data_type=tf.float32):
    return tf.truncated_normal_initializer(stddev=initializer_range, dtype=data_type)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# backbone
def batch_norm(inputs, training, dtype, data_format):
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True,
      beta_initializer=create_initializer(initializer_range, dtype),
      gamma_initializer=create_initializer(initializer_range, dtype),
      moving_mean_initializer=create_initializer(initializer_range, dtype),
      moving_variance_initializer=create_initializer(initializer_range, dtype))


def fixed_padding(inputs, kernel_size, data_format):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, dtype, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=create_initializer(initializer_range, dtype),
      data_format=data_format)


def building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       dtype, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, dtype, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      dtype=dtype, data_format=data_format)

    inputs = batch_norm(inputs, training, dtype, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      dtype=dtype, data_format=data_format)

    return inputs + shortcut


def bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides, dtype, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, dtype, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=1,
                                  strides=1,
                                  dtype=dtype,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, dtype, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=3,
                                  strides=strides,
                                  dtype=dtype,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, dtype, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=4 * filters,
                                  kernel_size=1,
                                  strides=1,
                                  dtype=dtype,
                                  data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, blocks, strides, training, dtype, name, data_format):
    filters_out = filters * 4 if bottleneck else filters
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1,
                                    strides=strides, dtype=dtype, data_format=data_format)

    if bottleneck:
        inputs = bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides,
                                     dtype, data_format)
    else:
        inputs = building_block_v2(inputs, filters, training, projection_shortcut, strides,
                                   dtype, data_format)

    for _ in range(1, blocks):
        if bottleneck:
            inputs = bottleneck_block_v2(inputs, filters, training, None, 1, dtype, data_format)
        else:
            inputs = building_block_v2(inputs, filters, training, None, 1, dtype, data_format)

    return tf.identity(inputs, name)


def build_resnet31(inputs, dtype):
    num_filters = 16
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=num_filters, kernel_size=3,
        strides=1, dtype=dtype, data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_conv')

    block_sizes = [5, 5, 5]
    block_strides = [1, 2, 2]
    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=False,
                             blocks=num_blocks, strides=block_strides[i], training=False, dtype=dtype,
                             name='block_layer{}'.format(i + 1), data_format='channels_first')
        num_filters *= 2

    inputs = batch_norm(inputs, False, dtype, 'channels_first')
    inputs = tf.nn.relu(inputs)
    return inputs


def build_resnet50(inputs, dtype):
    kernel_size = 7
    conv_stride = 2
    num_filters = 64
    inputs = conv2d_fixed_padding(inputs=inputs, filters=num_filters, kernel_size=kernel_size,
                                  strides=conv_stride, dtype=dtype, data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_conv')
    block_sizes = [3, 4, 6, 3]
    block_strides = [1, 2, 2, 1]
    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=True,
                             blocks=num_blocks, strides=block_strides[i], training=False, dtype=dtype,
                             name='block_layer{}'.format(i + 1), data_format='channels_first')
        num_filters *= 2

    inputs = batch_norm(inputs, False, dtype, 'channels_first')
    inputs = tf.nn.relu(inputs)
    return inputs


def build_resnet_vd(inputs, dtype, data_format='channels_first'):
    backbone = resnet_vd.resnet_vd(data_format)
    inputs = backbone(inputs, False, dtype)
    return inputs


def build_cnn(inputs, hidden_dim, dtype=tf.float32, is_resnet_vd=False):
    if is_resnet_vd:
        inputs = build_resnet_vd(inputs, dtype)
    else:
        with tf.variable_scope('resnet_model'):
            inputs = build_resnet50(inputs, dtype)

    debug_outputs = inputs
    inputs = tf.layers.conv2d(inputs, filters=512, kernel_size=1, strides=1,
                            name='ocr_transformer/backbone/post_conv', data_format='channels_first',
                            bias_initializer=create_initializer(initializer_range, dtype),
                            kernel_initializer=create_initializer(initializer_range, dtype))
    inputs = tf.layers.batch_normalization(inputs, training=False, axis=1,
                                         fused=True, name="post_bn",
                                         beta_initializer=create_initializer(initializer_range, dtype),
                                         gamma_initializer=create_initializer(initializer_range, dtype),
                                         moving_mean_initializer=create_initializer(initializer_range, dtype),
                                         moving_variance_initializer=create_initializer(initializer_range, dtype))

    inputs = tf.nn.relu(inputs)

    b, c, h, w = inputs.get_shape().as_list()
    b = tf.shape(inputs)[0]
    w = tf.shape(inputs)[3]
    inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
    inputs = tf.reshape(inputs, [b, w, h*c])

    inputs = tf.layers.dense(inputs, hidden_dim, name='ocr_transformer/backbone/dense',
                           bias_initializer=create_initializer(initializer_range, dtype),
                           kernel_initializer=create_initializer(initializer_range, dtype))
    inputs = tf.nn.relu(inputs)
    return inputs, debug_outputs

# encode
def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def layer_norm(x, scope, dtype, epsilon=1e-6, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        scale = tf.get_variable("gamma", [512], dtype=dtype, initializer=tf.ones_initializer(dtype=dtype))
        bias = tf.get_variable("beta", [512], dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        # 平方fp16会溢出，先要转化成fp32
        if dtype == tf.float16:
            mean = tf.cast(mean, tf.float32)
            x = tf.cast(x, tf.float32)

        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        if dtype == tf.float16:
            mean = tf.cast(mean, dtype)
            x = tf.cast(x, dtype)
            norm_x = (x - mean) * tf.cast(tf.rsqrt(variance + epsilon), dtype)
        else:
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias


def get_padding(inputs, inputs_unpadded_length, dtype):
    with tf.name_scope("padding"):
        input_shape = tf.shape(inputs)
        indexs = tf.tile(tf.range(input_shape[1]), tf.expand_dims(input_shape[0], axis=0))
        indexs = tf.reshape(indexs, input_shape[:2])  # shape: [batch_size, input_length]
        inputs_unpadded_length = tf.tile(inputs_unpadded_length, tf.stack([1, input_shape[1]]))
        conditions = indexs < inputs_unpadded_length
        return tf.cast(conditions, dtype=dtype)


def get_position_encoding(length, hidden_size, dtype, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return tf.cast(signal, dtype=dtype)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    tf_datatype=tf.float32):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        use_bias=False,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        use_bias=False,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        use_bias=False,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # attention_scores = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        factor = -1e9 if tf_datatype == tf.float32 else -5e4
        adder = tf.cast((1.0 - tf.cast(attention_mask, tf.float32)) * factor, tf_datatype)

        attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)

    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    context_layer = tf.matmul(attention_probs, value_layer)

    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    debug_output = context_layer

    return context_layer, debug_output


def tf_encoder(input_tensor,
               encoder_args,
               attention_mask,
               ffn_mask,
               initializer_range=0.02):
    intermediate_size = encoder_args.hidden_dim * 2
    if encoder_args.hidden_dim % encoder_args.head_num != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (encoder_args.hidden_dim, encoder_args.head_num))

    attention_head_size = int(encoder_args.hidden_dim / encoder_args.head_num)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    prev_output = input_tensor
    for layer_idx in range(encoder_args.num_layer):
        with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    short_cut = layer_input

                    layer_input = tf.layers.conv1d(layer_input, encoder_args.hidden_dim,
                        kernel_size=3, strides=1, padding='same', dilation_rate=1, activation=tf.nn.relu,
                        bias_initializer=create_initializer(initializer_range, encoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

                    layer_input = layer_norm(layer_input, 'LayerNorm', encoder_args.dtype)

                    layer_input = tf.layers.conv1d(layer_input, encoder_args.hidden_dim,
                        kernel_size=3, strides=1, padding='same', dilation_rate=1, activation=tf.nn.relu,
                        bias_initializer=create_initializer(initializer_range, encoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

                    layer_input = layer_norm(layer_input, 'LayerNorm_1', encoder_args.dtype)

                    layer_input += short_cut

                    short_cut = layer_input
                    layer_input = layer_norm(layer_input, 'LayerNorm_2', encoder_args.dtype)

                    layer_input, _ = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=encoder_args.head_num,
                        size_per_head=encoder_args.size_per_head,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=False,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length,
                        tf_datatype=encoder_args.dtype)

                    layer_input = tf.layers.dense(layer_input, encoder_args.hidden_dim, use_bias=False,
                        kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

                    layer_input += short_cut

            with tf.variable_scope("ffn"):
                shortcut = layer_input
                layer_input = layer_norm(layer_input, 'LayerNorm', encoder_args.dtype)

                layer_input = tf.layers.dense(layer_input, intermediate_size,
                                            activation=tf.nn.relu, use_bias=True,
                                            bias_initializer=create_initializer( initializer_range, encoder_args.dtype),
                                            kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

                layer_input = tf.layers.dense(layer_input, encoder_args.hidden_dim, use_bias=True,
                                            bias_initializer=create_initializer(initializer_range, encoder_args.dtype),
                                            kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

                layer_input = tf.multiply(layer_input, ffn_mask)
                prev_output = layer_input + shortcut

            debug_output = prev_output

    return layer_norm(prev_output, 'LayerNorm', encoder_args.dtype), debug_output


# decode
def layer_norm_v2(x, scope, dtype, epsilon=1e-6, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        scale = tf.get_variable("layer_norm_scale", [512], dtype=dtype, initializer=tf.ones_initializer(dtype=dtype))
        bias = tf.get_variable("layer_norm_bias", [512], dtype=dtype, initializer=tf.zeros_initializer(dtype=dtype))
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        # 平方fp16会溢出，先要转化成fp32
        if dtype == tf.float16:
            mean = tf.cast(mean, tf.float32)
            x = tf.cast(x, tf.float32)

        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        if dtype == tf.float16:
            mean = tf.cast(mean, dtype)
            x = tf.cast(x, dtype)
            norm_x = (x - mean) * tf.cast(tf.rsqrt(variance + epsilon), dtype)
        else:
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias


def _get_shape_invariants(tensor):
      """Returns the shape of the tensor but sets middle dims to None."""
      if isinstance(tensor, tf.TensorArray):
        shape = None
      else:
        shape = tensor.shape.as_list()
        for i in range(1, len(shape) - 1):
          shape[i] = None
      return tf.TensorShape(shape)


def build_sequence_mask(sequence_length,
                        num_heads=None,
                        maximum_length=None,
                        data_type=tf.float32):
    """Builds the dot product mask.

    Args:
      sequence_length: The sequence length.
      num_heads: The number of heads.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The type of the mask tensor.

    Returns:
      A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[batch_size, 1, 1, max_length]``.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=maximum_length, dtype=data_type)
    mask = tf.expand_dims(mask, axis=1)
    if num_heads is not None:
        mask = tf.expand_dims(mask, axis=1)
    return mask


def tf_decoder(decoder_args,
               inputs,
               memory,
               memory_sequence_length,
               step,
               batch_size,
               cache=None):

    # if memory is not None and not tf.contrib.framework.nest.is_sequence(memory):
    #     memory = (memory,)
    #     if memory_mask is not None:
    #         if not tf.contrib.framework.nest.is_sequence(memory_mask):
    #             memory_mask = (memory_mask,)
    #         memory_mask = [mask for mask in memory_mask]

    if memory is not None and not tf.contrib.framework.nest.is_sequence(memory):
        memory = (memory,)
        if memory_sequence_length is not None:
            if not tf.contrib.framework.nest.is_sequence(memory_sequence_length):
                memory_sequence_length = (memory_sequence_length,)
            memory_mask = [
                build_sequence_mask(
                    length, num_heads=decoder_args.head_num, maximum_length=tf.shape(m)[1], data_type=decoder_args.dtype)
                for m, length in zip(memory, memory_sequence_length)]

    for l in range(decoder_args.num_layer):
        layer_name = "layer_{}".format(l)
        layer_cache = cache[layer_name] if cache is not None else None

        with tf.variable_scope("layer_%d" % l):
            with tf.variable_scope("self_attention"):
                norm_inputs = layer_norm_v2(inputs, 'layer_normalization', decoder_args.dtype)

                with tf.variable_scope("self_attention"):
                    queries = tf.layers.dense(
                        norm_inputs,
                        decoder_args.hidden_dim,
                        activation=None,
                        name="q",
                        use_bias=False,
                        bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                    keys = tf.layers.dense(
                        norm_inputs,
                        decoder_args.hidden_dim,
                        activation=None,
                        name="k",
                        use_bias=False,
                        bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                    values = tf.layers.dense(
                        norm_inputs,
                        decoder_args.hidden_dim,
                        activation=None,
                        name="v",
                        use_bias=False,
                        bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                    keys = tf.reshape(keys, [batch_size * decoder_args.beam_width,
                                             1, decoder_args.head_num, decoder_args.size_per_head])
                    keys = tf.transpose(keys, [0, 2, 1, 3])
                    values = tf.reshape(values, [batch_size * decoder_args.beam_width, 1,
                                        decoder_args.head_num, decoder_args.size_per_head])
                    values = tf.transpose(values, [0, 2, 1, 3])

                    keys = tf.concat([layer_cache["self_keys"], keys], axis=2)
                    values = tf.concat([layer_cache["self_values"], values], axis=2)
                    layer_cache["self_keys"] = keys
                    layer_cache["self_values"] = values

                    queries = tf.reshape(queries, [
                                         batch_size * decoder_args.beam_width, 1, decoder_args.head_num, decoder_args.size_per_head])
                    queries = tf.transpose(queries, [0, 2, 1, 3])
                    queries *= (decoder_args.size_per_head)**-0.5

                    dot = tf.matmul(queries, keys, transpose_b=True)

                    attn = tf.cast(tf.nn.softmax(tf.cast(dot, decoder_args.dtype)), dot.dtype)
                    context = tf.matmul(attn, values)
                    context = tf.transpose(context, [0, 2, 1, 3])
                    context = tf.reshape(context, [
                                         batch_size * decoder_args.beam_width, 1, decoder_args.head_num * decoder_args.size_per_head])

                    outputs = tf.layers.dense(context, decoder_args.hidden_dim, use_bias=False, name="output_transform",
                                              kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                # drop_and_add
                input_dim = inputs.get_shape().as_list()[-1]
                output_dim = outputs.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    outputs += inputs
                last_context = outputs

            if memory is not None:
                for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
                    memory_cache = layer_cache["memory"][i] if layer_cache is not None else None

                    with tf.variable_scope("encdec_attention"):
                        norm_inputs = layer_norm_v2(last_context, 'layer_normalization', decoder_args.dtype)

                        with tf.variable_scope("attention"):
                            queries = tf.layers.dense(
                                norm_inputs,
                                decoder_args.hidden_dim,
                                activation=None,
                                name="q",
                                use_bias=False,
                                bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                                kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                            def _project_and_split():
                                keys = tf.layers.dense(
                                    mem,
                                    decoder_args.hidden_dim,
                                    activation=None,
                                    name="k",
                                    use_bias=False,
                                    bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                                    kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                                values = tf.layers.dense(
                                    mem,
                                    decoder_args.hidden_dim,
                                    activation=None,
                                    name="v",
                                    use_bias=False,
                                    bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                                    kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                                keys = tf.reshape(keys, [batch_size * decoder_args.beam_width, tf.shape(keys)[1],
                                                         decoder_args.head_num, decoder_args.size_per_head])
                                keys = tf.transpose(keys, [0, 2, 1, 3])
                                values = tf.reshape(values, [batch_size * decoder_args.beam_width, tf.shape(values)[1],
                                                             decoder_args.head_num, decoder_args.size_per_head])
                                values = tf.transpose(values, [0, 2, 1, 3])

                                return keys, values

                            keys, values = tf.cond(
                                tf.equal(
                                    tf.shape(memory_cache["memory_keys"])[2], 0),
                                true_fn=_project_and_split,
                                false_fn=lambda: (memory_cache["memory_keys"], memory_cache["memory_values"]))

                            memory_cache["memory_keys"] = keys
                            memory_cache["memory_values"] = values

                            queries = tf.reshape(queries, [batch_size * decoder_args.beam_width, 1,
                                                           decoder_args.head_num, decoder_args.size_per_head])
                            queries = tf.transpose(queries, [0, 2, 1, 3])
                            queries *= (decoder_args.size_per_head)**-0.5

                            dot = tf.matmul(queries, keys, transpose_b=True)

                            dot = tf.cast(tf.cast(dot, decoder_args.dtype) * mask +
                                          ((1.0 - mask) * decoder_args.dtype.min), dot.dtype)

                            attn = tf.cast(tf.nn.softmax(tf.cast(dot, decoder_args.dtype)), dot.dtype)
                            context = tf.matmul(attn, values)
                            context = tf.transpose(context, [0, 2, 1, 3])
                            context = tf.reshape(context, [batch_size * decoder_args.beam_width, 1,
                                                           decoder_args.head_num * decoder_args.size_per_head])

                            context = tf.layers.dense(context, decoder_args.hidden_dim, use_bias=False,
                                                      name="output_transform",
                                                      kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                        # drop_and_add
                        input_dim = last_context.get_shape().as_list()[-1]
                        output_dim = context.get_shape().as_list()[-1]
                        if input_dim == output_dim:
                            context += last_context

            with tf.variable_scope("ffn"):
                # forward
                normed_last_context = layer_norm_v2(context, 'layer_normalization', decoder_args.dtype)
                input_dim = normed_last_context.get_shape().as_list()[-1]
                with tf.variable_scope("feed_foward_network"):
                    inner = tf.layers.dense(normed_last_context, decoder_args.hidden_dim * 2,
                                            activation=tf.nn.relu, use_bias=True, name='filter_layer',
                                            bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                                            kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                    transformed = tf.layers.dense(inner, input_dim, use_bias=True, name='output_layer',
                                            bias_initializer=create_initializer(initializer_range, decoder_args.dtype),
                                            kernel_initializer=create_initializer(initializer_range, decoder_args.dtype))

                # drop_and_add
                input_dim = context.get_shape().as_list()[-1]
                output_dim = transformed.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    transformed += context

        inputs = transformed
    outputs = inputs
    debug_output = outputs
    return outputs, debug_output


def init_tf_cache(batch_size,
                  head_num,
                  size_per_head,
                  num_layer,
                  vocab_size,
                  dtype,
                  num_sources=1):
    cache = {}
    for l in range(num_layer):
        proj_cache_shape = [batch_size, head_num, 0, size_per_head]
        layer_cache = {}
        layer_cache["memory"] = [
            {
                "memory_keys": tf.zeros(proj_cache_shape, dtype=dtype, name="memory_keys"),
                "memory_values": tf.zeros(proj_cache_shape, dtype=dtype, name="memory_values")
            } for _ in range(num_sources)]
        layer_cache["self_keys"] = tf.zeros(proj_cache_shape, dtype=dtype, name="self_keys")
        layer_cache["self_values"] = tf.zeros(proj_cache_shape, dtype=dtype, name="self_values")
        cache["layer_{}".format(l)] = layer_cache
    mask = tf.ones([batch_size, vocab_size], dtype=dtype)
    cache["mask"] = mask
    return cache


def initialize_decoding_variables(decoding_args, batch_size):

    start_ids = tf.fill([batch_size * decoding_args.decoder_args.beam_width],
                         decoding_args.start_id)  # [batch_size * beam_width]

    step = tf.constant(0, dtype=tf.int32)
    # save the output ids for each step
    outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    cache = init_tf_cache(batch_size * decoding_args.decoder_args.beam_width,
                          decoding_args.decoder_args.head_num, decoding_args.decoder_args.size_per_head,
                          decoding_args.decoder_args.num_layer, decoding_args.vocab_size,
                          dtype=decoding_args.decoder_args.dtype, num_sources=1)

    finished = tf.zeros([batch_size * decoding_args.decoder_args.beam_width],
                        dtype=tf.bool)  # [batch_size * beam_width], record that a sentence is finished or not
    initial_log_probs = tf.cast(tf.tile([0.] + [-float("inf")] * (decoding_args.decoder_args.beam_width - 1),
                                        [batch_size]), dtype=tf.float32)
                                        # [batch_size * beam_width]
    # [batch_size * beam_width], record the lengths of all sentences
    sequence_lengths = tf.zeros([batch_size * decoding_args.decoder_args.beam_width],
                                dtype=tf.int32)
    # record the beam search indices, used for rebuild the whole sentence in the final
    parent_ids = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    extra_vars = tuple([parent_ids, sequence_lengths])

    return start_ids, step, outputs, cache, finished, initial_log_probs, sequence_lengths, extra_vars


def beam_search(beam_width,
                vocab_size,
                step,
                log_probs,
                cum_log_probs,
                finished,
                cache,
                extra_vars,
                op_self_cache=None):

    parent_ids = extra_vars[0]
    sequence_lengths = extra_vars[1]

    # [batch_size * beam_width, vocab_size] + [batch_size * beam_width], has to broadcast
    total_probs = log_probs + tf.expand_dims(cum_log_probs, 1)
    # [batch_size, beam_width * vocab_size], can skip in cuda
    total_probs = tf.reshape(total_probs, [-1, beam_width * vocab_size])

    # both shapes are: [batch_size, beam_width]
    _, sample_ids = tf.nn.top_k(total_probs, beam_width)
    # [batch_size * beam_width], can skip in cuda
    sample_ids = tf.reshape(sample_ids, [-1])
    debug_output = sample_ids
    word_ids = sample_ids % vocab_size  # [batch_size * beam_width]
    beam_ids = sample_ids // vocab_size  # [batch_size * beam_width]
    # [batch_size * beam_width]
    # beam_indices = (tf.range(sample_ids.shape[0]) // beam_width) * beam_width + beam_ids
    beam_indices = (tf.range(tf.shape(sample_ids)[0]) // beam_width) * beam_width + beam_ids

    sequence_lengths = tf.where(
        finished, x=sequence_lengths, y=sequence_lengths + 1)

    cum_log_probs_old = tf.gather(cum_log_probs, beam_indices)

    # [batch_size * beam_width]
    # batch_pos = tf.range(sample_ids.shape[0]) // beam_width
    batch_pos = tf.range(tf.shape(sample_ids)[0]) // beam_width
    cum_log_probs = tf.gather_nd(total_probs, tf.stack(
        [batch_pos, sample_ids], axis=-1))  # [batch_size * beam_width]
    finished = tf.gather(finished, beam_indices)
    sequence_lengths = tf.gather(sequence_lengths, beam_indices)

    cache = tf.contrib.framework.nest.map_structure(
            lambda s: tf.gather(s, beam_indices), cache)
    if op_self_cache != None:
        op_self_cache = tf.contrib.framework.nest.map_structure(
            lambda s: tf.gather(s, beam_indices, axis=3), op_self_cache)

    parent_ids = parent_ids.write(step, beam_ids)
    extra_vars = [parent_ids, sequence_lengths]

    return word_ids, cum_log_probs, finished, cache, tuple(extra_vars), op_self_cache, cum_log_probs_old, debug_output


def finalize(beam_width, parent_ids, sequence_lengths, outputs, end_id, max_seq_len=None):
    maximum_lengths = tf.reduce_max(tf.reshape(sequence_lengths, [-1, beam_width]), axis=-1)
    if max_seq_len != None:
        array_shape = [max_seq_len, -1, beam_width]
    else:
        # array_shape = [maximum_lengths[0], -1, beam_width]
        array_shape = [tf.reduce_max(sequence_lengths), -1, beam_width]

    step_ids = tf.reshape(outputs, array_shape)
    parent_ids = tf.reshape(parent_ids, array_shape)

    ids = tf.contrib.seq2seq.gather_tree(step_ids, parent_ids, maximum_lengths, end_id)

    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)
    return ids, lengths


def init_op_cache(decoder_args, input_length):
    self_cache = tf.zeros([decoder_args.num_layer, 2, 0, decoder_args.batch_size * decoder_args.beam_width,
                           decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_self_caches")
    mem_cache = tf.zeros([decoder_args.num_layer, 2, decoder_args.batch_size * decoder_args.beam_width,
                          input_length, decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_memory_caches")

    return self_cache, mem_cache


def tf_decoding(memory_tensor,
                memory_sequence_length,
                decoding_args):

    # extra_decode_length = 10
    extra_decode_length = 0

    with tf.variable_scope("Transformer", reuse=tf.AUTO_REUSE):
        # copy memory and memory_sequence_length by beam_width times
        # if memory is [a, b, c], beam_width = 3, then the result is: [a a a b b b c c c ]
        batch_size = tf.shape(memory_tensor)[0]
        # batch_size = decoding_args.decoder_args.batch_size
        input_length = tf.shape(memory_tensor)[1]
        max_decode_length = input_length + extra_decode_length

        extended_memory = tf.contrib.seq2seq.tile_batch(
            memory_tensor, multiplier=decoding_args.decoder_args.beam_width)
        extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
            memory_sequence_length, multiplier=decoding_args.decoder_args.beam_width)

        with tf.variable_scope("embedding_shared_weights/embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            embedding_table = tf.get_variable("weights", [decoding_args.vocab_size, decoding_args.decoder_args.hidden_dim],
                                              dtype=decoding_args.decoder_args.dtype,
                                              initializer=tf.random_normal_initializer(0.,
                                              decoding_args.decoder_args.hidden_dim**-0.5,
                                              dtype=decoding_args.decoder_args.dtype))

        def _cond(word_ids, cum_log_probs, finished, step, outputs, debug_output, my_cache, extra_vars, op_self_cache, op_mem_cache):
            return tf.reduce_any(tf.logical_not(finished))

        def _body(word_ids, cum_log_probs, finished, step, outputs, debug_output, my_cache, extra_vars, op_self_cache, op_mem_cache):
            # [batch_size * beam_width, hidden_dim]
            mask = tf.to_float(tf.not_equal(word_ids, 0))
            inputs = tf.nn.embedding_lookup(embedding_table, word_ids)
            inputs *= tf.cast(tf.expand_dims(mask, -1), dtype=decoding_args.decoder_args.dtype)
            # [batch_size * beam_width, 1, hidden_dim]
            inputs = tf.expand_dims(inputs, 1)
            inputs *= decoding_args.decoder_args.hidden_dim**0.5

            position_encoder = SinusoidalPositionEncoder()
            if position_encoder is not None:
                inputs = position_encoder(
                    inputs, position=step if step is not None else None)

            with tf.variable_scope("decoder_stack", reuse=tf.AUTO_REUSE):
                tf_result, _ = tf_decoder(decoder_args=decoding_args.decoder_args,
                                          inputs=inputs,
                                          memory=extended_memory,
                                          memory_sequence_length=extended_memory_sequence_length,
                                          step=step,
                                          batch_size=batch_size,
                                          cache=my_cache)
                result = tf_result
                result = layer_norm_v2(result, 'layer_normalization', decoding_args.decoder_args.dtype)
            # [batch_size * beam_width, hidden_dim]
            result = tf.squeeze(result, axis=1)
            logits = tf.matmul(result, embedding_table, transpose_b=True)

            end_ids = tf.fill([batch_size * decoding_args.decoder_args.beam_width],
                              decoding_args.end_id)  # [batch_size * beam_width]
            eos_max_prob = tf.one_hot(end_ids, decoding_args.vocab_size,
                                      on_value=decoding_args.decoder_args.dtype.max,
                                      off_value=decoding_args.decoder_args.dtype.min)  # [batch_size * beam_width, vocab_size]
            # [batch_size * beam_width, vocab_size]
            logits = tf.where(finished, x=eos_max_prob, y=logits)
            logits = tf.cast(logits, tf.float32)
            # [batch_size * beam_width, vocab_size]
            log_probs = tf.nn.log_softmax(logits)

            debug_output = log_probs

            output_id, next_cum_log_probs, finished, my_cache, \
                extra_vars, op_self_cache, cum_log_probs_old, _ = beam_search(decoding_args.decoder_args.beam_width,
                                                        decoding_args.vocab_size,
                                                        step,
                                                        log_probs,
                                                        cum_log_probs,
                                                        finished,
                                                        my_cache,
                                                        extra_vars,
                                                        op_self_cache)

            outputs = outputs.write(step, output_id)
            # 1、好像是费操，当finished时，log_probs在end_id处为0，其他位置为-inf，next_cum_log_probs和cum_log_probs值一样，
            # 2、finished经过了tf.gather(finished, beam_indices)重新排序过了，但是cum_log_probs还是排序之前的，存在不对应问题，这是个bug
            cum_log_probs = tf.where(finished, x=cum_log_probs_old, y=next_cum_log_probs)
            finished = tf.logical_or(finished, tf.equal(output_id, decoding_args.end_id))

            return output_id, cum_log_probs, finished, step + 1, outputs, debug_output, my_cache, extra_vars, op_self_cache, op_mem_cache

        # initialization
        start_ids, step, outputs, tf_decoder_cache, finished, initial_log_probs, \
            tf_sequence_lengths, extra_vars = initialize_decoding_variables(decoding_args, batch_size)

        word_ids = tf.identity(start_ids, name="word_ids")
        cum_log_probs = tf.identity(initial_log_probs, name="cum_log_probs")

        # if use_op == False, these two caches are useless
        op_self_cache, op_mem_cache = init_op_cache(decoding_args.decoder_args, input_length)

        debug_output = tf.zeros([batch_size*decoding_args.decoder_args.beam_width,
                                 decoding_args.vocab_size], dtype=tf.float32)

        _, _, _, _, outputs, debug_output, _, extra_vars, _, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(
                word_ids,
                cum_log_probs,
                finished,
                step,
                outputs,
                debug_output,
                tf_decoder_cache,
                extra_vars,
                op_self_cache,
                op_mem_cache
            ),
            back_prop=False,
            maximum_iterations=max_decode_length,
            shape_invariants=(
                start_ids.shape,
                initial_log_probs.shape,
                finished.shape,
                step.shape,
                tf.TensorShape(None),
                debug_output.shape,
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, tf_decoder_cache),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, extra_vars),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, op_self_cache),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, op_mem_cache))
        )

        tf_parent_ids = extra_vars[0].stack()
        tf_sequence_lengths = extra_vars[1]
        tf_output_ids = outputs.stack()

        finalized_tf_output_ids, finalized_tf_sequence_lengths = finalize(decoding_args.decoder_args.beam_width,
                                                                          tf_parent_ids,
                                                                          tf_sequence_lengths,
                                                                          tf_output_ids,
                                                                          decoding_args.end_id)

        finalized_tf_output_ids = tf.cast(
            finalized_tf_output_ids, start_ids.dtype)
        finalized_tf_sequence_lengths = tf.minimum(
            finalized_tf_sequence_lengths + 1, tf.shape(finalized_tf_output_ids)[2])

        return finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, tf_parent_ids, tf_sequence_lengths, debug_output


## 解码
PAD = "[PAD]"
PAD_ID = 0
EOS = "[EOS]"
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]
def predicted_ids_with_eos_to_string_v2(predicted_ids_with_eos, dic):
    """convert predicted_ids to string.
    Args:
        predicted_ids_with_eos: 3-D int64 Tensor of shape [b, t]. the EOS_ID and PAD_ID are padded at the end of each prediction.
        dic: dict. Dict for id->char mapping.

    Returns: 1-D string SparseTensor with dense shape: [batch_size,]

    """
    dic_tensor = tf.constant(dic, dtype=tf.string)
    predicted_char_list = tf.gather(dic_tensor, predicted_ids_with_eos, axis=0)    #[b,t]
    predicted_string = merge_chars_to_sentence(predicted_char_list, ignore_chars=RESERVED_TOKENS)
    return predicted_string, predicted_char_list


def merge_chars_to_sentence(chars, ignore_chars=None):
    """convert a list to sentence. For example:
    [ 'a', 'b'] -> [ 'ab']

    Args:
        chars: Tenosr or SparseTensor of rank 2.

    Returns: Tensor of rank 1, dtype `string`.

    """
    #char_list: SparseTensor: [ ['a', 'b'], ['a','b','c'] ]
    if isinstance(chars, tf.SparseTensor):
        dense_list = tf.sparse_tensor_to_dense(
            chars, '')    # Tensor: [[ 'a' , 'b', ''], [ 'a', 'b', 'c'] ]
    else:
        dense_list = chars

    return string_join(dense_list, ignore_chars)


def string_join(string_tensor, ignore_chars=None):
    """TODO: Docstring for string_join.

    Args:
        string_tensor: 1-D or 2-D string Tensor. join the strings in the last dimension
        ignore_chars: List. ignore the chars when join.
        stop_at_char: String. stop join when encountering the char.

    Returns: 0-D or 1-D string Tensor. The joined string.

    """
    i = tf.constant(0)
    string_num = tf.shape(string_tensor)[-1]
    if tf.rank(string_tensor) == 1:
        joined_string = ''
    else:
        dims = tf.rank(string_tensor)
        joined_string = tf.fill(tf.shape(string_tensor)[:-1], "")
        # joined_string = tf.tile([''],
        # tf.expand_dims(tf.shape(string_tensor)[0], 0))

    c = lambda i, joined_string: tf.less(i, string_num)

    def body_with_ignorechars(i, joined_string):
        char_i = string_tensor[..., i:i + 1]
        tile_char = tf.concat([char_i for i in range(len(ignore_chars))], axis=-1)
        in_ignore = tf.reduce_any(tf.equal(tile_char, tf.constant(ignore_chars)), axis=-1)
        joined_string = tf.where(in_ignore, joined_string, joined_string + char_i[..., 0])
        return tf.add(i, 1), joined_string

    body_no_ignorechars = lambda i, joined_string: [
        tf.add(i, 1), joined_string + string_tensor[..., i]
    ]

    b = body_no_ignorechars if ignore_chars is None else body_with_ignorechars
    # b = body_no_ignorechars
    i, joined_string = tf.while_loop(c, b, [i, joined_string], back_prop=False)
    return joined_string


