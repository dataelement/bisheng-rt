import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import cv2
import numpy as np
import math
from . import beam_search
from . import resnet_vd
np.set_printoptions(threshold=1e6, suppress=True)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_NEG_INF = -1e9
EOS_ID = 1

def batch_norm(inputs, training, data_format):
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


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


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

    return inputs + shortcut


def bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=1,
                                  strides=1,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=3,
                                  strides=strides,
                                  data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=4 * filters,
                                  kernel_size=1,
                                  strides=1,
                                  data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, blocks, strides, training, name, data_format):
    filters_out = filters * 4 if bottleneck else filters
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1,
            strides=strides, data_format=data_format)

    if bottleneck:
        inputs = bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides, data_format)
    else:
        inputs = building_block_v2(inputs, filters, training, projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        if bottleneck:
            inputs = bottleneck_block_v2(inputs, filters, training, None, 1, data_format)
        else:
            inputs = building_block_v2(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def build_resnet31(inputs):
    num_filters = 16
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=num_filters, kernel_size=3,
        strides=1, data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_conv')
    block_sizes = [5, 5, 5]
    block_strides = [1, 2, 2]
    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=False,
                             blocks=num_blocks, strides=block_strides[i], training=False,
                             name='block_layer{}'.format(i + 1), data_format='channels_first')
        num_filters *= 2

    inputs = batch_norm(inputs, False, 'channels_first')
    inputs = tf.nn.relu(inputs)
    return inputs


def build_resnet50(inputs):
    kernel_size = 7
    conv_stride = 2
    num_filters = 64
    inputs = conv2d_fixed_padding(inputs=inputs, filters=num_filters, kernel_size=kernel_size,
                                  strides=conv_stride, data_format='channels_first')
    inputs = tf.identity(inputs, 'initial_conv')
    block_sizes = [3, 4, 6, 3]
    block_strides = [1, 2, 2, 1]
    for i, num_blocks in enumerate(block_sizes):
        inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=True,
                             blocks=num_blocks, strides=block_strides[i], training=False,
                             name='block_layer{}'.format(i + 1), data_format='channels_first')
        num_filters *= 2

    inputs = batch_norm(inputs, False, 'channels_first')
    inputs = tf.nn.relu(inputs)
    return inputs


def build_resnet_vd(inputs):
    backbone = resnet_vd.resnet_vd("channels_first")
    inputs = backbone(inputs, False)
    return inputs

## encode
def get_padding(inputs, inputs_unpadded_length):
    with tf.name_scope("padding"):
        input_shape = tf.shape(inputs)
        indexs = tf.tile(tf.range(input_shape[1]), tf.expand_dims(input_shape[0], axis=0))
        indexs = tf.reshape(indexs, input_shape[:2])  # shape: [batch_size, input_length]
        inputs_unpadded_length = tf.tile(inputs_unpadded_length, tf.stack([1, input_shape[1]]))
        conditions = indexs < inputs_unpadded_length
        return 1 - tf.to_float(conditions)


def get_padding_bias(inputs, inputs_unpadded_length):
    with tf.name_scope("attention_bias"):
        padding = get_padding(inputs, inputs_unpadded_length)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def output_normalization(x, scale, bias, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def feed_forward_network(x, padding=None):
    allow_pad = True
    filter_size = 1024
    hidden_size = 512
    padding = None if not allow_pad else padding
    batch_size = x.get_shape()[0]
    length = tf.shape(x)[1]
    output = tf.layers.dense(x, filter_size, use_bias=True, activation=tf.nn.relu, name='filter_layer')
    output = tf.layers.dense(output, hidden_size, use_bias=True, name='output_layer')

    if padding is not None:
        padding = 1-padding # nopaddings are ones and paddings are zeros.
        padding = tf.expand_dims(padding, axis=-1)  # [batch_size, length, 1]
        padding = tf.tile(padding, [1,1,hidden_size] ) # [batch_size, length, hidden_size]
        output = tf.multiply(output, padding)
    return output


def split_heads(x, hidden_size, num_heads):
    with tf.name_scope("split_heads"):
        batch_size = tf.shape(x)[0]
        depth = (hidden_size // num_heads)
        x = tf.reshape(x, [batch_size, -1, num_heads, depth])
        return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x, hidden_size):
    with tf.name_scope("combine_heads"):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, -1, hidden_size])


def attention_layer(x, y, bias, cache=None):
    hidden_size = 512
    num_heads = 8
    q = tf.layers.dense(x, hidden_size, use_bias=False, name="q")
    k = tf.layers.dense(y, hidden_size, use_bias=False, name="k")
    v = tf.layers.dense(y, hidden_size, use_bias=False, name="v")

    if cache is not None:
        # Combine cached keys and values with new keys and values.
        k = tf.concat([cache["k"], k], axis=1)
        v = tf.concat([cache["v"], v], axis=1)

        # Update cache
        cache["k"] = k
        cache["v"] = v

    q = split_heads(q, hidden_size, num_heads)
    k = split_heads(k, hidden_size, num_heads)
    v = split_heads(v, hidden_size, num_heads)

    #[b, num_heads, w, depth]
    depth = (hidden_size // num_heads)
    q *= depth**-0.5

    logits = tf.matmul(q, k, transpose_b=True)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    attention_output = tf.matmul(weights, v)
    attention_output = combine_heads(attention_output, hidden_size)

    attention_output = tf.layers.dense(attention_output, hidden_size, use_bias=False, name="output_transform")
    return attention_output


def encoder_stack(encoder_inputs, attention_bias, inputs_padding):
    hidden_size = 512
    filter_size = 1024
    num_heads = 8
    attention_causal = False
    relu_dropout = 0.1
    allow_ffn_pad = True
    postprocess_dropout = 0.1
    num_hidden_layers = 3

    scale1 = tf.get_variable("encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_scale", [hidden_size],
                                     initializer=tf.ones_initializer())
    bias1 = tf.get_variable("encoder_stack/layer_0/self_attention/layer_normalization/layer_norm_bias", [hidden_size],
                        initializer=tf.zeros_initializer())
    for i in range(num_hidden_layers):
        with tf.variable_scope("encoder_stack/layer_%d" % i):
            with tf.variable_scope("self_attention"):
                inputs = encoder_inputs
                encoder_inputs = tf.layers.conv1d(encoder_inputs,
                                                  hidden_size,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding='same',
                                                  dilation_rate=1,
                                                  activation=tf.nn.relu)

                encoder_inputs = output_normalization(encoder_inputs, scale1, bias1)

                encoder_inputs = tf.layers.conv1d(encoder_inputs,
                                                  hidden_size,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding='same',
                                                  dilation_rate=1,
                                                  activation=tf.nn.relu)
                encoder_inputs = output_normalization(encoder_inputs, scale1, bias1)
                encoder_inputs += inputs

                if i == 0:
                    scale = tf.get_variable("layer_normalization_1/layer_norm_scale", [hidden_size],
                                         initializer=tf.ones_initializer())
                    bias = tf.get_variable("layer_normalization_1/layer_norm_bias", [hidden_size],
                            initializer=tf.zeros_initializer())
                else:
                    scale = tf.get_variable("layer_normalization/layer_norm_scale", [hidden_size],
                                         initializer=tf.ones_initializer())
                    bias = tf.get_variable("layer_normalization/layer_norm_bias", [hidden_size],
                            initializer=tf.zeros_initializer())
                shortcut = encoder_inputs

                encoder_inputs = output_normalization(encoder_inputs, scale, bias)
                with tf.variable_scope("self_attention"):
                    encoder_inputs = attention_layer(encoder_inputs, encoder_inputs, attention_bias)
                encoder_inputs = shortcut + encoder_inputs

            with tf.variable_scope("ffn"):
                scale = tf.get_variable("layer_normalization/layer_norm_scale", [hidden_size],
                                     initializer=tf.ones_initializer())
                bias = tf.get_variable("layer_normalization/layer_norm_bias", [hidden_size],
                                    initializer=tf.zeros_initializer())
                shortcut = encoder_inputs
                encoder_inputs = output_normalization(encoder_inputs, scale, bias)
                with tf.variable_scope("feed_foward_network"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)
                encoder_inputs = shortcut + encoder_inputs

    return output_normalization(encoder_inputs, scale1, bias1)


def encode(inputs, attention_bias, inputs_unpadded_length):
    hidden_size = 512
    filter_size = 1024
    num_heads = 8
    relu_dropout = 0.1
    allow_ffn_pad = True
    postprocess_dropout = 0.1
    num_hidden_layers = 3
    with tf.name_scope("encode"):
        encoder_inputs = inputs
        with tf.name_scope("add_pos_encoding"):
            #length = encoder_inputs.get_shape()[1]
            length = tf.shape(encoder_inputs)[1]
            encoder_inputs += get_position_encoding(length, hidden_size)
        inputs_padding = get_padding(inputs, inputs_unpadded_length)
        return encoder_stack(encoder_inputs, attention_bias, inputs_padding)


def build_trans_encode(inputs, inputs_unpadded_length):
    initializer = tf.variance_scaling_initializer(1.0, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
        attention_bias = get_padding_bias(inputs, inputs_unpadded_length)
        encoder_outputs = encode(inputs, attention_bias, inputs_unpadded_length)
        return encoder_outputs


## decode
def embedding_shared_weights(x, vocab_size, hidden_size, shared_weights):
    with tf.name_scope("embedding"):
        # Create binary mask of size [batch_size, length]
        mask = tf.to_float(tf.not_equal(x, 0))

        #embeddings = tf.gather(self.shared_weights, x)
        # to ensure the grad is also a tensor so that it can be scattered during multi-gpu training
        embeddings = tf.gather(tf.matmul(shared_weights, tf.eye(hidden_size)), x)
        embeddings *= tf.expand_dims(mask, -1)

        # Scale embedding by the sqrt of the hidden size
        embeddings *= hidden_size**0.5

        return embeddings


def embedding_linear(x, vocab_size, hidden_size, shared_weights):
    with tf.name_scope("presoftmax_linear"):
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        x = tf.reshape(x, [-1, hidden_size])
        logits = tf.matmul(x, shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, vocab_size])


def decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias, cache=None):
    num_hidden_layers_decoder = 3
    num_heads = 8
    feature_len = 512
    filter_size = 1024
    relu_dropout = 0.1
    allow_ffn_pad = True

    for i in range(num_hidden_layers_decoder):
        layer_name = "layer_%d" % i
        layer_cache = cache[layer_name] if cache is not None else None

        with tf.variable_scope("decoder_stack/layer_%d" % i):
            with tf.variable_scope("self_attention"):
                scale = tf.get_variable("layer_normalization/layer_norm_scale", [feature_len],
                                        initializer=tf.ones_initializer())
                bias = tf.get_variable("layer_normalization/layer_norm_bias", [feature_len],
                                       initializer=tf.zeros_initializer())
                shortcut = decoder_inputs
                decoder_inputs = output_normalization(decoder_inputs, scale, bias)

                with tf.variable_scope("self_attention"):
                    decoder_inputs = attention_layer(decoder_inputs, decoder_inputs, decoder_self_attention_bias,
                                                     cache=layer_cache)
                decoder_inputs = shortcut + decoder_inputs

            with tf.variable_scope("encdec_attention"):
                scale = tf.get_variable("layer_normalization/layer_norm_scale", [feature_len],
                                        initializer=tf.ones_initializer())
                bias = tf.get_variable("layer_normalization/layer_norm_bias", [feature_len],
                                       initializer=tf.zeros_initializer())
                shortcut = decoder_inputs

                decoder_inputs = output_normalization(decoder_inputs, scale, bias)
                with tf.variable_scope("attention"):
                    decoder_inputs = attention_layer(decoder_inputs, encoder_outputs, attention_bias)
                decoder_inputs = shortcut + decoder_inputs

            with tf.variable_scope("ffn"):
                scale = tf.get_variable("layer_normalization/layer_norm_scale", [feature_len],
                                     initializer=tf.ones_initializer())
                bias = tf.get_variable("layer_normalization/layer_norm_bias", [feature_len],
                                    initializer=tf.zeros_initializer())
                shortcut = decoder_inputs
                decoder_inputs = output_normalization(decoder_inputs, scale, bias)
                with tf.variable_scope("feed_foward_network"):
                    decoder_inputs = feed_forward_network(decoder_inputs)
                decoder_inputs = shortcut + decoder_inputs

    scale = tf.get_variable("decoder_stack/layer_normalization/layer_norm_scale", [feature_len],
                                     initializer=tf.ones_initializer())
    bias = tf.get_variable("decoder_stack/layer_normalization/layer_norm_bias", [feature_len],
                                    initializer=tf.zeros_initializer())
    debug_output = decoder_inputs
    return output_normalization(decoder_inputs, scale, bias), debug_output


def get_decoder_self_attention_bias(length):
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def _get_symbols_to_logits_fn(max_decode_length, feature_len, vocab_size, hidden_size, shared_weights):
    timing_signal = get_position_encoding(max_decode_length + 1, feature_len)
    decoder_self_attention_bias = get_decoder_self_attention_bias(max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
        """Generate logits for next potential IDs.

        Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

        Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
        """
        # Set decoder input to the last generated IDs
        decoder_input = ids[:, -1:]

        # Preprocess decoder input by getting embeddings and adding timing signal.
        decoder_input = embedding_shared_weights(decoder_input, vocab_size, hidden_size, shared_weights)
        decoder_input += timing_signal[i:i + 1]

        self_attention_bias = decoder_self_attention_bias[:, :, i:i +1, :i + 1]
        decoder_outputs, _ = decoder_stack(decoder_input,
                                        cache.get("encoder_outputs"), self_attention_bias,
                                        cache.get("encoder_decoder_attention_bias"), cache)
        logits = embedding_linear(decoder_outputs, vocab_size, hidden_size, shared_weights)
        logits = tf.squeeze(logits, axis=[1])
        debug_output = logits
        return logits, cache, debug_output

    return symbols_to_logits_fn


def predict(encoder_outputs, encoder_decoder_attention_bias, target_vocab_size, mask=None):
    extra_decode_length = 10
    feature_len = 512
    hidden_size = 512
    num_hidden_layers_decoder = 3
    beam_size = 5
    alpha = 0.6

    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + extra_decode_length

    with tf.variable_scope("embedding_shared_weights/embedding_and_softmax", reuse=tf.AUTO_REUSE):
        # Create and initialize weights. The random normal initializer was chosen
        # randomly, and works well.
        shared_weights = tf.get_variable("weights", [target_vocab_size, hidden_size],
                                         initializer=tf.random_normal_initializer(0., hidden_size**-0.5))

    symbols_to_logits_fn = _get_symbols_to_logits_fn(max_decode_length, feature_len,
                                                     target_vocab_size, hidden_size, shared_weights)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, feature_len]),
            "v": tf.zeros([batch_size, 0, feature_len]),
        }
        for layer in range(num_hidden_layers_decoder)
    }

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
    cache["prob_matrix"] = tf.zeros([batch_size, 0, target_vocab_size])

    if mask is None:
        mask = tf.ones([batch_size, target_vocab_size])
    cache["mask"] = mask

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores, prob_matrix, debug_output = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        alpha=alpha,
        max_decode_length=max_decode_length,
        eos_id=EOS_ID)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {
        "outputs": top_decoded_ids,
        "scores": top_scores,
        "all_decoded_res": decoded_ids,
        "all_decoded_scores": scores,
        "prob_matrix": prob_matrix,
        "debug_output": debug_output
    }


def build_trans(inputs, inputs_unpadded_length, target_vocab_size, mask=None):
    initializer = tf.variance_scaling_initializer(1.0, mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
        attention_bias = get_padding_bias(inputs, inputs_unpadded_length)
        encoder_outputs = encode(inputs, attention_bias, inputs_unpadded_length)
        logits = None
        decoded = predict(encoder_outputs, attention_bias, target_vocab_size, mask)
        return logits, decoded, encoder_outputs

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

