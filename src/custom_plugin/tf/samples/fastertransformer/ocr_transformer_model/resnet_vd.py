import tensorflow as tf

initializer_range = 0.02

def create_initializer(initializer_range=0.02, data_type=tf.float32):
    return tf.truncated_normal_initializer(stddev=initializer_range, dtype=data_type)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, dtype, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(inputs=inputs,
                                         axis=1 if data_format == 'channels_first' else 3,
                                         momentum=_BATCH_NORM_DECAY,
                                         epsilon=_BATCH_NORM_EPSILON,
                                         center=True,
                                         scale=True,
                                         training=training,
                                         fused=True,
                                         beta_initializer=create_initializer(initializer_range, dtype),
                                         gamma_initializer=create_initializer(initializer_range, dtype),
                                         moving_mean_initializer=create_initializer(initializer_range, dtype),
                                         moving_variance_initializer=create_initializer(initializer_range, dtype))


class resnet_vd:
    def __init__(self, data_format):
        self.data_format = data_format
        # resnet vd 34 case
        # downsample rate of W is 4
        self.depth = [3, 4, 6, 3]
        self.num_filters = [64, 128, 256, 512]

    def __call__(self, inputs, training, dtype=tf.float32):
        # inputs b, h, w, c
        # outputs b, h/32, w/4, 512
        self.training = training
        self.dtype = dtype
        x = self.module_0(inputs)
        x = self.module_1(x)
        return x

    def module_0(self, inputs):
        x = self.conv_bn_layer(inputs, 32, 3, 1)
        x = self.conv_bn_layer(x, 32, 3, 1)
        x = self.conv_bn_layer(x, 64, 3, 1)
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=3,
                                    strides=2,
                                    padding='SAME',
                                    data_format=self.data_format)
        return x

    def module_1(self, x):
        for block_id, block_depth in enumerate(self.depth):
            for i in range(block_depth):
                if i == 0 and block_id != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)

                x = self.basic_block(x, num_filters=self.num_filters[block_id],
                                        stride=stride,
                                        if_first=(block_id == i == 0))
        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=2,
                                    strides=2,
                                    padding='SAME',
                                    data_format=self.data_format)
        return x

    def conv_bn_layer(self, inputs, num_filters, filter_size, stride, activation='relu'):
        x = tf.layers.conv2d(inputs=inputs,
                            filters=num_filters,
                            kernel_size=filter_size,
                            strides=stride,
                            padding='SAME',
                            use_bias=False,
                            kernel_initializer=create_initializer(initializer_range, self.dtype),
                            data_format=self.data_format)
        x = batch_norm(x, self.training, self.dtype, self.data_format)
        if activation == 'relu':
            x = tf.nn.relu(x)
        return x

    def basic_block(self, x, num_filters, stride, if_first):
        conv0 = self.conv_bn_layer(x, num_filters, 3, stride)
        conv1 = self.conv_bn_layer(conv0, num_filters, 3, 1, activation='none')
        short = self.shortcut(x, num_filters, stride, if_first=if_first)
        return tf.nn.relu(short + conv1)

    def shortcut(self, input, ch_out, stride, if_first=False):
        if self.data_format == 'channels_first':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[3]
        if ch_in != ch_out or stride[0] != 1:
            if if_first:
                return self.conv_bn_layer(input, ch_out, 1, stride)
            else:
                return self.conv_bn_layer_new(input, ch_out, 1, stride)
        elif if_first:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def conv_bn_layer_new(self, input, num_filters, filter_size, stride):
        # key modification of Resnet-D
        pool = tf.layers.average_pooling2d(
                        input,
                        stride,
                        stride,
                        padding='same',
                        data_format=self.data_format
                    )
        conv = tf.layers.conv2d(inputs=pool,
                            filters=num_filters,
                            kernel_size=filter_size,
                            strides=1,
                            padding='SAME',
                            use_bias=False,
                            kernel_initializer=create_initializer(initializer_range, self.dtype),
                            data_format=self.data_format)
        return batch_norm(conv, self.training, self.dtype, self.data_format)
