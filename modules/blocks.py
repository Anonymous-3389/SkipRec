from .convs import conv1d
from .others import layer_norm
import tensorflow as tf


def rezero_block(
    input_, dilation, layer_id, residual_channels, kernel_size, causal=True, train=True
):
    block_name = "rez_layer_{}_{}".format(layer_id, dilation)
    with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
        rez = tf.get_variable("rez", [1], initializer=tf.constant_initializer(0.0))

        dilated_conv = conv1d(
            input_,
            residual_channels,
            dilation,
            kernel_size,
            causal=causal,
            name="dilated_conv1",
        )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(
            relu1,
            residual_channels,
            2 * dilation,
            kernel_size,
            causal=causal,
            name="dilated_conv2",
        )
        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu2 = tf.nn.relu(input_ln)

        return input_ + relu2 * rez


def plain_block(
    input_, dilation, layer_id, residual_channels, kernel_size, causal=True, train=True
):
    block_name = "layer_{}_{}".format(layer_id, dilation)
    with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(
            input_,
            residual_channels,
            dilation,
            kernel_size,
            causal=causal,
            name="dilated_conv1",
        )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(
            relu1,
            residual_channels,
            2 * dilation,
            kernel_size,
            causal=causal,
            name="dilated_conv2",
        )
        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu2 = tf.nn.relu(input_ln)

        return input_ + relu2
