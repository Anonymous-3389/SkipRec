import tensorflow as tf


def conv1d(
    input_,
    output_channels,
    dilation=1,
    kernel_size=1,
    causal=False,
    name="dilated_conv",
):
    with tf.variable_scope(name):
        weight = tf.get_variable(
            "weight",
            [1, kernel_size, input_.get_shape()[-1], output_channels],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        bias = tf.get_variable(
            "bias", [output_channels], initializer=tf.constant_initializer(0.0)
        )

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, axis=1)
            out = (
                tf.nn.atrous_conv2d(
                    input_expanded, weight, rate=dilation, padding="VALID"
                )
                + bias
            )
        else:
            input_expanded = tf.expand_dims(input_, axis=1)
            out = (
                tf.nn.atrous_conv2d(
                    input_expanded, weight, rate=dilation, padding="SAME"
                )
                + bias
            )

        return tf.squeeze(out, [1])
