import tensorflow as tf


def feed_forward_rezero(x, units, dropout, is_training):
    name = "feed_forward_rezero"
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        rez = tf.get_variable("rez", [1], initializer=tf.constant_initializer(0.0))

        out = tf.layers.conv1d(
            x, filters=units, kernel_size=1, activation=tf.nn.relu, use_bias=True
        )
        out = tf.layers.dropout(out, rate=dropout, training=is_training)
        out = tf.layers.conv1d(
            out, filters=units, kernel_size=1, activation=None, use_bias=True
        )
        out = tf.layers.dropout(out, rate=dropout, training=is_training)
        outputs = x + rez * out
    return outputs


def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        # (B, SessLen - 1, DilatedChannels)
        shape = x.get_shape()
        beta = tf.get_variable(
            "beta",
            [int(shape[-1])],
            initializer=tf.constant_initializer(0),
            trainable=trainable,
        )
        gamma = tf.get_variable(
            "gamma",
            [int(shape[-1])],
            initializer=tf.constant_initializer(1),
            trainable=trainable,
        )

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta
