import tensorflow as tf


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_(logits, temperature=10, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def gumbel_softmax_v2(logits, temperature=10, hard=False, given_action=None):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y_out = y

    if given_action is None:
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keep_dims=True)), y.dtype)
        else:
            action_num = tf.shape(y)[1]
            sample_action = tf.multinomial(
                tf.reshape(tf.log(y), [-1, 2]), num_samples=1
            )
            sample_action = tf.reshape(sample_action, [-1, action_num, 1])
            y_hard = tf.cast(
                tf.concat([1 - sample_action, sample_action], axis=-1), y.dtype
            )
    else:
        given_action = tf.expand_dims(given_action, axis=-1)
        y_hard = tf.cast(tf.concat([given_action, 1 - given_action], axis=-1), y.dtype)

    y = tf.stop_gradient(y_hard - y) + y
    return y, y_out
