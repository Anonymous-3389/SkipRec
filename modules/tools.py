from __future__ import print_function

import tensorflow as tf


def print_shape(x, name="Shape"):
    print_op = tf.print(name, ": ", tf.shape(x)) 
    with tf.control_dependencies([print_op]):
        return tf.identity(x)


def print_tensor(x):
    print_op = tf.print("Tensor: ", x)
    with tf.control_dependencies([print_op]):
        return tf.identity(x)
