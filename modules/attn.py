import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
        q: [..., seq_len_q, depth]
        k: [..., seq_len_k, depth]
        v: [..., seq_len_v, depth_v]
    """
    with tf.variable_scope("scaled_attn_layer"):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def split_heads(x, batch_size, num_heads, depth):
    """
    => (num_heads, depth).
    shape => (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


def multi_attn(q, k, v, mask, hidden_size, num_heads):
    with tf.variable_scope("multi_{}_attn".format(num_heads)):
        assert hidden_size % num_heads == 0
        depth = hidden_size // num_heads

        batch_size = tf.shape(q)[0]

        q = tf.layers.dense(q, hidden_size)
        k = tf.layers.dense(k, hidden_size)
        v = tf.layers.dense(v, hidden_size)

        q = split_heads(q, batch_size, num_heads, depth)
        k = split_heads(k, batch_size, num_heads, depth)
        v = split_heads(v, batch_size, num_heads, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, hidden_size)
        )  # (batch_size, seq_len_q, d_model)

        output = tf.layers.dense(concat_attention, hidden_size)

    return output, attention_weights


def multi_head_attention_rezero(
    queries,
    keys,
    num_units,
    num_heads=8,
    dropout_rate=0,
    is_training=True,
    causality=False,
):
    name = "multi_head_attention_rezero"
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        rez = tf.get_variable("rez", [1], initializer=tf.constant_initializer(0.0))

        q = tf.layers.dense(queries, num_units)  # (N, T_q, C)
        k = tf.layers.dense(keys, num_units)  # (N, T_k, C)
        v = tf.layers.dense(keys, num_units)  # (N, T_k, C)

        # Split and concat
        q = tf.concat(tf.split(q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        outputs = outputs / tf.math.sqrt(dk)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals
            ).to_dense()  # (T_q, T_k)
            masks = tf.tile(
                tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
            )  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        # Weighted sum
        outputs = tf.matmul(outputs, v)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection with rezero
        outputs = queries + rez * outputs

    return outputs
