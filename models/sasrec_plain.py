import tensorflow as tf

from modules.others import layer_norm, feed_forward_rezero
from modules.attn import multi_head_attention_rezero


def sas_block(x, hidden_size, num_heads, dropout, layer_id, is_training):
    name = "sas_layer_{}".format(layer_id)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = layer_norm(x, "ln_input", trainable=is_training)
        out = multi_head_attention_rezero(
            out,
            out,
            num_units=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout,
            causality=True,
            is_training=is_training,
        )
        out = layer_norm(out, "ln_intermediate", trainable=is_training)
        out = feed_forward_rezero(
            out, hidden_size, dropout=dropout, is_training=is_training
        )
        output = layer_norm(out, "ln_output", trainable=is_training)
    return output


class SASRec:
    def __init__(self, args):
        # self.args = args
        self.neg_sample_ratio = 0.2
        self.blocks = args["num_blocks"]
        self.dropout = args["dropout"]
        self.hidden_size = args["hidden"]
        self.item_size = args["item_size"]
        self.num_heads = args["num_head"]
        self.max_len = args["max_len"]

        self.item_embedding = tf.get_variable(
            "item_embedding",
            [self.item_size, self.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.01),
        )

        self.pos_embedding = tf.get_variable(
            "pos_embedding",
            [self.max_len, self.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.01),
        )
        self.softmax_w = tf.get_variable(
            "softmax_w",
            [self.item_size, self.hidden_size],
            tf.float32,
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )
        self.softmax_b = tf.get_variable(
            "softmax_b",
            [self.item_size],
            tf.float32,
            initializer=tf.constant_initializer(0.1),
        )

        # TRAIN
        self.input_train = tf.placeholder(
            "int32", [None, None], name="input_train"
        )  # [B, SessionLen]
        self.loss_train = self.build_train_graph()

        # TEST
        self.input_test = tf.placeholder(
            "int32", [None, None], name="input_test"
        )  # [B, SessionLen]
        self.loss_test, self.probs_test = self.build_test_graph()

    def build_train_graph(self):
        # label_seq: [B, Sess]
        # dilate_input: [B, Sess, C]
        label_seq, hidden = self._hidden_forward(self.input_train, train=True)

        # use negative sampling
        logits_2d = tf.reshape(hidden, [-1, self.hidden_size])  # [B*Sess, C]
        label_flat = tf.reshape(label_seq, [-1, 1])  # [B*Sess, 1]
        num_sampled = int(self.item_size * self.neg_sample_ratio)
        loss = tf.nn.sampled_softmax_loss(
            self.softmax_w,
            self.softmax_b,
            labels=label_flat,
            inputs=logits_2d,
            num_sampled=num_sampled,
            num_classes=self.item_size,
        )  # [B*(SessLen-1), 1]
        loss_train = tf.reduce_mean(loss)  # output of training steps, [1]
        return loss_train

    def build_test_graph(self):
        tf.get_variable_scope().reuse_variables()
        label_seq, hidden = self._hidden_forward(self.input_test, train=False)

        logits_2d = tf.reshape(hidden[:, -1:, :], [-1, self.hidden_size])
        logits_2d = tf.matmul(logits_2d, tf.transpose(self.softmax_w))
        logits_2d = tf.nn.bias_add(logits_2d, self.softmax_b)

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_flat, logits=logits_2d
        )

        loss = tf.reduce_mean(loss)  # [1,]
        probs = tf.nn.softmax(logits_2d)  # [B, ItemSize]
        return loss, probs

    def _hidden_forward(self, item_seq_input, train=True):
        context_seq = item_seq_input[:, 0:-1]  # [B, Sess]
        label_seq = item_seq_input[:, 1:]  # [B, Sess]

        context_pos = tf.tile(
            tf.expand_dims(tf.range(tf.shape(context_seq)[1]), 0),
            [tf.shape(context_seq)[0], 1],
        )

        item_emb = tf.nn.embedding_lookup(
            self.item_embedding, context_seq, name="item_context_emb"
        )
        pos_emb = tf.nn.embedding_lookup(
            self.pos_embedding, context_pos, name="pos_context_emb",
        )

        hidden = item_emb + pos_emb

        for i in range(self.blocks):
            hidden = sas_block(
                hidden,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                layer_id=i,
                is_training=train,
            )

        return label_seq, hidden
