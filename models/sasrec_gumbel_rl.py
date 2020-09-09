import tensorflow as tf

from modules.gumbel_softmax import gumbel_softmax_, gumbel_softmax_v2
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

        self.input_train = tf.placeholder("int32", [None, None], name="input_train")
        self.input_test = tf.placeholder("int32", [None, None], name="input_test")

    def _get_test_result(self, hidden, label_seq):
        logits_2d = tf.reshape(hidden[:, -1:, :], [-1, self.hidden_size])
        logits_2d = tf.matmul(logits_2d, tf.transpose(self.softmax_w))
        logits_2d = tf.nn.bias_add(logits_2d, self.softmax_b)

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_flat, logits=logits_2d
        )

        loss = tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits_2d)
        return loss, probs

    def build_train_graph(self, policy_action):
        # label_seq: [B, Sess]
        # dilate_input: [B, Sess, C]
        label_seq, hidden = self._hidden_forward(
            self.input_train, policy_action, train=True
        )

        # use sampling
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

        self.loss_train = tf.reduce_mean(loss)  # output of training steps, [1]

    def build_test_graph(self, policy_action):
        tf.get_variable_scope().reuse_variables()

        label_seq, hidden = self._hidden_forward(
            self.input_test, policy_action, train=False
        )
        self.loss_test, self.probs_test = self._get_test_result(hidden, label_seq)

    def _hidden_forward(self, item_seq_input, policy_action, train):
        context_seq = item_seq_input[:, 0:-1]  # [B, Sess]
        label_seq = item_seq_input[:, 1:]  # [B, Sess]

        # [B, Sess, DilatedChannels]->Sess->[Sess]->[1,Sess]->[B,Sess]
        context_pos = tf.tile(
            tf.expand_dims(tf.range(tf.shape(context_seq)[1]), 0),
            [tf.shape(context_seq)[0], 1],
        )

        # [B, Sess, DilatedChannels]
        item_emb = tf.nn.embedding_lookup(
            self.item_embedding, context_seq, name="item_context_emb"
        )
        # [B, Sess, DilatedChannels]
        pos_emb = tf.nn.embedding_lookup(
            self.pos_embedding, context_pos, name="pos_context_emb",
        )

        hidden = item_emb + pos_emb

        for layer_id in range(self.blocks):
            layer_input = hidden
            layer_output = sas_block(
                hidden,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                layer_id=layer_id,
                is_training=train,
            )

            action_mask = tf.reshape(policy_action[:, layer_id], [-1, 1, 1])
            hidden = layer_output * action_mask + layer_input * (1 - action_mask)

        return label_seq, hidden


class SASPolicyGumbelRL:
    def __init__(self, args):
        # self.args = args
        self.action_num = args["num_blocks"]
        self.dropout = args["dropout"]
        self.hidden_size = args["hidden"]
        self.item_size = args["item_size"]
        self.max_len = args["max_len"]
        self.neg_sample_ratio = 0.2
        self.num_heads = args["num_head"]
        self.temp = args["temp"]

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
            [self.action_num * 2, self.hidden_size],  # change to action number
            tf.float32,
            tf.random_normal_initializer(0.0, 0.01),
        )
        self.output_bias = tf.get_variable(
            "softmax_b",
            shape=[self.action_num * 2],
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )
        # [B, SessionLen]
        self.input = tf.placeholder("int32", [None, None], name="input")
        self.method = tf.placeholder("int32", name="choose_method")
        self.sample_action = tf.placeholder(
            "float32", [None, None], name="sample_action"
        )
        self.reward = tf.placeholder("float32", [None], name="reward_input")

        self.action_no_reward, _ = self.build_policy(self.input, train=True)

        self.action_with_reward, self.rl_loss = self.build_policy(
            self.input, train=False
        )

    def build_policy(self, input, train):
        tf.get_variable_scope().reuse_variables()

        dilate_input = self._hidden_forward(input, train=train)

        dilate_input = tf.reduce_mean(dilate_input, axis=1)
        logits = tf.matmul(dilate_input, self.softmax_w, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        logits = tf.reshape(logits, [-1, self.action_num, 2])
        logits = tf.sigmoid(logits)

        hard_action, _ = gumbel_softmax_v2(logits, hard=True)
        soft_action, _ = gumbel_softmax_v2(logits, hard=False)
        given_action, _ = gumbel_softmax_v2(logits, given_action=self.sample_action)

        # Part.1 Output of block usage
        action_predict = tf.case(
            {
                tf.equal(self.method, 1): lambda: hard_action[:, :, 0],
                tf.equal(self.method, 0): lambda: soft_action[:, :, 0],
                tf.less(self.method, 0): lambda: given_action[:, :, 0],
            },
            name="condition_action_predict",
            exclusive=True,
        )

        # Part.2 Reinforcement learning loss
        sample_action = tf.expand_dims(self.sample_action, -1)
        ont_hot_sample_action = tf.concat([sample_action, 1 - sample_action], axis=-1)
        entropy = tf.losses.softmax_cross_entropy(
            tf.reshape(ont_hot_sample_action, [-1, 2]),
            tf.reshape(logits, [-1, 2]),
            reduction=tf.losses.Reduction.NONE,
        )
        entropy = tf.reshape(entropy, [-1, self.action_num])
        rl_loss = tf.reduce_mean(
            tf.reduce_sum(entropy * tf.expand_dims(self.reward, axis=-1), axis=-1)
        )

        return action_predict, rl_loss

    def _hidden_forward(self, item_seq_input, train):
        context_seq = item_seq_input[:, 0:-1]  # [B, Sess]
        context_pos = tf.tile(
            tf.expand_dims(tf.range(tf.shape(context_seq)[1]), 0),
            [tf.shape(context_seq)[0], 1],
        )

        # [B, Sess, DilatedChannels]
        item_emb = tf.nn.embedding_lookup(
            self.item_embedding, context_seq, name="item_context_emb"
        )
        # [B, Sess, DilatedChannels]
        pos_emb = tf.nn.embedding_lookup(
            self.pos_embedding, context_pos, name="pos_context_emb",
        )

        hidden = item_emb + pos_emb

        hidden = sas_block(
            hidden,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            layer_id=0,
            is_training=train,
        )

        return hidden
