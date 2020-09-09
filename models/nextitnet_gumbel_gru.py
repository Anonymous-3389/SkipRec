from __future__ import print_function

import tensorflow as tf

from modules import blocks
from modules.convs import conv1d
from modules.gumbel_softmax import gumbel_softmax_


class NextItNetGumbel:
    def __init__(self, args):
        # self.args = args
        self.dilations = args["dilations"]
        self.item_size = args["item_size"]
        self.kernel_size = args["kernel_size"]
        self.dilated_channels = args["dilated_channels"]
        self.negative_sampling_ratio = args["negative_sampling_ratio"]
        self.using_negative_sampling = args["using_negative_sampling"]

        self.item_embedding = tf.get_variable(
            "item_embedding",
            [args["item_size"], args["dilated_channels"]],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        self.input_train = tf.placeholder(
            "int32", [None, None], name="input_train"
        )  # [B, SessionLen]
        self.input_test = tf.placeholder(
            "int32", [None, None], name="input_test"
        )  # [B, SessionLen]

        self.softmax_w = tf.get_variable(
            "softmax_w",
            [self.item_size, self.dilated_channels],
            tf.float32,
            tf.random_normal_initializer(0.0, 0.01),
        )
        self.softmax_b = tf.get_variable(
            "softmax_b", [self.item_size], tf.float32, tf.constant_initializer(0.1)
        )

    def build_train_graph(self, policy_action):
        # tf.get_variable_scope().reuse_variables()
        label_seq, dilate_input = self._expand_model_graph(
            self.input_train, policy_action, train=True
        )

        if self.using_negative_sampling:
            logits_2d = tf.reshape(
                dilate_input, [-1, self.dilated_channels]
            )  # [B*(SessLen-1), DilatedChannels]
            label_flat = tf.reshape(label_seq, [-1, 1])  # [B*(SessLen-1), 1]
            num_sampled = int(self.negative_sampling_ratio * self.item_size)
            loss = tf.nn.sampled_softmax_loss(
                self.softmax_w,
                self.softmax_b,
                labels=label_flat,
                inputs=logits_2d,
                num_sampled=num_sampled,
                num_classes=self.item_size,
            )  # [B*(SessLen-1), 1]
        else:
            logits = conv1d(
                tf.nn.relu(dilate_input), output_channels=self.item_size, name="logits"
            )  # [B,SessLen-1,ItemSize]
            logits_2d = tf.reshape(
                logits, [-1, self.item_size]
            )  # [B*(SessLen-1),ItemSize]
            label_flat = tf.reshape(label_seq, [-1])  # [B*(SessLen-1), 1]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_flat, logits=logits_2d
            )  # [B*(SessLen-1), 1]

        self.loss = tf.reduce_mean(loss)  # output of training steps, [1]

    def build_test_graph(self, policy_action):
        tf.get_variable_scope().reuse_variables()

        label_seq, dilate_input = self._expand_model_graph(
            self.input_test, policy_action, train=False
        )
        self.loss_test, self.probs = self._get_test_result(dilate_input, label_seq)

    def _get_test_result(self, dilate_input, label_seq):
        if self.using_negative_sampling:
            # dilate_input[:, -1:, :] [B, 1, DilatedChannels]
            logits_2d = tf.reshape(
                dilate_input[:, -1:, :], [-1, self.dilated_channels]
            )  # [B, DilatedChannels]
            logits_2d = tf.matmul(
                logits_2d, tf.transpose(self.softmax_w)
            )  # [B, ItemSize]
            logits_2d = tf.nn.bias_add(logits_2d, self.softmax_b)  # [B, ItemSize]
        else:
            logits = conv1d(
                tf.nn.relu(dilate_input[:, -1:, :]), self.item_size, name="logits"
            )
            logits_2d = tf.reshape(logits, [-1, self.item_size])

        label_flat = tf.reshape(label_seq[:, -1], [-1])  # [B,]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_flat, logits=logits_2d
        )  # [B,]

        loss = tf.reduce_mean(loss)  # [1,]
        probs = tf.nn.softmax(logits_2d)  # [B, ItemSize]
        return loss, probs

    def _expand_model_graph(self, item_seq_input, policy_action, train):
        context_seq = item_seq_input[:, 0:-1]  # [B, SessLen - 1]
        label_seq = item_seq_input[:, 1:]  # [B, SessLen - 1]

        dilate_input = tf.nn.embedding_lookup(
            self.item_embedding, context_seq, name="context_embedding"
        )  # [B, SessLen - 1, DilatedChannels]

        for layer_id, dilation in enumerate(self.dilations):
            layer_input = dilate_input
            layer_output = blocks.rezero_block(
                dilate_input,
                dilation,
                layer_id,
                self.dilated_channels,
                self.kernel_size,
                causal=True,
                train=train,
            )
            action_mask = tf.reshape(policy_action[:, layer_id], [-1, 1, 1])
            dilate_input = layer_output * action_mask + layer_input * (1 - action_mask)
        return label_seq, dilate_input


class PolicyNetGumbelGru:
    def __init__(self, args):
        # self.args = args
        self.temp = args["temp"]
        self.dilations = args["dilations"]
        self.item_size = args["item_size"]
        self.kernel_size = args["kernel_size"]
        self.dilated_channels = args["dilated_channels"]
        self.dilations_block = args["block_shape"]
        self.negative_sampling_ratio = args["negative_sampling_ratio"]
        self.using_negative_sampling = args["using_negative_sampling"]
        self.action_num = len(args["dilations"])

        self.item_embedding = tf.get_variable(
            "item_embedding",
            [args["item_size"], args["dilated_channels"]],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )
        self.gru = tf.keras.layers.GRU(self.dilated_channels)

        self.softmax_w = tf.get_variable(
            "softmax_w",
            [self.action_num * 2, self.dilated_channels],  # change to action number
            tf.float32,
            tf.random_normal_initializer(0.0, 0.01),
        )
        self.output_bias = tf.get_variable(
            "softmax_b",
            shape=[self.action_num * 2],
            initializer=tf.random_normal_initializer(0.0, 0.01),
        )
        self.input = tf.placeholder(
            "int32", [None, None], name="item_seq_input"
        )  # [B, SessionLen]

    def build_policy(self):
        context_seq = self.input[:, 0:-1]  # [B, S]
        dilate_input = tf.nn.embedding_lookup(
            self.item_embedding, context_seq, name="context_embedding"
        )  # [B, S, C]

        dilate_input = self.gru(dilate_input)  # [B, C]

        logits = tf.matmul(dilate_input, self.softmax_w, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        logits = tf.reshape(logits, [-1, self.action_num, 2])
        logits = tf.sigmoid(logits)

        self.logits_check = tf.nn.softmax(logits)

        action = gumbel_softmax_(logits, hard=True, temperature=self.temp)
        self.action_predict = action[:, :, 0]
