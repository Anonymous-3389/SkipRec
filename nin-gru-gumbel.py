import argparse
import logging
import os
import time
from time import strftime, localtime

import numpy as np
import tensorflow as tf

from data import DataLoader
from models.nextitnet_gumbel_gru import (
    NextItNetGumbel,
    PolicyNetGumbelGru,
)
from utils import logger
from utils.metrics import get_metric, sample_top_ks
from utils.tools import (
    dict_to_str,
    get_proto_config,
    get_proto_config_with_occupy,
    setup_folder,
)


def get_model_args(item_size, global_args):
    config = {
        "using_negative_sampling": True,
        "negative_sampling_ratio": 0.2,
        "gen_sub_sess": False,
        "pad_token": "<UNK>",
        "item_size": item_size,
    }
    config.update(
        {
            "log_every": global_args["log_every"],
            "temp": global_args["temp"],
            "dilated_channels": global_args["channel"],
            "kernel_size": global_args["kernel_size"],
            "block_shape": global_args["block_shape"],
            "dilations": global_args["block_shape"] * global_args["n_blocks"],
            "batch_size": global_args["batch_size"],
            "iterations": global_args["iter"],
            "lr": global_args["lr"],
        }
    )
    return config


def preset(global_args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_args["gpu"])
    if global_args["seed"] is None:
        from random import randint

        seed = randint(1, 1000000)
    else:
        seed = global_args["seed"]
    logging.info("Using seed: {}".format(seed))
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_data_and_config(global_args):
    data_loader = DataLoader(global_args)
    train_set, test_set = data_loader.split()

    model_args = get_model_args(data_loader.item_nums, global_args)
    logging.info(dict_to_str(model_args, "Model Configurations"))

    if model_args["gen_sub_sess"]:
        train_set = data_loader.generate_sub_sessions(
            train_set, model_args["pad_token"]
        )

    return model_args, train_set, test_set


def summary_block(usage, block_size, title):
    logging.info("<Usage>::Block Usage of [{}]".format(title))
    block_usage = np.array(usage, dtype=np.float)

    ratio = np.sum(block_usage, axis=0) / len(block_usage)
    logging.info("\tUsage: " + " ".join(["{:.2f}".format(x) for x in ratio]))

    row_sum = np.sum(block_usage, axis=1)
    min_usage = np.min(row_sum)
    max_usage = np.max(row_sum)

    per_sample_usage = row_sum.squeeze()
    mean_usage = np.mean(per_sample_usage)
    std_usage = np.std(per_sample_usage)

    logging.info(
        "\t{} / [{} -> {}], Mean: {:.3f}, Std: {:.3f}".format(
            block_size, min_usage, max_usage, mean_usage, std_usage
        )
    )

    unique_policies = np.unique(block_usage, axis=0)
    lp = len(unique_policies)
    la = len(block_usage)
    logging.info(
        "\tUnique Policies: {}, Testcases: {}, Ratio: {:.3f}".format(
            lp, la, 1.0 * lp / la
        )
    )


def start(global_args):
    preset(global_args)

    model_args, train_set, test_set = get_data_and_config(global_args)

    ratio = global_args["occupy"]
    if ratio is None:
        gpu_config = get_proto_config()
        logging.info("Auto-growth GPU memory.")
    else:
        gpu_config = get_proto_config_with_occupy(ratio)
        logging.info("{:.1f}% GPU memory occupied.".format(ratio * 100))

    sess = tf.Session(config=gpu_config)

    with tf.variable_scope("policy_net"):
        policy_net = PolicyNetGumbelGru(model_args)
        policy_net.build_policy()

    with tf.variable_scope(tf.get_variable_scope()):
        model = NextItNetGumbel(model_args)
        model.build_train_graph(policy_action=policy_net.action_predict)
        model.build_test_graph(policy_action=policy_net.action_predict)

    variables = tf.contrib.framework.get_variables_to_restore()
    model_variables = [v for v in variables if not v.name.startswith("policy_net")]
    policy_variables = [v for v in variables if v.name.startswith("policy_net")]

    with tf.variable_scope(tf.get_variable_scope()):
        optimizer_finetune = tf.train.AdamOptimizer(
            learning_rate=model_args["lr"], name="Adam_finetune"
        )
        train_model = optimizer_finetune.minimize(model.loss, var_list=model_variables)
    with tf.variable_scope("policy_net"):
        optimizer_policy = tf.train.AdamOptimizer(
            learning_rate=model_args["lr"], name="Adam_policy"
        )
        train_policy = optimizer_policy.minimize(model.loss, var_list=policy_variables)

    init = tf.global_variables_initializer()
    sess.run(init)

    # restore if needed
    if global_args["use_pre"]:
        restore_op = tf.train.Saver(var_list=model_variables)
        restore_op.restore(sess, global_args["pre"])
        sess.run(tf.assign(policy_net.item_embedding, model.item_embedding))
        logging.info(">>>>> Parameters loaded from pre-trained model.")
    else:
        logging.info(">>>>> Training without pre-trained model.")

    logging.info("Start @ {}".format(strftime("%m.%d-%H:%M:%S", localtime())))

    saver = tf.train.Saver(max_to_keep=3)

    batch_size = model_args["batch_size"]
    log_meter = model_args["log_every"]
    total_iters = model_args["iterations"]
    total_steps = int(train_set.shape[0] / batch_size)
    test_steps = int(test_set.shape[0] / batch_size)

    model_save_path = global_args["store_path"]
    model_name = global_args["name"]

    logging.info("Batch size = {}, Batches = {}".format(batch_size, total_steps))

    best_mrr_at5 = 0.0

    for idx in range(total_iters):
        logging.info("-" * 30)
        logging.info("Iter: {} / {}".format(idx + 1, total_iters))
        num_iter = 1
        tic = time.time()

        train_usage_sample = []
        for batch_step in range(total_steps):
            train_batch = train_set[
                batch_step * batch_size : (batch_step + 1) * batch_size, :
            ]
            _, _, loss, action = sess.run(
                [train_model, train_policy, model.loss, policy_net.action_predict],
                feed_dict={
                    model.input_train: train_batch,
                    policy_net.input: train_batch,
                },
            )
            train_usage_sample.extend(np.array(action).tolist())

            if num_iter % log_meter == 0:
                logging.info(
                    "\t{:5d} /{:5d} Loss: {:.3f}".format(
                        batch_step + 1, total_steps, loss
                    )
                )
            num_iter += 1

        summary_block(train_usage_sample, len(model_args["dilations"]), "Train")

        # 1. eval model
        mrr_list = {5: [], 20: []}
        hr_list = {5: [], 20: []}
        ndcg_list = {5: [], 20: []}

        test_usage_sample = []
        for batch_step in range(test_steps):
            test_batch = test_set[
                batch_step * batch_size : (batch_step + 1) * batch_size, :
            ]

            action, pred_probs = sess.run(
                [policy_net.action_predict, model.probs],
                feed_dict={model.input_test: test_batch, policy_net.input: test_batch,},
            )

            test_usage_sample.extend(np.array(action).tolist())

            ground_truth = test_batch[:, -1]
            top_5_rank, top_20_rank = sample_top_ks(pred_probs, [5, 20])
            indices_5 = [
                np.argwhere(line == item)
                for line, item in zip(top_5_rank, ground_truth)
            ]
            indices_20 = [
                np.argwhere(line == item)
                for line, item in zip(top_20_rank, ground_truth)
            ]

            mrr5_sub, hr5_sub, ndcg5_sub = get_metric(indices_5)
            mrr20_sub, hr20_sub, ndcg20_sub = get_metric(indices_20)

            mrr_list[5].extend(mrr5_sub), mrr_list[20].extend(mrr20_sub)
            hr_list[5].extend(hr5_sub), hr_list[20].extend(hr20_sub)
            ndcg_list[5].extend(ndcg5_sub), ndcg_list[20].extend(ndcg20_sub)

        summary_block(test_usage_sample, len(model_args["dilations"]), "Test")

        ndcg_5, ndcg_20 = np.mean(ndcg_list[5]), np.mean(ndcg_list[20])
        mrr_5, mrr_20 = np.mean(mrr_list[5]), np.mean(mrr_list[20])
        hr_5, hr_20 = np.mean(hr_list[5]), np.mean(hr_list[20])

        logging.info("<Metric>::TestSet")
        logging.info(
            "\t MRR@5: {:.4f},  HIT@5: {:.4f},  NDCG@5: {:.4f}".format(
                mrr_5, hr_5, ndcg_5
            )
        )
        logging.info(
            "\tMRR@20: {:.4f}, HIT@20: {:.4f}, NDCG@20: {:.4f}".format(
                mrr_20, hr_20, ndcg_20
            )
        )

        mrr_at5 = mrr_5

        # 2. save model
        if mrr_at5 > best_mrr_at5:
            logging.info(
                ">>>>> Saving model due to better MRR@5: {:.4f} <<<<< ".format(mrr_at5)
            )
            saver.save(
                sess,
                os.path.join(
                    model_save_path, "{}_{}.tfkpt".format(model_name, num_iter)
                ),
            )
            best_mrr_at5 = mrr_at5

        toc = time.time()
        logging.info(
            "Iter: {} / {} finish. Time: {:.2f} min".format(
                idx + 1, total_iters, (toc - tic) / 60
            )
        )

    sess.close()


def parse_arg(args_inline=None, **patch):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name to create folder which saves log and model parameters.",
    )
    parser.add_argument(
        "--use_pre", action="store_true", help="Whether to use pretrained model."
    )
    parser.add_argument("--pre", type=str, help="/path/to/pre-trained_model")
    parser.add_argument(
        "--temp", type=int, required=True, help="Parameter `temp` in the paper."
    )

    parser.add_argument("--n_blocks", required=True, type=int, help="Number of blocks.")
    parser.add_argument(
        "--block_shape", type=str, default="1,4", help="Dilated conv structure."
    )
    parser.add_argument("--channel", type=int, default=256, help="Embedding channels.")
    parser.add_argument(
        "--kernel_size", type=int, default=3, help="Size of conv kernel."
    )

    parser.add_argument(
        "--gpu", type=int, required=True, help="The index of GPU which you want to use."
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="movielen_30.csv",
        help="Specific file that holds all data.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size.")
    parser.add_argument("--iter", type=int, default=50, help="Num of epoch to train.")

    parser.add_argument(
        "--store_root", type=str, default="store", help="Path of data sets' folder."
    )
    parser.add_argument(
        "--console_output", action="store_true", help="Print logger info to console"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/data",
        help="Folder that holds data set files.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for tensorflow and numpy."
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.1,
        help="Train/Test Split ratio, 0.2 means 80% training and 20% testing",
    )
    parser.add_argument(
        "--log_every", type=int, default=500, help="Print log info every (x) iters."
    )
    parser.add_argument(
        "--occupy", type=float, help="Occupy {x}-ratio of GPU memory at beginning.",
    )

    if args_inline is not None:
        data = vars(parser.parse_args(args_inline))
    else:
        data = vars(parser.parse_args())

    blocks = data["block_shape"]
    data["block_shape"] = [int(t) for t in blocks.split(",")]

    data.update(patch)
    return data


def log_configs(args):
    logging.info(dict_to_str(args, "Global Configurations"))


if __name__ == "__main__":
    pack = {
        # "console_output": True,
        # "store_root": "store/1.TEST",
        # "log_every": 10,
    }
    configs = parse_arg(**pack)

    path = setup_folder(configs)
    configs.update({"store_path": path})

    logger.logger_setup(configs)
    log_configs(configs)

    start(configs)
