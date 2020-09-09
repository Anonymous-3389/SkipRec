import argparse
import logging
import os
import time

import numpy as np
import tensorflow as tf

from data import DataLoader
from models.sasrec_plain import SASRec
from utils import logger
from utils.metrics import get_metric, sample_top_ks
from utils.tools import (
    dict_to_str,
    get_proto_config,
    get_proto_config_with_occupy,
    setup_folder,
)


def get_model_args(item_size, global_args):
    config = global_args.copy()
    config.update(
        {"item_size": item_size,}
    )
    return config


def preset(global_args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_args["gpu"])
    if global_args["seed"] is None:
        from random import randint

        seed = randint(1, 1000000)
        global_args.update({"seed": seed})
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

    return model_args, train_set, test_set


def start(global_args):
    preset(global_args)

    model_args, train_set, test_set = get_data_and_config(global_args)
    model = SASRec(model_args)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        optim = tf.train.AdamOptimizer(model_args["lr"])
        if global_args["grad_clip"]:
            # loss: nan issue
            gvs = optim.compute_gradients(model.loss_train)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            optimizer = optim.apply_gradients(capped_gvs)
        else:
            optimizer = optim.minimize(model.loss_train)

    logging.info(
        "Item Size: {}, Train Set Size: {}, Test Set Size: {}".format(
            model_args["item_size"], len(train_set), len(test_set)
        )
    )

    ratio = global_args["occupy"]
    if ratio is None:
        gpu_config = get_proto_config()
        logging.info("Auto-growth GPU memory.")
    else:
        gpu_config = get_proto_config_with_occupy(ratio)
        logging.info("{:.1f}% GPU memory occupied.".format(ratio * 100))

    with tf.Session(config=gpu_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=3)

        batch_size = global_args["batch_size"]
        log_meter = global_args["log_every"]
        total_iters = global_args["iter"]
        model_save_path = global_args["store_path"]
        model_name = global_args["name"]

        total_steps = int(train_set.shape[0] / batch_size)

        logging.info("Batch size = {}, Batches = {}".format(batch_size, total_steps))

        best_mrr_at5 = 0.0

        for idx in range(total_iters):
            logging.info("-" * 30)
            logging.info("Iter: {} / {}".format(idx + 1, total_iters))
            num_iter = 1
            tic = time.time()
            for batch_step in range(total_steps):
                batch = train_set[
                    batch_step * batch_size : (batch_step + 1) * batch_size, :
                ]
                _, loss = sess.run(
                    [optimizer, model.loss_train], feed_dict={model.input_train: batch}
                )

                if num_iter % log_meter == 0:
                    logging.info(
                        "\t{:5d}/{:5d} Loss: {:.4f}".format(
                            batch_step + 1, total_steps, loss
                        )
                    )
                num_iter += 1

            # 1. eval model
            mrr_at5 = test_model(model, sess, test_set, model_args)
            # 2. save model
            if mrr_at5 > best_mrr_at5:
                logging.info(
                    ">>>>> Saving model due to better MRR@5: {:.4f} <<<<< ".format(
                        mrr_at5
                    )
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


def test_model(model, sess, test_set, model_args):
    batch_size = model_args["batch_size"]
    total_steps = int(test_set.shape[0] / batch_size)

    mrr_list = {5: [], 20: []}
    hr_list = {5: [], 20: []}
    ndcg_list = {5: [], 20: []}

    time_buffer = []
    for batch_step in range(total_steps):
        test_batch = test_set[
            batch_step * batch_size : (batch_step + 1) * batch_size, :
        ]

        tic = time.time()
        [pred_probs] = sess.run(
            [model.probs_test], feed_dict={model.input_test: test_batch}
        )
        toc = time.time()

        time_buffer.append(toc - tic)

        ground_truth = test_batch[:, -1]

        top_5_rank, top_20_rank = sample_top_ks(pred_probs, [5, 20])
        indices_5 = [
            np.argwhere(line == item) for line, item in zip(top_5_rank, ground_truth)
        ]
        indices_20 = [
            np.argwhere(line == item) for line, item in zip(top_20_rank, ground_truth)
        ]

        mrr5_sub, hr5_sub, ndcg5_sub = get_metric(indices_5)
        mrr20_sub, hr20_sub, ndcg20_sub = get_metric(indices_20)

        mrr_list[5].extend(mrr5_sub), mrr_list[20].extend(mrr20_sub)
        hr_list[5].extend(hr5_sub), hr_list[20].extend(hr20_sub)
        ndcg_list[5].extend(ndcg5_sub), ndcg_list[20].extend(ndcg20_sub)

    logging.info(
        "[Test] Time: {:.3f}s +- {:.3f}s per batch".format(
            np.mean(time_buffer), np.std(time_buffer)
        )
    )

    ndcg_5, ndcg_20 = np.mean(ndcg_list[5]), np.mean(ndcg_list[20])
    mrr_5, mrr_20 = np.mean(mrr_list[5]), np.mean(mrr_list[20])
    hr_5, hr_20 = np.mean(hr_list[5]), np.mean(hr_list[20])

    logging.info(
        "\t MRR@5: {:.4f},  HIT@5: {:.4f},  NDCG@5: {:.4f}".format(mrr_5, hr_5, ndcg_5)
    )
    logging.info(
        "\tMRR@20: {:.4f}, HIT@20: {:.4f}, NDCG@20: {:.4f}".format(
            mrr_20, hr_20, ndcg_20
        )
    )

    return mrr_5


def parse_arg(args_inline=None, **patch):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name to create folder which saves log and model parameters.",
    )
    # ---------[Model Hyper-parameters]---------
    parser.add_argument(
        "--num_blocks", required=True, type=int, help="Number of SAS blocks."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Specific file that holds all data.",
    )
    parser.add_argument(
        "--max_len", required=True, type=int, help="Max Session Length."
    )
    parser.add_argument("--hidden", type=int, default=256, help="Embedding channels.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--num_head", type=int, default=1, help="Num of multi-head.")
    # ------------------------------------------

    parser.add_argument(
        "--gpu", type=int, required=True, help="The index of GPU which you want to use."
    )

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--grad_clip", action="store_true", help="Clip gradient when training."
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size.")
    parser.add_argument("--iter", type=int, default=20, help="Num of epoch to train.")

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
        "--occupy", type=float, help="Occupy {x}% of GPU memory at beginning.",
    )

    if args_inline is not None:
        data = vars(parser.parse_args(args_inline))
    else:
        data = vars(parser.parse_args())

    data.update(patch)
    return data


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

    start(configs)
