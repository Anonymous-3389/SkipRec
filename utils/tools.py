import os
from time import strftime, localtime

import tensorflow as tf


def dict_to_str(data, title=None):
    _title = title if title is not None else "dict"

    s = "\n"
    s += "-----[{}]-----".format(_title)

    len_k = max([len(str(item)) for item in data.keys()])

    for k, v in data.items():
        line = ("{:>%d}" % len_k).format(k)
        s += "\n{} | {}".format(line, v)

    s += "\n"
    s += "-----[End of {}]-----".format(_title)

    return s


def setup_folder(args):
    root = args["store_root"]
    if not os.path.isdir(root):
        print "Root folder of store is created: {}".format(root)
        os.mkdir(root)

    folder_name = args["name"] + strftime("-%m.%d-%H.%M.%S", localtime())
    full_path = os.path.join(root, folder_name)
    if os.path.isdir(full_path):
        raise ValueError("Folder with name `{}` already exists.".format(full_path))
    os.mkdir(full_path)
    return full_path


def get_proto_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def get_proto_config_with_occupy(ratio):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = ratio
    return config
