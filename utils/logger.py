import logging
from os import path


def logger_setup(config):
    dir_root = config["store_path"]
    log_name = "logger.log"
    full_path = path.join(dir_root, log_name)
    print full_path

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    file_hdl = logging.FileHandler(full_path)
    file_hdl.setFormatter(formatter)

    root_logger.addHandler(file_hdl)

    if config["console_output"]:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        root_logger.addHandler(console_hdl)
