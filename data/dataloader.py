import pandas as pd
import os
import numpy as np
from collections import Counter


class DataLoader:
    def __init__(self, args):
        self._split_ratio = args["split_ratio"]

        folder = args["data_folder"]
        file_name = args["data_file"]
        data_file = os.path.join(folder, file_name)

        data = pd.read_csv(data_file, sep=",", header=None)

        counter = Counter()
        for x in data:
            counter.update(data[x].value_counts().to_dict())

        freq_list = [k for k, v in counter.most_common()]
        table = {ele: i + 1 for i, ele in enumerate(freq_list)}
        table.update({"<UNK>": 0})

        for x in data:
            data[x] = data[x].map(table)

        data = data.to_numpy(dtype=np.int)

        self._item = data
        self._item_dict = table

    @property
    def item_nums(self):
        return len(self._item_dict)

    def split(self):
        data_size = len(self._item)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self._item = self._item[shuffle_indices]

        test_idx = -1 * int(self._split_ratio * float(data_size))
        train_set, test_set = self._item[:test_idx], self._item[test_idx:]
        return train_set, test_set

    def generate_sub_sessions(self, data, pad_token):
        pad_idx = self._item_dict[pad_token]
        sub_seq_data = []

        sess_size = len(data[0])

        for i in range(len(data)):
            seq = data[i]
            for j in range(sess_size - 1):  # minimal size 2
                sub_seq_end = seq[: len(seq) - j]
                sub_seq_pad = [pad_idx] * j
                sub_seq_data.append(list(sub_seq_pad) + list(sub_seq_end))
        x_train = np.array(sub_seq_data)
        del sub_seq_data

        return x_train
