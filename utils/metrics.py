import numpy as np


def sample_top_k_old(a, top_k):
    idx = np.argsort(a)[:, ::-1]
    idx = idx[:, :top_k]
    return idx


def sample_top_k(a, top_k):
    idx = np.argpartition(a, -top_k)[:, -top_k:]
    part = np.take_along_axis(a, idx, 1)
    return np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


def sample_top_ks_old(a, top_ks):
    # O(n * log(n)) + b * O(1)
    idx = np.argsort(a)[:, ::-1]
    for k in top_ks:
        yield idx[:, :k]


def sample_top_ks(a, top_ks):
    # O(b * (n + k * log(k)))
    for k in top_ks:
        idx = np.argpartition(a, -k)[:, -k:]
        part = np.take_along_axis(a, idx, 1)
        yield np.take_along_axis(idx, np.argsort(part), 1)[:, ::-1]


# mrr@K, hit@K, ndcg@k
def get_metric(rank_indices):
    mrr_list, hr_list, ndcg_list = [], [], []
    for t in rank_indices:
        if len(t):
            mrr_list.append(1.0 / (t[0][0] + 1))
            ndcg_list.append(1.0 / np.log2(t[0][0] + 2))
            hr_list.append(1.0)
        else:
            mrr_list.append(0.0)
            ndcg_list.append(0.0)
            hr_list.append(0.0)

    return mrr_list, hr_list, ndcg_list
