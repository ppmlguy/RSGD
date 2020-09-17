import numpy as np


def batch_start_idx(batch_idx, batch_size, rem):
    start_idx = min(batch_idx, rem) * (batch_size + 1)
    start_idx += max(0, batch_idx-rem) * batch_size

    return start_idx


def get_batch_index(N, batch_size):
    n_batch, rem = divmod(N, batch_size)
    extra = rem / n_batch
    batch_size += extra
    rem = N % batch_size

    batch_idx = np.zeros(n_batch+1, dtype=int)

    for i in range(n_batch):
        batch_idx[i+1] = batch_start_idx(i+1, batch_size, rem)

    return batch_idx
