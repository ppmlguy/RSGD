import argparse
import os
import numpy as np
from gaussian_moments import compute_log_moment
from gaussian_moments import get_privacy_spent
from rsgd.common.dat import load_dat


def compute_epsilon(X, y, batch_size, sigma=4):
    N, dim = X.shape

    # sampling probability
    q = batch_size / (N * 1.0)

    # moments accountant
    max_lmbd = 32
    epsilon = []
    T = [1, 10, 100, 1000]
    delta = 1e-8

    for niter in T:
        log_moments = []
        for lmbd in xrange(1, max_lmbd+1):
            log_moment = compute_log_moment(q, sigma, niter, lmbd)
            log_moments.append((lmbd, log_moment))

        eps, _ = get_privacy_spent(log_moments, target_delta=delta)
        epsilon.append(eps)

    return epsilon


def main():
    parser = argparse.ArgumentParser(description='sgd-ma')
    parser.add_argument('dname', help='dataset name')

    args = parser.parse_args()

    # load the dataset
    fpath = os.path.join(args.data_dir, f"{args.dname}.dat")
    X, y = load_dat(fpath)

    batch_size = int(np.sqrt(X.shape[0]) + 10)
    eps = compute_epsilon(X, y, batch_size)
    print(eps)


if __name__ == "__main__":
    main()
