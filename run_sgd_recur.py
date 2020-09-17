import argparse
import os
import time
import timeit
import numpy as np
from scipy.special import logsumexp
from rsgd.algo.recur_mech import sgd_recur
from rsgd.algo.gmm import renyi
from rsgd.common.dat import load_dat
from rsgd.common.logistic import logit_loss
from rsgd.common.logistic import logistic_grad
from rsgd.common.logistic import logistic_test
from rsgd.common.svm import hsvm_loss
from rsgd.common.svm import hsvm_grad
from rsgd.common.svm import svm_test
from sklearn.model_selection import RepeatedKFold


def compute_epsilon(sens, alpha, sigma_sq, delta, m):
    expo = (alpha*(alpha-1)*np.square(sens)[:, np.newaxis]
            / sigma_sq[:, np.newaxis, np.newaxis])
    log_eta = logsumexp(expo, axis=1) - np.log(m)
    rdp_eps = log_eta / (alpha - 1.0)
    dp_eps = rdp_eps + np.log(1.0/delta) / (alpha - 1.0)

    # min_idx = np.argmin(dp_eps, axis=1)
    # for i in range(len(min_idx)):
    #     sigma = np.sqrt(0.5*sigma_sq[i])
    #     print "sigma={} rdp_eps={} min_eps={} alpha={}".format(
    #         sigma, rdp_eps[i, min_idx[i]], dp_eps[i, min_idx[i]],
    #         alpha[min_idx[i]])

    return np.min(dp_eps, axis=1)


def epsilon_worst_case(sens, alpha, sigma_sq, delta):
    sens_sq = (sens[-1])**2
    rdp_eps = alpha*sens_sq/sigma_sq[:, np.newaxis]
    dp_eps = rdp_eps + np.log(1.0/delta) / (alpha - 1.0)

    return np.min(dp_eps, axis=1)


def compute_exact_renyi(sens, alpha, sigma, delta, m):
    n_alpha = len(alpha)
    n_sigma = len(sigma)
    bottom = np.zeros(m)

    min_eps = np.zeros(n_sigma)

    for i in range(n_sigma):
        dp_eps = np.zeros(n_alpha)
        rdp_eps = np.zeros_like(dp_eps)

        for j in range(n_alpha):
            rdp_eps[j] = renyi(alpha[j], sigma[i], sens, bottom)
            dp_eps[j] = rdp_eps[j] + np.log(1.0/delta) / (alpha[j] - 1.0)

        min_eps[i] = np.min(dp_eps)

        # min_idx = np.argmin(dp_eps)
        # print "simga={} min_eps={} alpha={} rdp_eps={}".format(
        #     sigma[i], min_eps[i], alpha[min_idx], rdp_eps[min_idx])

    return min_eps


def main(args):
    # load the dataset
    dname = f"{args.dname}.dat"
    fpath = os.path.join(args.data_dir, dname)
    X, y = load_dat(fpath, shuffle=args.shuffle, normalize=args.norm)
    N, dim = X.shape

    # order of Renyi divergence
    alpha = np.linspace(1.5, 2560, 2000)
    delta = args.delta
    sigma = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4])
    sigma = np.flip(sigma, 0)
    batch_size = args.batch_size

    n_sigma = len(sigma)
    cv_rep = 10
    K = 5
    n_rep = K * cv_rep

    eps = np.zeros((2, n_sigma, n_rep))
    acc = np.zeros_like(eps)
    obj = np.zeros_like(eps)
    j = 0

    # task
    if args.svm:
        loss_func = hsvm_loss
        grad_func = hsvm_grad
        test_func = svm_test
        y[y < 0.5] = -1.0
        task = 'svm'
    else:
        loss_func = logit_loss
        grad_func = logistic_grad
        test_func = logistic_test
        task = 'logres'

    rkf = RepeatedKFold(n_splits=K, n_repeats=cv_rep)

    for train_idx, test_idx in rkf.split(X):
        train_X, train_y = X[train_idx, :], y[train_idx]
        test_X, test_y = X[test_idx, :], y[test_idx]

        train_size = train_X.shape[0] * 1.0
        m = int(train_size / batch_size)

        noise = np.random.randn(dim)

        # new recurrence relation
        w, sens = sgd_recur(train_X, train_y, grad_func,
                            batch_size, args.T, args.L,
                            reg_coeff=args.mu, R=args.norm,
                            init_step=args.init_step,
                            verbose=False)

        sigma_sq = 2.0 * np.square(sigma)
        eps[0, :, j] = compute_epsilon(sens[-1, :], alpha, sigma_sq, delta, m)
        # eps[0, :, j] = compute_exact_renyi(sens[-1], alpha, sigma, delta, m)

        noisy_w = w[-1, :] + sigma[:, np.newaxis] * noise
        acc[0, :, j] = test_func(noisy_w, test_X, test_y)*100
        obj[0, :, j] = loss_func(noisy_w, train_X, train_y, reg_coeff=args.mu)

        # averaged sgd
        avg_sens = np.mean(sens, axis=0)
        eps[1, :, j] = compute_epsilon(avg_sens, alpha, sigma_sq, delta, m)

        noisy_w = np.mean(w, axis=0) + sigma[:, np.newaxis] * noise
        acc[1, :, j] = test_func(noisy_w, test_X, test_y)*100
        obj[1, :, j] = loss_func(noisy_w, train_X, train_y, reg_coeff=args.mu)
        j += 1

    avg_acc = np.mean(acc, axis=2)
    avg_eps = np.mean(eps, axis=2)
    avg_obj = np.mean(obj, axis=2)

    str_mu = "{0}".format(args.mu)[2:]
    str_is = "{0}".format(args.init_step).replace('.', '').rstrip('0')
    filename = "sgdr_{5}_T{0}B{1}mu{2}IS{3}_{4}".format(
        args.T, args.batch_size, str_mu, str_is, args.dname, task)
    rs_dir = "./plot/results"
    np.savetxt("{0}/{1}_eps.out".format(rs_dir, filename), avg_eps, fmt='%.5f')
    np.savetxt("{0}/{1}_acc.out".format(rs_dir, filename), avg_acc, fmt='%.5f')
    np.savetxt("{0}/{1}_obj.out".format(rs_dir, filename), avg_obj, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursive mechanism')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('T', type=int, help='total number of iterations')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--init_step', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--L', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--norm', type=float, default=1.0)
    parser.add_argument('--svm', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    start_time = timeit.default_timer()

    main(args)
