import argparse
import os
import numpy as np
from scipy.special import logsumexp
from rsgd.common.dat import load_dat
from rsgd.common.logistic import logistic_grad
from rsgd.common.logistic import logistic_loss
from rsgd.common.logistic import logistic_test
from rsgd.common.utils import get_batch_index


def sgd_recur(X, y, grad, batch_size, n_epoch, L,
              init_step=2.0, R=1.0, reg_coeff=0.001,
              verbose=False, loss_func=None, test_func=None):
    N, dim = X.shape

    batch_idx = get_batch_index(N, batch_size)
    m = len(batch_idx) - 1
    mu = reg_coeff

    niter = n_epoch*m + 1
    it = 0

    # initialization
    w = np.zeros((niter, dim))
    sens = np.zeros((niter, m))
    step_size = init_step

    for t in range(n_epoch):
        if m > 1:
            step_size = init_step/np.sqrt(t + 1)

        # recurrence coefficient
        contr_coeff = max(np.abs(1. - step_size*mu),
                          np.abs(1. - step_size*L))
        b = (2.0*R*step_size) / batch_size

        for j in range(m):
            mini_X = X[batch_idx[j]:batch_idx[j+1], :]
            mini_y = y[batch_idx[j]:batch_idx[j+1]]

            # gradient desecent update
            gt = grad(w[it], mini_X, mini_y) / batch_size
            gt += reg_coeff * w[it]

            w[it+1, :] = w[it] - step_size*gt

            for k in range(m):
                sens[it+1, k] = contr_coeff*sens[it, k]

                if k == j:
                    sens[it+1, k] += b

            it += 1

        if verbose:
            objval = loss_func(w[it], X, y)
            acc = test_func(w[it], X, y)*100
            print("[{0}] loss={1:.5f} acc={2:7.3f}".format(t+1, objval, acc))

    # avg_sens = sens[-1, :]
    # last_it = w[-1, :]

    # return last_it, avg_sens
    return w[1:, ], sens[1:, :]


# def sgd_recur(X, y, grad, batch_size, n_epoch, alpha, L, sigma,
#               init_step=2.0, reg_coeff=0.001, averaged=False,
#               verbose=False, loss_func=None, test_func=None, dec_step=True):
#     N, dim = X.shape

#     batch_idx = get_batch_index(N, batch_size)
#     m = len(batch_idx) - 1
#     n_alpha = len(alpha)
#     sig_sq = 2.0 * (sigma**2)
#     mu = reg_coeff
#     R = 1.0

#     niter = n_epoch*m + 1
#     it = 0

#     # initialization
#     w = np.zeros((niter, dim))
#     sens = np.zeros((niter, m, n_alpha))
#     step_size = init_step

#     for t in range(n_epoch):
#         if m > 1:
#             step_size = init_step/np.sqrt(t + 1)

#         # recurrence coefficient
#         contr_coeff = max(np.abs(1. - step_size*mu),
#                           np.abs(1. - step_size*L))
#         b = (2.0*R*step_size) / batch_size

#         for j in range(m):
#             mini_X = X[batch_idx[j]:batch_idx[j+1], :]
#             mini_y = y[batch_idx[j]:batch_idx[j+1]]

#             # gradient desecent update
#             gt = grad(w[it], mini_X, mini_y) / batch_size
#             gt += reg_coeff * w[it]

#             w[it+1, :] = w[it] - step_size*gt

#             for k in range(m):
#                 sens[it+1, k, :] = contr_coeff*sens[it, k, :]
#                 if k == j:
#                     sens[it+1, j, :] += b

#             it += 1

#         if verbose:
#             objval = (loss_func(w[it], X, y)/N
#                       + 0.5*reg_coeff*np.dot(w[it], w[it]))
#             acc = test_func(w[it], X, y)*100
#             print "[{0}] loss={1:.5f} acc={2:7.3f}".format(t+1, objval, acc)

#     if averaged:
#         avg_sens = np.mean(sens[1:, :, :], axis=0)
#         last_it = np.mean(w[1:], axis=0)
#     else:
#         avg_sens = sens[-1, :, :]
#         last_it = w[-1, :]

#     expo = alpha*(alpha-1)*np.square(avg_sens)/sig_sq
#     log_eta = logsumexp(expo, axis=0) - np.log(m)

#     return last_it, log_eta


def momentum(X, y, grad, batch_size, beta, L, sigma, alpha,
             n_epoch, reg_coeff=0.001, verbose=False):
    N, dim = X.shape

    n_batch = int(N / batch_size)
    rem = N % batch_size
    extra = rem/n_batch
    batch_size += extra
    rem = N % batch_size

    m = n_batch
    mu = reg_coeff
    n_alpha = len(alpha)
    sig_sq = 2.0 * (sigma**2)

    batches = np.arange(N)
    np.random.shuffle(batches)

    # initialization
    w = np.zeros(dim)
    v = np.zeros_like(w)
    sens = np.zeros((m, n_alpha))
    sens_p = np.zeros_like(sens)

    for t in range(n_epoch):
        step_size = 2./(t + 1)

        for j in range(m):
            batch_start = batch_start_idx(j, batch_size, rem)
            batch_finish = batch_start_idx(j+1, batch_size, rem)
            rand_idx = batches[batch_start:batch_finish]
            mini_X = X[rand_idx, :]
            mini_y = y[rand_idx]

            # gradient desecent update
            gt = grad(w, mini_X, mini_y)
            gt /= batch_size
            gt += mu * w

            v[:] = beta * v + step_size * gt
            w -= v

            if verbose:
                loss = logistic_loss(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
                print("[{0}] loss={1}".format(t+1, loss))

        # recurrence coefficient
        contr_coeff = max(np.absolute(1. - np.sqrt(step_size*mu)),
                          np.absolute(1. - np.sqrt(step_size*L)))
        expan = 2.0*step_size/batch_size

        if t == 0:
            for j in range(1, m):
                sens[0, :] = (expan**2)*(contr_coeff**(2*(m-j-1)))
        else:
            for j in range(m):
                sens[j, :] = (contr_coeff**(2*m)*(sens[j, :]+sens_p[j, :])
                              + (2.0*expan)*(contr_coeff**(2*(m-1)-j))*np.sqrt(
                                  sens[j, :] + sens_p[j, :])
                              + (expan**2)*(contr_coeff**(2*(m-j-1))))

    expo = alpha*(alpha-1)*sens/sig_sq
    log_eta = logsumexp(expo, axis=0) - np.log(m)

    return w, log_eta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursive mechanism')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('T', type=int, help='epoch')
    parser.add_argument('--data_dir', type=str, default=None)

    args = parser.parse_args()

    # load the dataset
    fpath = os.path.join(args.data_dir, f"{args.dname}.dat")
    X, y = load_dat(fpath, minmax=(0, 1), bias_term=True)
    # y[y < 0.5] = -1.0

    w, sen = sgd_recur(X, y, logistic_grad, 4000, args.T, 0.25,
                       reg_coeff=0.001, init_step=0.5, verbose=True,
                       loss_func=logistic_loss, test_func=logistic_test)
    acc = logistic_test(w[-1], X, y)
    print("accuracy={}".format(acc))
    print("sensitivity={}".format(sen[-1]))
