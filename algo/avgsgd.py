import argparse
import os
import numpy as np
from rsgd.common.dat import load_dat
from rsgd.common.logistic import logistic_grad
from rsgd.common.logistic import logistic_loss
from rsgd.common.logistic import logistic_test
from rsgd.common.utils import get_batch_index
from sklearn.utils import shuffle


def sgd_restart(X, y, grad, batch_size, n_epoch, L,
                init_step=2.0, R=1.0, reg_coeff=0.001, reset_intvl=20,
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
    last_avg_idx = 1
    epoch_cnt = 0

    for t in range(n_epoch):
        if m > 1:
            step_size = init_step/(epoch_cnt + 1)

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
            gt /= np.linalg.norm(gt)

            w[it+1, :] = w[it] - step_size*gt

            sens[it+1, :] = contr_coeff*sens[it, :]
            sens[it+1, j] += b

            # increase the total number of iteration counts
            it += 1

        # averaging and reset the step size
        if (t % reset_intvl) == 0:
            w[it, :] = np.mean(w[last_avg_idx:it+1, :], axis=0)
            sens[it, :] = np.mean(sens[last_avg_idx:it+1, :], axis=0)
            last_avg_idx = it + 1
            epoch_cnt = 0
        else:
            epoch_cnt += 1

        if verbose:
            objval = loss_func(w[it], X, y)
            acc = test_func(w[it], X, y)*100
            print("[{0}] loss={1:.5f} acc={2:7.3f}".format(t, objval, acc))

    return w[-1], sens[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursive mechanism')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('T', type=int, help='epoch')
    parser.add_argument('rst', type=int, help='reset interval')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--init_step', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--L', type=float, default=1.81)
    parser.add_argument('--norm', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-12)

    args = parser.parse_args()

    # load the dataset    
    fpath = os.path.join(args.data_dir, f"{args.dname}.dat")
    X, y = load_dat(fpath, normalize=args.norm)
    # y[y < 0.5] = -1.0

    batch_size = args.batch_size
    n_epoch = args.T
    w, sens = sgd_restart(X, y, logistic_grad, batch_size, n_epoch, args.L,
                          reg_coeff=args.mu, reset_intvl=args.rst,
                          R=args.norm, init_step=0.5, verbose=True,
                          loss_func=logistic_loss, test_func=logistic_test)
    acc = logistic_test(w, X, y)*100
    print("accuracy={}".format(acc))
    print("sensitivity={}".format(sens))
