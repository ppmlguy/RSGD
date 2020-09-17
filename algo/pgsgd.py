import argparse
import numpy as np
import timeit
import time
from rsgd.common.dat import load_dat
from rsgd.common.logistic import logistic_grad
from rsgd.common.logistic import logistic_test
from rsgd.common.logistic import logistic_loss


def pgsgd(X, y, grad_func, max_epoch, rho, batch_size=1,
          reg_coeff=0.0, verbose=False, loss_func=None, test_func=None):
    N, dim = X.shape

    # initialization
    wt = np.zeros((max_epoch+1, dim))

    for s in range(max_epoch):
        w = np.zeros((s+2, dim))
        w[0, :] = wt[s, :]

        for t in range(s+1):
            rand_idx = np.random.randint(N, size=batch_size)
            mini_X = X[rand_idx, :]
            mini_y = y[rand_idx]

            # gradient desecent update
            v = grad_func(w[t], mini_X, mini_y)/batch_size + reg_coeff*w[t]
            v += rho * (w[t] - wt[s])

            step_size = 2/(rho*(t+50.0))
            w[t+1] = w[t] - step_size * v

        wt[s+1] = np.mean(w, axis=0)

        if verbose:
            theta = wt[s+1]
            loss = (loss_func(theta, X, y)/N
                    + 0.5*reg_coeff*np.dot(theta, theta))
            acc = test_func(theta, X, y)*100
            print "[{0}] loss={1:.5f} acc={2:8.5f}".format(s+1, loss, acc)

    return wt[-1]


def main(args):
    # load the dataset
    fpath = os.path.join(args.data_dir, f"{args.dname}.dat")
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N, dim = X.shape

    sigma = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])

    w = pgsgd(X, y, logistic_grad, args.T, args.rho,
              reg_coeff=args.mu, verbose=True,
              loss_func=logistic_loss,
              test_func=logistic_test)

    noise = np.random.randn(dim)

    for i in range(len(sigma)):
        noisy_w = w + sigma[i] * noise
        acc = logistic_test(noisy_w, X, y) * 100.0
        print "sigma={0:.5f} accuracy={1:5.3f}".format(sigma[i], acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursive mechanism')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('T', type=int, help='number of epochs')
    parser.add_argument('rho', type=float, default=0.001,
                        help='strong convexity const.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--mu', type=float, default=0.001,
                        help='strong convexity const.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    args = parser.parse_args()

    print "Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    start_time = timeit.default_timer()

    main(args)

    elapsed = timeit.default_timer() - start_time
    mins, sec = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)

    print "The program finished. [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Elasepd time: %d:%02d:%02d" % (hrs, mins, sec)
