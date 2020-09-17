import argparse
import time
import numpy as np
from prox_grad.common.utils import get_batch_index
from prox_grad.common.dat import load_dat
from agd.common.logistic import logistic_grad
from agd.common.logistic import logistic_test
from agd.common.logistic import logistic_loss


class SGD(object):
    def __init__(self, w0, algo='sgd', reg_coeff=0.001, averaged=False,
                 beta1=0.9, beta2=0.999):
        self.algo = algo
        self.func = None
        self.funcdict = {
            'sgd': self._sgd_update,
            'adagrad': self._adagrad_update,
            'rmsprop': self._rmsprop_update,
            'adam': self._adam_update,
            'hball': self._heavyball_update,
            'nag': self._nag_update,
        }
        self.reg_coeff = reg_coeff

        self.w = np.copy(w0)
        self.beta1 = beta1

        if self.algo != 'sgd':
            self.v = np.zeros_like(self.w)

        if self.algo in ['adam', 'adadelta']:
            self.u = np.zeros_like(self.w)
            self.beta2 = beta2

        if self.algo == 'adagrad':
            self.eps = 1e-8
        elif self.algo == 'adam':
            self.eps = 1e-8
            self.beta1_exp = 1.0
            self.beta2_exp = 1.0
        elif self.algo in ['rmsprop', 'adadelta']:
            self.eps = 1e-5

        self.averaged = averaged

        if averaged:
            self.avg_w = np.zeros_like(self.w)
            self.cnt = 0

    def update(self, grad, step_size, X, y):
        algo = self.algo
        N = X.shape[0]

        if algo == 'nag':
            jump = self.w - self.beta1*self.v
            gt = grad(jump, X, y) / N + self.reg_coeff * jump
        else:
            gt = grad(self.w, X, y) / N + self.reg_coeff * self.w

        # update parameter
        self.funcdict[algo](gt, step_size)

        if self.averaged:
            self.avg_w = (self.avg_w*self.cnt + self.w) / (self.cnt + 1)
            self.cnt += 1

    def _sgd_update(self, grad, step_size):
        self.w -= step_size * grad

    def _adagrad_update(self, grad, step_size):
        self.v += np.square(grad)
        self.w -= step_size*grad/np.sqrt(self.v + self.eps)

    def _rmsprop_update(self, grad, step_size):
        self.v[:] = self.beta1*self.v + (1.0 - self.beta1)*np.square(grad)
        self.w -= step_size/np.sqrt(self.v + self.eps) * grad

    def _adam_update(self, grad, step_size):
        beta1 = self.beta1
        beta2 = self.beta2

        self.u[:] = beta1*self.u + (1.0-beta1)*grad
        self.v[:] = beta2*self.v + (1.0-beta2)*np.square(grad)
        self.beta1_exp *= beta1
        self.beta2_exp *= beta2

        uhat = self.u / (1.0 - self.beta1_exp)
        vhat = self.v / (1.0 - self.beta2_exp)
        self.w -= step_size*uhat/(np.sqrt(vhat) + self.eps)

    def _heavyball_update(self, grad, step_size):
        self.v = self.beta1 * self.v + step_size * grad
        self.w -= self.v

    def _nag_update(self, grad, step_size):
        self.v[:] = self.beta1*self.v + step_size*grad
        self.w -= self.v

        # alternative implementation
        # oldv = np.copy(self.v)
        # self.v[:] = self.w - step_size*grad
        # self.w[:] = (1+self.beta1)*self.v - self.beta1*oldv


def gd(X, y, grad, max_iter, reg_coeff=0.001, init_step=0.5, pgtol=1e-5,
       verbose=False, loss_func=None, test_func=None):
    N, dim = X.shape

    # initialization
    w = np.zeros(dim)
    step_size = init_step  # fixed step size

    for t in range(max_iter):
        # gradient desecent update
        gt = grad(w, X, y) / N + reg_coeff * w
        w -= step_size * gt

        if np.max(np.abs(gt)) <= pgtol:
            break

        if verbose:
            loss = loss_func(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
            acc = test_func(w, X, y)*100
            print "[{0}] loss={1} acc={2}".format(t+1, loss, acc)

    return w


def nag(X, y, grad, batch_size, max_epoch, gamma, reg_coeff=0.001,
        init_step=2.0, verbose=False, loss_func=None, test_func=None):
    N, dim = X.shape

    if batch_size <= 0:
        batch_size = N

    batch_idx = get_batch_index(N, batch_size)
    m = len(batch_idx) - 1

    # initialization
    w = np.zeros(dim)
    v = np.zeros_like(w)

    for t in range(max_epoch):
        step_size = init_step/np.sqrt(t + 1)

        for j in range(m):
            mini_X = X[batch_idx[j]:batch_idx[j+1], :]
            mini_y = y[batch_idx[j]:batch_idx[j+1]]

            # gradient desecent update
            jump = w - gamma*v
            gt = grad(jump, mini_X, mini_y) / batch_size + reg_coeff * jump

            v[:] = gamma*v + step_size*gt
            w -= v

        # print accuracy
        if verbose:
            loss = loss_func(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
            acc = test_func(w, X, y)*100
            print "[{0}] loss={1:.5f} acc={2:8.5f}".format(t+1, loss, acc)

    return w


def rmsprop(X, y, grad, batch_size, max_epoch, gamma, eps, reg_coeff=0.001,
            init_step=0.01, verbose=False, loss_func=None, test_func=None):
    N, dim = X.shape

    if batch_size <= 0:
        batch_size = N

    batch_idx = get_batch_index(N, batch_size)
    m = len(batch_idx) - 1

    # initialization
    w = np.zeros(dim)
    v = np.zeros_like(w)

    for t in range(max_epoch):
        step_size = init_step/np.sqrt(t + 1)

        for j in range(m):
            mini_X = X[batch_idx[j]:batch_idx[j+1], :]
            mini_y = y[batch_idx[j]:batch_idx[j+1]]

            # gradient desecent update
            gt = grad(w, mini_X, mini_y) / batch_size
            gt += reg_coeff * w

            v[:] = gamma*v + (1.0 - gamma)*np.square(gt)
            w -= step_size/np.sqrt(v + eps) * gt

        # print accuracy
        if verbose:
            loss = loss_func(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
            acc = test_func(w, X, y)*100
            print "[{0}] loss={1:.5f} acc={2:8.5f}".format(t+1, loss, acc)

    return w


def adam(X, y, grad, batch_size, max_epoch, beta1=0.9, beta2=0.999, eps=1e-8,
         reg_coeff=0.001, init_step=0.01, verbose=False, loss_func=None,
         test_func=None):
    N, dim = X.shape

    if batch_size <= 0:
        batch_size = N

    batch_idx = get_batch_index(N, batch_size)
    m = len(batch_idx) - 1

    # initialization
    w = np.zeros(dim)
    v = np.zeros_like(w)
    u = np.zeros_like(w)

    beta1_exp = 1.0
    beta2_exp = 1.0

    for t in range(max_epoch):
        step_size = init_step/np.sqrt(t + 1)

        for j in range(m):
            mini_X = X[batch_idx[j]:batch_idx[j+1], :]
            mini_y = y[batch_idx[j]:batch_idx[j+1]]

            # gradient desecent update
            gt = grad(w, mini_X, mini_y) / batch_size + reg_coeff * w

            u[:] = beta1*u + (1.0-beta1)*gt
            v[:] = beta2*v + (1.0-beta2)*np.square(gt)
            beta1_exp *= beta1
            beta2_exp *= beta2

            uhat = u / (1.0 - beta1_exp)
            vhat = v / (1.0 - beta2_exp)
            w -= step_size*uhat/(np.sqrt(vhat) + eps)

            # print accuracy
            if verbose:
                loss = loss_func(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
                acc = test_func(w, X, y)*100
                print "[{0}] loss={1:.5f} acc={2:8.5f}".format(t+1, loss, acc)

    return w


def adagrad(X, y, grad, batch_size, max_epoch, eps=1e-8,
            reg_coeff=0.001, init_step=0.01, verbose=False, loss_func=None,
            test_func=None):
    N, dim = X.shape

    if batch_size <= 0:
        batch_size = N

    batch_idx = get_batch_index(N, batch_size)
    m = len(batch_idx) - 1

    # initialization
    w = np.zeros(dim)
    v = np.zeros_like(w)

    for t in range(max_epoch):
        step_size = init_step/np.sqrt(t + 1)

        for j in range(m):
            mini_X = X[batch_idx[j]:batch_idx[j+1], :]
            mini_y = y[batch_idx[j]:batch_idx[j+1]]

            # gradient desecent update
            gt = grad(w, mini_X, mini_y) / batch_size + reg_coeff * w

            v += np.square(gt)
            w -= step_size*gt/np.sqrt(v + eps)

            # print accuracy
            if verbose:
                loss = loss_func(w, X, y)/N + 0.5*reg_coeff*np.dot(w, w)
                acc = test_func(w, X, y)*100
                print "[{0}] loss={1:.5f} acc={2:8.5f}".format(t+1, loss, acc)

    return w


def main(args):
    # load the dataset
    fpath = "../../../Experiment/Dataset/dat/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, normalize=args.norm)
    N, dim = X.shape

    gd(X, y, logistic_grad, args.T, init_step=args.init_step,
       verbose=True, loss_func=logistic_loss, test_func=logistic_test)

    # sgd(X, y, logistic_grad, args.batch_size, args.T, args.mu, args.init_step,
    #     verbose=True)
    # rmsprop(X, y, logistic_grad, args.batch_size, args.T, gamma, eps,
    #         reg_coeff=args.mu, init_step=args.init_step, verbose=True,
    #         loss_func=logistic_loss, test_func=logistic_test)

    # adam(X, y, logistic_grad, args.batch_size, args.T, beta1=0.9, beta2=0.999,
    #      eps=1e-8, reg_coeff=args.mu, init_step=args.init_step, verbose=True,
    #      loss_func=logistic_loss, test_func=logistic_test)

    # adagrad(X, y, logistic_grad, args.batch_size, args.T, eps=eps,
    #         reg_coeff=args.mu, init_step=args.init_step, verbose=True,
    #         loss_func=logistic_loss, test_func=logistic_test)

    # nag(X, y, logistic_grad, args.batch_size, args.T, 0.9,
    #     reg_coeff=args.mu, init_step=args.init_step, verbose=True,
    #     loss_func=logistic_loss, test_func=logistic_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursive mechanism')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('T', type=int, help='number of epochs')
    parser.add_argument('--mu', type=float, default=0.001,
                        help='strong convexity const.')
    parser.add_argument('--batch_size', type=int, default=4000,
                        help='batch size')
    parser.add_argument('--init_step', type=float, default=2.0,
                        help="initial step size")
    parser.add_argument('--norm', type=float, default=-1)

    args = parser.parse_args()

    print "Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    start_time = time.clock()

    main(args)

    elapsed = time.clock() - start_time
    mins, sec = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)

    print "The program finished. [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Elasepd time: %d:%02d:%02d" % (hrs, mins, sec)
