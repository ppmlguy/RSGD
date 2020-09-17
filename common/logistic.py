import numpy as np
from scipy.special import expit


def logistic_loss(w, X, y, clip=-1):
    is_1d = w.ndim == 1

    w = np.atleast_2d(w)
    y = np.atleast_2d(y)

    wx = np.dot(X, w.T)

    # obj = -log_logistic(-wx) - (y.T * wx)
    obj = np.log(1.+np.exp(wx)) - (y.T * wx)

    if clip > 0:
        obj[obj > clip] = clip

    loss = np.sum(obj, axis=0)

    if is_1d:
        loss = np.asscalar(loss)

    return loss


def logit_loss(w, X, y, reg_coeff=0.0, axis=1):
    N = X.shape[0] * 1.0

    is_1d = w.ndim == 1

    loss = logistic_loss(w, X, y) / N

    if is_1d:
        loss += 0.5 * reg_coeff * np.dot(w, w)
    else:
        loss += 0.5 * reg_coeff * np.sum(np.square(w), axis=axis)

    return loss


def logistic_grad(w, X, y, clip=-1):
    N = X.shape[0]

    wx = np.dot(X, w)
    z = expit(wx)
    z0 = z - y

    if clip > 0:
        per_grad = X * z0.reshape(N, -1)
        norm = np.linalg.norm(per_grad, axis=1)
        to_clip = norm > clip
        per_grad[to_clip, :] = ((clip * per_grad[to_clip])
                                / np.atleast_2d(norm[to_clip]).T)
        grad = np.sum(per_grad, axis=0)
    else:
        grad = np.dot(X.T, z0)

    return grad


def logistic_loss_and_grad(w, X, y, obj_clip=-1, grad_clip=-1):
    """
    assume that w is 1d ndarray object.
    """
    N = X.shape[0]

    wx = np.dot(X, w)

    # obj = -log_logistic(-wx) - (y * wx)
    obj = np.log(1.+np.exp(wx)) - (y*wx)

    if obj_clip > 0:
        obj[obj > obj_clip] = obj_clip

    loss = np.sum(obj, axis=0)

    z = expit(wx)
    z0 = z - y

    if grad_clip > 0:
        per_grad = X * z0.reshape(N, -1)
        norm = np.linalg.norm(per_grad, axis=1)
        to_clip = norm > grad_clip
        per_grad[to_clip, :] = ((grad_clip * per_grad[to_clip])
                                / np.atleast_2d(norm[to_clip]).T)
        grad = np.sum(per_grad, axis=0)
    else:
        grad = np.dot(X.T, z0)

    return loss, grad


def logistic_grad_hess(w, X, y, lmbda=0.0):
    """
    Computes the gradient and the Hessian, in the case of a logistic loss.

    Parameters
    ----------
    w : coefficient vector
    X, y: input dataset
    """
    N, dim = X.shape

    wx = np.dot(X, w)
    z = expit(wx)
    z0 = z - y

    grad = np.dot(X.T, z0) / N + lmbda * w

    # The mat-vec product of the Hessian
    d = z * (1 - z)
    dX = d[:, np.newaxis] * X / N

    def Hs(s):
        ret = np.empty_like(s)
        ret[:] = X.T.dot(dX.dot(s))
        ret += lmbda*s

        return ret

    return grad, Hs


def logit_loss_and_grad(w, X, y, reg_coeff=0.0):
    """
    assume that w is 1d ndarray object.
    """
    N = X.shape[0]

    wx = np.dot(X, w)

    # obj = -log_logistic(-wx) - (y * wx)
    obj = np.log(1.+np.exp(wx)) - (y * wx)

    loss = np.sum(obj, axis=0) / float(N)
    loss += 0.5 * reg_coeff * np.dot(w, w)

    z = expit(wx)
    z0 = z - y

    grad = np.dot(X.T, z0) / float(N)
    grad += reg_coeff * w

    return loss, grad


def logistic_test(w, X, y):
    is_1d = w.ndim == 1

    n_obs = X.shape[0]

    Y = np.copy(y)
    Y[Y < 0.5] = -1.0
    Y = np.atleast_2d(Y)

    w2d = np.atleast_2d(w)
    wx = np.dot(X, w2d.T)
    wx[wx == 0] = 1.0  # points at the decision boundary are classified as positive
    sign = Y.T * wx

    cnt = np.count_nonzero(sign > 0, axis=0)

    if is_1d:
        cnt = np.squeeze(cnt)

    return cnt / float(n_obs)
