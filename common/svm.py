import numpy as np
from rsgd.common.dat import load_dat


def svm_loss(w, X, y, clip=-1, reg_coeff=0.0):
    is_1d = w.ndim == 1
    N = X.shape[0] * 1.0

    w2d = np.atleast_2d(w)
    y = np.atleast_2d(y)

    wx = np.dot(X, w2d.T)
    obj = 1.0 - (y.T * wx)
    obj[obj < 0] = 0

    # clipping
    if clip > 0:
        obj[obj > clip] = clip

    # reg = lmbda * np.sum(np.square(w[:, 1:]), axis=1)
    loss = np.sum(obj, axis=0)

    loss /= N

    # loss = hinge + reg

    if is_1d:
        loss = np.asscalar(loss)
        loss += 0.5 * reg_coeff * np.dot(w, w)
    else:
        loss += 0.5 * reg_coeff * np.sum(np.square(w2d), axis=1)

    return loss


def svm_grad(w, X, y, clip=-1):
    y2d = np.atleast_2d(y)

    ywx = y * np.dot(X, w)
    loc = ywx < 1
    per_grad = -1.0 * y2d[:, loc].T * X[loc]

    if clip > 0:
        norm = np.linalg.norm(per_grad, axis=1)
        to_clip = norm > clip
        per_grad[to_clip, :] = ((clip * per_grad[to_clip])
                                / np.atleast_2d(norm[to_clip]).T)
        grad = np.sum(per_grad, axis=0)
    else:
        grad = np.sum(per_grad, axis=0)

    return grad


def hsvm_loss(w, X, y, h=0.5, reg_coeff=0.0, clip=-1):
    """
    huberized SVM loss
    """
    is_1d = w.ndim == 1
    N = X.shape[0] * 1.0

    w2d = np.atleast_2d(w)
    y = np.atleast_2d(y)

    wx = np.dot(X, w2d.T)
    z = y.T * wx

    obj = 0.25 * np.square(1 + h - z) / h
    above = z > 1 + h
    below = z < 1 - h
    obj[above] = 0
    obj[below] = 1 - z[below]

    if clip > 0:
        obj[obj > clip] = clip

    loss = np.sum(obj, axis=0)

    if is_1d:
        loss = np.asscalar(loss)

    loss /= N

    if is_1d:
        loss += 0.5 * reg_coeff * np.dot(w, w)
    else:
        loss += 0.5 * reg_coeff * np.sum(np.square(w2d), axis=1)

    return loss


def hsvm_grad(w, X, y, h=0.5, clip=-1):
    y2d = np.atleast_2d(y)

    ywx = y * np.dot(X, w)
    per_grad = 0.5 * y2d.T * np.atleast_2d(ywx - 1 - h).T * X / h

    above_loc = ywx > 1 + h
    below_loc = ywx < 1 - h
    per_grad[above_loc, :] = 0
    per_grad[below_loc, :] = -y2d[:, below_loc].T * X[below_loc, :]

    if clip > 0:
        per_grad[per_grad > clip] = clip

    grad = np.sum(per_grad, axis=0)

    return grad


def hsvm_loss_and_grad(w, X, y, h=0.5, reg_coeff=0.0):
    N = X.shape[0] * 1.0

    loss = hsvm_loss(w, X, y, h, reg_coeff=reg_coeff)
    grad = hsvm_grad(w, X, y, h) / N
    grad += reg_coeff * w

    return loss, grad


def svm_test(w, X, y):
    is_1d = w.ndim == 1

    N = X.shape[0]
    w2d = np.atleast_2d(w)
    y2d = np.atleast_2d(y)

    wx = np.dot(X, w2d.T)
    sign = y2d.T * wx

    cnt = np.count_nonzero(sign >= 0, axis=0)

    if is_1d:
        cnt = np.squeeze(cnt)

    return cnt / float(N)
