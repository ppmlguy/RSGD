from scipy.stats import norm as ssnorm
from scipy.integrate import quad as siquad
from numpy import exp
from numpy import log1p
from numpy import log
import numpy as np


def addlog(accum, mu, sigma, x):
    den = ssnorm.logpdf(x, loc=mu, scale=sigma)
    ma = max(accum, den)
    mi = min(accum, den)
    result = ma + log1p(exp(mi-ma))
    return result


def getdensity(alpha, sigma, top, bottom):
    def density(x):
        topsum = ssnorm.logpdf(x, loc=top[0], scale=sigma)
        for mu in top[1:]:
            topsum = addlog(topsum, mu, sigma, x)
        bottomsum = ssnorm.logpdf(x, loc=bottom[0], scale=sigma)
        for mu in bottom[1:]:
            bottomsum = addlog(bottomsum, mu, sigma, x)
        res = alpha * (topsum - bottomsum) + bottomsum - log(len(bottom))
        return exp(res)
    return density


def exprenyi(alpha, sigma, top, bottom):
    """ exponential of renyi divergence """
    return siquad(getdensity(alpha, sigma, top, bottom),
                  -np.infty, np.infty)[0]


def renyi(alpha, sigma, top, bottom):
    return log(exprenyi(alpha, sigma, top, bottom))/(alpha - 1)
