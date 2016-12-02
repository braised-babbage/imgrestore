from sys import stdout
from math import sqrt


def cp_saddle(K, prox_Fstar, prox_G, iter,
              tau, sigma, theta, x0, y0, log=stdout):
    """ Solves the saddle point problem"""
    """    min_x max_y <Kx,y> + G(x) - Fs(y) """
    """ Uses iter iterations of the Chambolle-Pock algorithm."""
    """We require that tau, sigma > 0   and 0 <= theta <= 1 """
    x = x0
    y = y0
    z = x0
    Kt = K.T
    for n in range(0,iter):
        log.write("cp it. %d\n" % (n+1))
        y = prox_Fstar(y + sigma*K*z, sigma)
        xold = x
        x = prox_G(xold - tau*Kt*y, tau)
        z = x + theta*(x-xold)
    return x,y


def cp_saddle_uniform(K, prox_Fstar, prox_G, gamma, iter,
                      tau0, sigma0, theta0, x0, y0, log=stdout):
    """ Solves the saddle point problem
        min_x max_y <Kx,y> + G(x) - Fs(y)
    Uses iter iterations of the Chambolle-Pock algorithm.
    G is uniformly convex with constant gamma
    We require that tau > 0, sigma0 > 0   and 0 <= theta0 <= 1. """
    x = x0
    y = y0
    z = x0
    tau = tau0
    sigma = sigma0
    theta = theta0
    Kt = K.T
    for n in range(0,iter):
        log.write("cp it. %d\n" % (n+1))
        y = prox_Fstar(y + sigma*K*z, sigma)
        xold = x
        x = prox_G(xold - tau*Kt*y, tau)
        theta = 1. / sqrt(1+2*gamma*tau)
        tau = theta*tau
        sigma = sigma/theta
        z = x + theta*(x-xold)
    return x,y
