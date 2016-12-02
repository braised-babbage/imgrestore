import numpy as np
import scipy as sp
from os import stdout
from math import sqrt
from scipy.sparse.linalg import cgs


def fista(A,b,mu,g,prox_g, x_0=False,iter=200, log=stdout):
    """Solves min (1/2)||Ax - b||^2 + mu*g(x) by backtracking FISTA
    prox_g is the proximal function of g,
    prox_g(v,s) = argmin_u (1/2)||u-v||^2 + s*g(u)
    
    x_0 is the initial value (set to 0 by default)
    mu is a regularization parameter"""

    def F(x):
        r = A*x - b
        return 0.5*np.dot(r,r) + mu*g(x)
    
    def P(L,y):
        grad_f = A.T * (A*y - b)
        return prox_g(y - 1./L * grad_f, mu/L)

    n,m = A.shape

    if x_0:
        x = x_0
    else: 
        # this might be a terrible idea, but i'm not sure what else to do
        x,_ = cgs(A.T*A + mu*sp.sparse.identity(m),(A.T)*b)

    y = x
    t = 1.0
    L = 1.0
    eta = 2.0
    
    log.write('L = %f\n' % L)
    for k in range(0,iter):
        log.write('iter. %d\n' % k)

        xold = x
        x = P(L,y)
        r = A*y - b
        f_y = 0.5*np.dot(r,r)
        grad_f_y = (A.T)*r
        Q = f_y + np.dot(x-y,grad_f_y) + (L/2)*np.dot(x-y,x-y) + mu*g(x)
        while F(x) > Q:
            L = L*eta
            log.write('L = %f\n' % L)
            x = P(L,y)
            Q = f_y + np.dot(x-y,grad_f_y) + (L/2)*np.dot(x-y,x-y) + mu*g(x)
        told = t
        t = (1+sqrt(1+4*told*told))/2
        y = x + ((told-1)/t)*(x - xold)

    return x


def fistaL(A,b,prox_g,L,x_0 = False, iter=200, log=stdout):
    """ Solves min (1/2)||Ax-b||^2 + g(x) by FISTA
    
    L is the lipschitz bound for A
    
    prox_g is the proximal function of g
    prox_g(v,s) = argmin_u (1/2)||u-v||^2 + s*g(u)
    
    x_0 is the initial value (set to 0 by default) """

    n,m = A.shape

    if x_0 == None:
        # this might be a terrible idea, but i'm not sure what else to do
        x,_ = np.zeros(n)
    else:
        x = x_0
    y = x
    t = 1.0

    M = sp.sparse.identity(m) - (1./L)*A.T*A
    d = (1./L)*A.T*b 
    for k in range(0,iter):
        log.write('fistaL iter. %d\n' % k)
        xold = x
        x = prox_g(M*y + d, 1./L)
        told = t
        t = (1 + sqrt(1+ 4*told**2))/2.0
        y = x + ((told - 1)/t)*(x - xold)
        r = A*x - b
        fval = 0.5*np.dot(r,r)
        log.write('fistaL fval %d\n' % fval)
    return x
