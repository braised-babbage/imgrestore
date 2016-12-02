import numpy as np
import scipy as sp
import proximal as px

from os import stdout
from scipy.linalg import norm
from scipy.sparse.linalg import cgs
from util import diff_oper
from math import sqrt


def sb_itv(g,mu,tol=1e-3,cgs_tol=1e-5,log=stdout):
    """Split Bregman Isotropic Total Variation Denoising
    u = arg min_u 1/2||u-g||_2^2 + mu*ITV(u)"""
    n,m = g.shape
    l = n*m
    Dx,Dy = diff_oper(n,m)
    g = g.flatten()
    B = sp.sparse.bmat([[Dx],[Dy]])

    Bt = B.T
    BtB = Bt*B
    
    b = np.zeros(2*l)
    z = np.zeros(l)
    d,u = b,g
    err,k = 1,1
    lam = 1.0

    M = sp.sparse.identity(l) + BtB

    while err > tol:
        log.write("tv it. %d\n" % k)
        up = u
        # optimize this
        u,_ = cgs(M, g-lam*Bt*(b-d), tol=cgs_tol, maxiter=100)
        Bub = B*u+b

        d = px.prox_l1_2(Bub,mu/lam)
        b = Bub-d
        err = norm(up-u)/norm(u)
        log.write("err=%f\n" % err)
        k = k+1

    log.write("Stopped because norm(up-u)/norm(u) <= tol=%f\n" % tol)
    z = (B*u - d)[:l].reshape((n,m))
    #u = u.reshape((n,m))
    return z


def fast_admm_tv(f,mu,tol=1e-3, cgs_tol=1e-5,log=stdout):
    """Fast ADMM Isotropic Total Variation Denoising
       u = arg min_u 1/2||u-f||_2^2 + mu*ITV(u) """
    n,m = f.shape
    l = n*m
    Dx,Dy = diff_oper(n,m)
    f = f.flatten()
    D = sp.sparse.bmat([[Dx],[Dy]])

    Dt = D.T
    DtD = Dt*D
    
    v = np.zeros(2*l)
    vhat = v

    z = np.zeros(2*l)
    zhat = z

    u = f
    err,k = 1,1
    tau = 1.0

    eta = 0.999
    c = 1000000000.0  # some big number

    a = 1.0

    M = sp.sparse.identity(l) + tau*DtD

    while c > tol:
        log.write("tv it. %d\n" % k)

        # optimize this
        u,_ = cgs(M, f+Dt*(zhat + tau*vhat), tol=cgs_tol, maxiter=100)
        #        u,_ = cgs(M,f+Dt*(z + tau*v),tol=cgs_tol,maxiter=100)
        vp = v
        v = px.prox_l1(D*u - zhat/tau, mu/tau)
        #v = prox_l1_2(D*u - z/tau, mu/tau)
        zp = z
        z = zhat - tau*(D*u - v)
        #z = z - tau*(D*u - v)

        cp = c
        c = (1/tau)*np.dot(z-zhat,z-zhat) + tau*np.dot(v-vhat,v-vhat)
        #c = norm(u-up)/norm(u)
        
        ap = a
        if c < eta*cp:
            a = (1+sqrt(1+4*ap*ap))/2.0
            vhat = v + ((ap-1)/a)*(v - vp)
            zhat = z + ((ap-1)/a)*(z - zp)
        else:
            log.write("admm restart\n")
            a = 1
            vhat = vp
            zhat = zp
            c = cp / eta
        log.write("c = %f\n" % c)
        k = k+1

    log.write("Stopped because c  <= tol=%f\n" % tol)
    u = u.reshape((n,m))
    return u
