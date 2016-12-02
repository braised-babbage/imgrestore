import numpy as np
import scipy as sp
import proximal as px

from util import diff_oper
from fista import fistaL


def tv_denoise(v,mu=1.0,iter=200):
    """ Returns argmin 0.5||u-v||^2 + mu*||Du||_1 """
    n,m = v.shape
    v = v.flatten()
    Dx,Dy = diff_oper(n,m)
    D = sp.sparse.vstack([Dx,Dy])
    
    A = mu*D.T
    # solve p = argmin 0.5||mu*div p + v||^2 + I_K(p)
    p = fistaL(A,-v,px.prox_linfty_2_ball,mu*mu*8.0,
               x_0=np.zeros(n*m*2),iter=iter)
    u = v + A*p
    return u.reshape((n,m))


def proj_K(v,div,a0=1.0,a1=1.0,iter=200):
    """ Computes the projection onto the set K
    defined by constraints ||p||_\infty \leq a_0
    and ||div p||_\infty \leq a_1
    by solving the dual problem (pg. 26 of TGV paper) """
    
    def prox(x,s):
        return px.prox_mixed_l1(x,s,a0=a0,a1=a1)

    A = sp.sparse.hstack([sp.sparse.identity(len(v)), -div.T])
    L = 8.0  # Lipschitz constant for A.T A
    p = fistaL(A,v,prox,L)
    return v - A*p


def tgv_denoise(v,a0=1.0,a1=1.0,iter=200,tau=0.015625):
    """ computes argmin (1/2)||u-v||^2 + TGV_{a0,a1}(u)
    by first solving argmin (1/2)|| v - div^2 p||^2 + I_K(p)
    where K = \{ v : ||v||_{\infty} \leq a_0,
                      ||div v||_{\infty} \leq a_1\}"""
    n,m = v.shape
    v = v.flatten()

    Dxf,Dyf = diff_oper(n,m)
    Dxb = -Dxf.T
    Dyb = -Dyf.T
    div = sp.sparse.bmat([[Dxf,None, Dyf],[None,Dyf,Dxf]])
    div2 = sp.sparse.bmat([[Dxb*Dxf, Dyb*Dyf, Dyb*Dxf+Dxb*Dyf]])
    
    def prox(x,s):
        return proj_K(x,div,a0=a0,a1=a1)

    p = fistaL(div2,v,prox,1.0/64.0)
    return (v - div2*p).reshape(n,m)
