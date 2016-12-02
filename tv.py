import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import proximal as px
from util import diff_oper
from chambolle_pock import cp_saddle, cp_saddle_uniform


def tv_denoise(f, mu, iter=100):
    """ Solves min_x G(x) + mu*F(Dx), where
        G(x) = (1/2)||x-f||^2 and F(x) = ||x||_1,
    via the saddle point problem
        min_x max_y <Dx,y> + (1/mu)*G(x) - Fstar(y) """
    n,m = f.shape
    f = f.flatten()
    Dx,Dy = diff_oper(n,m)
    K = sp.sparse.vstack([Dx,Dy])
    prox_Fstar = px.prox_l1star
    prox_G = px.make_scaled_prox(px.make_dist2_prox(f), 1./mu)
    tau = 0.25
    sigma = 0.25
    theta = 1.0
    y0 = np.zeros(2*n*m)
    x,y = cp_saddle(K, prox_Fstar, prox_G, iter,
                    tau, sigma, theta, f, y0)
    return x.reshape(n,m)


def tv_denoise_fast(f,mu,iter=100):
    n,m = f.shape
    f = f.flatten()
    Dx,Dy = diff_oper(n,m)
    K = sp.sparse.vstack([Dx,Dy])
    prox_Fstar = px.prox_l1star
    prox_G = px.make_scaled_prox(px.make_dist2_prox(f), 1./mu)
    tau = 0.25
    sigma = 0.25
    theta = 1.0
    y0 = np.zeros(2*n*m)
    # x,y = cp_saddle(K, prox_Fstar, prox_G, iter, tau, sigma, theta, f, y0)
    x,y = cp_saddle_uniform(K, prox_Fstar, prox_G, 1./mu, iter,
                            tau, sigma, theta, f, y0)
    return x.reshape(n,m)


def tv_demo(I, std, mu, iter):
    n,m = I.shape
    Inoisy = I + std*np.random.randn(n,m)
    Irec = tv_denoise(Inoisy, mu, iter)
    plt.subplot(1,3,1)
    plt.imshow(I,cmap=plt.cm.gray)
    plt.subplot(1,3,2)
    plt.imshow(Inoisy,cmap=plt.cm.gray)
    plt.subplot(1,3,3)
    plt.imshow(Irec,cmap=plt.cm.gray)
    return Inoisy,Irec
