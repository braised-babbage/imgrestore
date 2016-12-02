import numpy as np


def prox_l1(v,s):
    """Returns argmin_u 0.5||u-v||^2 + s*||u||_1"""
    return np.sign(v)*np.maximum(np.abs(v) - s,0)


def prox_l1_2(v,s):
    l = len(v)/2
    p = (v[:l]**2 + v[l:]**2)**(0.5)
    mx = np.maximum(p - s,0)
    return np.append(mx*v[:l]/p,mx*v[l:]/p)


def make_star_prox(prox):
    """ We use Moreau's identity """
    """     v = prox_F(v,s) + s*prox_Fstar(v/s,1/s)"""
    """ remark: in code, double dual """
    return lambda v,s: v - s*prox(v/s,1./s)


def prox_l1star(v,s):
    v[v > 1.0] = 1.0
    v[v < -1.0] = -1.0
    return v


def make_scaled_prox(prox,mu):
    """ given prox(v,s) = argmin_x ||x-v||/(2s) + F(x) """
    """ returns argmin_x ||x-v||/(2s) + mu*F(x)"""
    return lambda v,s: prox(v,s*mu)


def make_dist2_prox(f):
    """ Creates proximal operator for the function """
    """   G(x) = (1/2)||x-f||^2 """
    return lambda v,s: (f + (v / s)) / (1+1./s)
