from PIL import Image
import numpy as np
import scipy as sp
import scipy.sparse


def load_image(filename):
    """ Load an image (converted to grayscale) as numpy array. """
    img = Image.open(filename).convert("L")
    img.load()
    data = np.asarray(img, dtype=np.float32)
    return data


def save_image(npdata,filename):
    """ Save a grayscale image (numpy array). """
    img = Image.fromarray(np.asarray(np.clip(npdata,0,255),
                                     dtype="uint8"),"L")
    img.save(filename)


def diff_oper(n,m):
    """ Constructs finite difference operators for n x m images
    We assume row-major order (numpy default) """
    D = sp.sparse.spdiags([-(np.append(np.ones(m-1),np.zeros(1))), np.ones(m)],[0,1],m,m)
    Dx = sp.sparse.kron(sp.sparse.identity(n),D)
    D = sp.sparse.spdiags([-(np.append(np.zeros(1),np.ones(n-1))), np.ones(n)],[0,-1],n,n)
    Dy = sp.sparse.kron(D,sp.sparse.identity(m))
    return Dx,Dy
