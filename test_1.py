"""
test.
"""

import numpy as np
import matplotlib.pyplot as plt
from rwpde import rwla2d

def fx0(L):
    "fx0"
    return np.ones(L)

def fxL(L):
    "fxL"
    return np.zeros(L)

def fy0(L):
    "fy0"
    return np.zeros(L)

def fyL(L):
    "fyL"
    return np.zeros(L)

def fi(L):
    "fi"
    return np.zeros((L-2, L-2))

if __name__ == '__main__':
    u = rwla2d(20, 500, 2318927)
    u.init_bv(fx0, fxL, fy0, fyL)
    u.init_q(fi)
    u.rw_all()

    # Plot
    U_T = np.transpose(u.U[1:u.L-1, 1:u.L-1])
    plt.imshow(U_T, origin="lower")
    plt.savefig("test.pdf", bbox_inches="tight")
    plt.close()
