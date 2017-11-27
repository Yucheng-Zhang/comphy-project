"""
test. multiprocessing
"""

import time
import multiprocessing as mp
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
    t_s = time.time()
    "main process"
    num_pro = mp.cpu_count()
    # num_pro = 4
    print("You have", num_pro, "cores.")

    seeds = [13203179, 3274672, 2176387, 12381121, 4367845, 215376, 439583, 2137812]

    L = 100
    Nr = 5000
    us = [rwla2d(L, Nr//num_pro, seeds[i]) for i in range(num_pro)]
    for i in range(num_pro): # initialize
        us[i].init_bv(fx0, fxL, fy0, fyL)
        us[i].init_q(fi)

    processes = [mp.Process(target=us[i].rw_all, args=(i,)) for i in range(num_pro)]

    for p in processes:
        p.start()
    print("Processes start successfully!")

    for p in processes:
        p.join()
    print("Processes end successfully!")

    # Process the data
    U_ave = np.zeros((L, L))
    U_r = []
    for i in range(num_pro):
        U_r.append(np.load("U_"+str(L)+"_"+str(Nr//num_pro)+"_"+str(i)+".npy"))
    for i in range(num_pro):
        U_ave += U_r[i] / num_pro

    # Plot
    U_T = np.transpose(U_ave[1:L-1, 1:L-1])
    plt.imshow(U_T, origin="lower")
    plt.savefig("test.pdf", bbox_inches="tight")
    plt.close()

    print("Takes", time.time()-t_s, "s.")
