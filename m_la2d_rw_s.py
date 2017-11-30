"""
The main process for la2d_rw_s.
"""

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from rwpde import la2d_rw_s

def fx0(l):
    "fx0"
    return np.ones(l)

def fxL(l):
    "fxL"
    return np.zeros(l)

def fy0(l):
    "fy0"
    return np.zeros(l)

def fyL(l):
    "fyL"
    return np.zeros(l)

def fi(l):
    "fi"
    return np.zeros((l-2, l-2))

if __name__ == '__main__':
    t_s = time.time()

    # num_pro = 1
    num_pro = mp.cpu_count() # Number of cores
    print("The number of cores:", num_pro)

    seeds = [13203179, 3274672, 2176387, 12381121, 4367845, 215376, 439583, 2137812]

    L = 10
    Nr = 1000
    us = [la2d_rw_s(L, Nr//num_pro, seeds[i]) for i in range(num_pro)]
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
        U_r.append(np.load("./data/"+"U_"+str(L)+"_"+str(Nr//num_pro)+"_"+str(i)+".npy"))
    for i in range(num_pro):
        U_ave += U_r[i] / num_pro

    # Plot
    U_T = np.transpose(U_ave[1:L-1, 1:L-1])
    plt.imshow(U_T, origin="lower")
    plt.savefig("test.pdf", bbox_inches="tight")
    plt.close()

    print("Takes", time.time()-t_s, "s.")
