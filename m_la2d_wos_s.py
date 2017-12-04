"""
The main process for la2d_wos_s.

Yucheng Zhang, yz4035@nyu.edu, 12/02/2017

Note: For a new problem, find the "problem dependent" parts and modify them.
"""

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from rwpde import la2d_wos_s

def main():
    "main method."

    # get the number of cores
    num_pro = mp.cpu_count()
    print("The number of cores:", num_pro)

    # generate seeds for different processes
    rng = np.random.RandomState(seed=13203179)
    seeds = rng.random_integers(0, high=2147483647, size=num_pro)

    # set the parameters, problem dependent
    L = 10 # the length of the square boundary
    Nr = 2000 # number of runs (estimates) for every point
    epsilon = 0.001 # thickness of the shell

    # the set of points to be evaluated
    Nl = 200

    # initialize objects for all processes, problem dependent
    us = [la2d_wos_s(L, Nr//num_pro, epsilon, Nl, seeds[i]) for i in range(num_pro)]

    # set up all processes
    processes = [mp.Process(target=us[i].rw_all, args=(i,)) for i in range(num_pro)]

    # get the beginning time
    t_s = time.time()

    # start all processes
    for p in processes:
        p.start()
    print("Processes start successfully!")

    # wait for all processes to end
    for p in processes:
        p.join()
    print("Processes end successfully!")

    # output the total running time of the program
    print("Takes", time.time()-t_s, "s.")

    # process the data, problem dependent
    U_ave = np.zeros((Nl-1)**2)
    U_r = []
    for i in range(num_pro):
        U_r.append(np.load("./data/"+"U_wos_s_"+str(L)+"_"+str(Nr//num_pro)+"_"+str(i)+".npz"))
    for i in range(num_pro):
        U_ave += U_r[i]["arr_2"] / num_pro
    X = U_r[0]["arr_0"]
    Y = U_r[0]["arr_1"]

    # plot, problem dependent
    plt.scatter(X, Y, c=U_ave, s=1, marker="s")
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.jet()
    plt.colorbar()
    plt.savefig("./figs/wos_s.pdf", bbox_inches="tight")
    plt.close()

if __name__ == '__main__': main()
