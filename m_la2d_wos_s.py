"""
The main process for la2d_wos_s.
"""

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from rwpde import la2d_wos_s

if __name__ == '__main__':
    t_s = time.time()

    # num_pro = 1
    num_pro = mp.cpu_count() # Number of cores
    print("The number of cores:", num_pro)

    
    seeds = [13203179, 3274672, 2176387, 12381121, 4367845, 215376, 439583, 2137812] # The seeds for different processes

    L = 100 # 
    Nr = 10000
    us = [la2d_wos_s(L, Nr//num_pro, 0.1, seeds[i]) for i in range(num_pro)]

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
        U_r.append(np.load("./data/"+"U_sp_"+str(L)+"_"+str(Nr//num_pro)+"_"+str(i)+".npy"))
    for i in range(num_pro):
        U_ave += U_r[i] / num_pro

    # Plot
    U_T = np.transpose(U_ave[1:L, 1:L])
    plt.imshow(U_T, origin="lower")
    plt.savefig("test_sp.pdf", bbox_inches="tight")
    plt.close()

    print("Takes", time.time()-t_s, "s.")
