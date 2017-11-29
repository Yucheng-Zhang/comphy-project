"""
test. multiprocessing
"""

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from rwpde import rwla2d_sp_c

if __name__ == '__main__':
    t_s = time.time()
    "main process"
    num_pro = mp.cpu_count() # Number of cores
    # num_pro = 1
    print("You have", num_pro, "cores.")

    seeds = [13203179, 3274672, 2176387, 12381121, 4367845, 215376, 439583, 2137812]

    R = 10
    Nr = 1000
    Nro = 100
    Nphio = 5
    # us = [rwla2d_sp_c(R, Nr//num_pro, 0.1, Nro, Nphio, seeds[i]) for i in range(num_pro)]

    # processes = [mp.Process(target=us[i].rw_all, args=(i,)) for i in range(num_pro)]

    # for p in processes:
    #     p.start()
    # print("Processes start successfully!")

    # for p in processes:
    #     p.join()
    # print("Processes end successfully!")

    # Process the data
    U_ave = np.zeros(Nphio * Nro * (Nro - 1) // 2)
    U_r = []
    for i in range(num_pro):
        U_r.append(np.load("./data/"+"U_sp_c_"+str(Nro)+str(Nphio)+"_"+str(Nr//num_pro)+"_"+str(i)+".npz"))
    for i in range(num_pro):
        U_ave += U_r[i]["arr_2"] / num_pro
    X = U_r[0]["arr_0"]
    Y = U_r[0]["arr_1"]

    # Plot
    plt.scatter(X, Y, c=U_ave, s=0.3)
    plt.colorbar()
    plt.savefig("test_sp_c.pdf", bbox_inches="tight")
    plt.close()

    print("Takes", time.time()-t_s, "s.")
