"""
A simple convergence test.
"""

import numpy as np
import matplotlib.pyplot as plt
from rwpde import la2d_wos_s

Nr_set = np.array([10000 * i for i in range(1, 30, 2)])

u_set = []

for Nr in Nr_set:
    print(Nr)
    u = la2d_wos_s(10, Nr, 1e-11, 2, 13203179)
    u.rw_at(5.0, 5.0)
    u_set.append(u.U[0])

u_set = np.array(u_set)

np.savez("./data/conv", Nr_set, u_set)

data = np.load("./data/conv.npz")
Nr_set = data["arr_0"]
u_set = data["arr_1"]

err = np.abs(u_set-0.25)

plt.plot(Nr_set, err, ".")
plt.show()

Nr_log = np.log(Nr_set)
err_log = np.log(err)
plt.plot(Nr_log, err_log, ".")
plt.show()
