"""
Process the epsilon-time data.
"""

import numpy as np
import matplotlib.pyplot as plt

data_c = np.loadtxt("./data/epsilon_time_c")
data_s = np.loadtxt("./data/epsilon_time_s")

epsilon_c = data_c[:, 0]
time_c = data_c[:, 1]
epsilon_s = data_s[:, 0]
time_s = data_s[:, 1]

epsilon_c_log = np.log10(epsilon_c)
epsilon_s_log = np.log10(epsilon_s)

c_f = np.polyfit(-epsilon_c_log, time_c, 1)
s_f = np.polyfit(-epsilon_s_log, time_s, 1)

plt.plot(-epsilon_c_log, time_c, "r.", ms=7, label="Circle Boundary")
plt.plot(-epsilon_c_log, c_f[0]*(-epsilon_c_log)+c_f[1], label="k="+str(c_f[0])[:4])
plt.plot(-epsilon_s_log, time_s, "kx", ms=6, label="Square Boundary")
plt.plot(-epsilon_s_log, s_f[0]*(-epsilon_s_log)+s_f[1], label="k="+str(s_f[0])[:4])
plt.xlabel(r"$-\log_{10} \epsilon$")
plt.ylabel("running time (s)")
plt.legend()
plt.savefig("./figs/ep_t.pdf", bbox_inches="tight")
plt.show()
