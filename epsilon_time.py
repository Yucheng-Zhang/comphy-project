"""
Process the epsilon-time data.
"""

import numpy as np
import matplotlib.pyplot as plt

data_c = np.loadtxt("./data/epsilon_time_c")

epsilon_c = data_c[:, 0]
time_c = data_c[:, 1]

plt.plot(1.0/epsilon_c, time_c, "k.")
plt.show()
plt.plot(np.log(1.0/epsilon_c), time_c, "k.")
plt.show()
