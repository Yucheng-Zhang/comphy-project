"""
Use random walk to solve PDEs with BVPs.

Random number generator: numpy.random

1. Laplace Equation in 2D, with square lattice and boundray conditon.

2, Walk on Sphere algorithm for Laplace Equation on 2D, lattice free.
"""

import numpy as np

class rwla2d:
    "Random walk & Laplace (Poisson) Equation & 2D & Square lattice."

    L = 10 # Linear size of the square lattice
    Nr = 100 # Number of random walks for each point
    V = None # Value at the boudary & interior(for possion equation)
    U = None # Store the result values
    rng = None # Random number generator

    def __init__(self, L, Nr, seed):
        self.L = L
        self.Nr = Nr
        self.V = np.zeros((L, L))
        self.U = np.zeros((L, L))
        self.rng = np.random.RandomState(seed=seed)

    def init_bv(self, fx0, fxL, fy0, fyL):
        "Initialize the boudary condition."
        self.V[0, :] = fx0(self.L)
        self.V[self.L-1, :] = fxL(self.L)
        self.V[:, 0] = fy0(self.L)
        self.V[:, self.L-1] = fyL(self.L)
        self.U = self.V

    def init_q(self, fi):
        "Initialize the interior "
        self.V[1:self.L-1, 1:self.L-1] = fi(self.L)

    def arrive_b(self, x, y):
        "Judge whether (x,y) is on the boundary."
        return bool(x == 0 or x == self.L-1 or y == 0 or y == self.L-1)

    def rw_at(self, xo, yo):
        "Evaluate point (xo, yo)."
        # print("ZYC")
        for _ in range(self.Nr):
            rx, ry = xo, yo # Initial position of the rw
            while True:
                dr = self.rng.random_integers(0, 3)
                if dr == 0:
                    rx += 1
                elif dr == 1:
                    rx -= 1
                elif dr == 2:
                    ry += 1
                else:
                    ry -= 1
                if self.arrive_b(rx, ry): # arrive at the boundary
                    self.U[xo, yo] += self.V[rx, ry] / self.Nr
                    break

    def rw_all(self, i):
        "Evaluate all points."
        for xo in range(1, self.L-1):
            print(xo)
            for yo in range(1, self.L-1):
                self.rw_at(xo, yo)
        self.save_data(i)

    def save_data(self, i):
        "Save the data."
        fn = "./data/"+"U_"+str(self.L)+"_"+str(self.Nr)+"_"+str(i)
        np.save(fn, self.U)

class rwla2d_sp:
    "Random walk on Spheres Algorithm for Laplace Eqaution, 2D.\
    Here applied to the square boundary."

    epsilon = 0.5 # thickness of the shell
    U = None # Store the result values
    rng = None # Random number generator

    L = 10 # Linear length of the sqaure
    Nr = 100 # Number of random walks for each point

    def __init__(self, L, Nr, epsilon, seed):
        self.L = L
        self.Nr = Nr
        self.epsilon = epsilon
        self.U = np.zeros((L, L))
        self.rng = np.random.RandomState(seed=seed)

    def arrive_b(self, r):
        "Judge whether the rw reaches the boundary.\
        This is obviously boundary dependant."
        return bool(r < self.epsilon)

    def update_u(self, xo, yo, r, x, y):
        "Get the value & update U when the rw reaches the boundary.\
        This is obviously boundary dependent."
        if r == x:
            self.U[xo, yo] += 1.0 / self.Nr
        else:
            self.U[xo, yo] += 0

    def rw_at(self, xo, yo):
        "Evaluate point (xo, yo)."
        for _ in range(self.Nr):
            x, y = xo, yo
            while True:
                r = min(x, y, self.L-x, self.L-y)
                if self.arrive_b(r):
                    self.update_u(xo, yo, r, x, y)
                    break
                theta = 2 * np.pi * self.rng.uniform()
                x += r * np.cos(theta)
                y += r * np.sin(theta)

    def rw_all(self, i):
        "Evaluate all points."
        for xo in range(1, self.L):
            print(xo)
            for yo in range(1, self.L):
                self.rw_at(xo, yo)
        self.save_data(i)

    def save_data(self, i):
        "Save the data."
        fn = "./data/"+"U_sp_"+str(self.L)+"_"+str(self.Nr)+"_"+str(i)
        np.save(fn, self.U)
