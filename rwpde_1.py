"""
Use random walk to solve PDEs with BVPs.

Random number generator: numpy.random

1. Laplace Equation in 2D, with square lattice and boundray conditon.
"""

import numpy as np

class rwla2d:
    "Random walk & Laplace (Poisson) Equation & 2D & Square lattice."

    L = 10 # Linear size of the square lattice
    Nr = 100 # Number of random walks for each point
    V = None # Value at the boudary & interior(for possion equation)
    U = None # Store the result value
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

    def rw_all(self):
        "Evaluate all points."
        for xo in range(1, self.L-1):
            for yo in range(1, self.L-1):
                self.rw_at(xo, yo)

    def save_data(self):
        "Save the data."
