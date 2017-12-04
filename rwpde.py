"""
The definition of all classes.

Yucheng Zhang, yz4035@nyu.edu, 12/04/2017

Random Number Generator: numpy.random

- la2d_rw_s: 2D Laplace (Poisson) Equation, Square Boundray, Random Walk on square lattice grid

- la2d_wos_s: 2D Laplace Equation, Square Boundary, Walk on Sphere algorithm

- la2d_wos_c: 2D Laplace Equation, Circle Boundary, Walk on Sphere algorithm

Note: Given a problem, the things to consider include dimension, the shape of the boundary, the value at the boundary etc. These details affect the code in many places, so even though the algorithm is the similar, it's hard to write a general code for all problems. But it's easy to modify the codes for a new practical problem. The classes defined here are some examples, which are used in my report. For a new problem, find the "problem dependent" parts and modify them.
"""

import numpy as np

###########################################################################

class la2d_rw_s:
    "2D Laplace Equation, Square boundary, Square lattice grid."

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
        fn = "./data/srw/"+"U_"+str(self.L)+"_"+str(self.Nr)+"_"+str(i)
        np.save(fn, self.U)

###########################################################################

class la2d_wos_s:
    "2D Laplace Equation, Square Boundary, Walk on Sphere."

    epsilon = 0.5 # thickness of the shell
    U = None # Store the result values
    X = None
    Y = None
    rng = None # Random number generator

    L = 10 # the length of the square boundary
    Nr = 100 # Number of random walks for each point
    Nl = 100 # number of points to be evaluated per line
    count = 0

    def __init__(self, L, Nr, epsilon, Nl, seed):
        self.L = L
        self.Nr = Nr
        self.epsilon = epsilon
        self.U = np.zeros((L, L))
        self.rng = np.random.RandomState(seed=seed)
        self.Nl = Nl

        Nops = (self.Nl-1)**2
        self.X = np.zeros(Nops)
        self.Y = np.zeros(Nops)
        self.U = np.zeros(Nops)
        self.count = 0

    def arrive_b(self, r):
        "Judge whether the rw reaches the boundary.\
        This is obviously boundary dependant."
        return bool(r < self.epsilon)

    def update_u(self, r, x, y):
        "Get the value & update U when the rw reaches the boundary.\
        This is obviously boundary dependent."
        if r == self.L-y:
            self.U[self.count] += 1.0 / self.Nr
        else:
            self.U[self.count] += 0

    def rw_at(self, xo, yo):
        "Evaluate point (xo, yo)."
        for _ in range(self.Nr):
            x, y = xo, yo
            while True:
                r = min(x, y, self.L-x, self.L-y)
                if self.arrive_b(r):
                    self.update_u(r, x, y)
                    break
                theta = 2 * np.pi * self.rng.uniform()
                x += r * np.cos(theta)
                y += r * np.sin(theta)

    def rw_all(self, k):
        "Evaluate all points."
        for i in range(1, self.Nl):
            print(i)
            xo = i / self.Nl * self.L
            for j in range(1, self.Nl):
                yo = j / self.Nl * self.L
                self.X[self.count] = xo
                self.Y[self.count] = yo
                self.rw_at(xo, yo)
                self.count += 1
        self.save_data(k)

    def save_data(self, k):
        "Save the data."
        fn = "./data/"+"U_wos_s_"+str(self.L)+"_"+str(self.Nr)+"_"+str(k)
        np.savez(fn, self.X, self.Y, self.U)

###########################################################################

class la2d_wos_c:
    "2D Laplace Eqaution, Circle Boundary, Walk on Sphere."

    epsilon = 0.5 # thickness of the shell
    U = None # Store the result values, together with X, Y
    X = None
    Y = None
    rng = None # Random number generator

    R = 10 # Radius of the circle
    Nr = 100 # Number of random walks for each point
    Nro = 100 # Number of ro to be evaluated
    Nphio = 10 # Number of phio to be evaluated on the first ro
    count = 0

    def __init__(self, R, Nr, epsilon, Nro, Nphio, seed):
        self.R = R
        self.Nr = Nr
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed=seed)

        self.Nro = Nro
        self.Nphio = Nphio

        Nops = Nphio * Nro * (Nro - 1) // 2
        self.X = np.zeros(Nops)
        self.Y = np.zeros(Nops)
        self.U = np.zeros(Nops)

        self.count = 0

    def arrive_b(self, r):
        "Judge whether the rw reaches the boundary."
        return bool(self.R - r < self.epsilon)

    def update_u(self, c_phi, s_phi):
        "Get the value & update U when the rw reaches the boundary."
        # The boundary value is set here.
        self.U[self.count] += 2*c_phi*s_phi / self.Nr
        # self.U[self.count] += np.sign(c_phi) / self.Nr

    def rw_at(self, xo, yo):
        "Evaluate point (xo, yo)."
        for _ in range(self.Nr):
            x, y = xo, yo
            while True:
                r = np.sqrt(x**2 + y**2)
                s_phi = y / r
                c_phi = x / r
                if self.arrive_b(r):
                    self.update_u(c_phi, s_phi)
                    break
                theta = 2 * np.pi * self.rng.uniform()
                x += (self.R - r) * np.cos(theta)
                y += (self.R - r) * np.sin(theta)

    def rw_all(self, k):
        "Evaluate all points."
        for i in range(1, self.Nro):
            print(i)
            ro = i / self.Nro * self.R
            Nphi = i * self.Nphio
            for j in range(Nphi):
                phio = j / Nphi * 2 * np.pi
                xo = ro * np.cos(phio)
                yo = ro * np.sin(phio)
                self.X[self.count] = xo
                self.Y[self.count] = yo
                self.rw_at(xo, yo)
                self.count += 1
        self.save_data(k)

    def save_data(self, k):
        "Save the data."
        fn = "./data/"+"U_wos_c_"+str(self.Nro)+"_"+str(self.Nphio)+"_"+str(self.Nr)+"_"+str(k)
        np.savez(fn, self.X, self.Y, self.U)

###########################################################################
