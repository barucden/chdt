import sys
import random

import torch as t
import numpy as np

import chdt
from scipy.ndimage import distance_transform_cdt

def scipy_dist(Ys):
    Ys = Ys > 0
    D = [distance_transform_cdt(Y, metric='chessboard') for Y in Ys]
    D = np.stack(D)
    D[D == -1] = chdt.INF
    return D

def check_simple():
    Y = -t.ones((1, 1, 7, 7))
    Y[:, :, 1:-1, 1:-1] = 1
    D = t.tensor([[[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]]]])

    D_ = chdt.transform(Y)

    assert (D == D_).all()

    print("Simple problem is OK!")

def random_problem():
    batchsize = random.randint(1, 15)
    nrows = random.randint(1, 15)
    ncols = random.randint(1, 15)

    Y = np.random.rand(batchsize, 1, nrows, ncols)
    Y = np.sign(Y).astype('float32')

    return Y

def check_random(niters=1):
    for i in range(1, niters + 1):
        Y = random_problem()
        D = scipy_dist(Y)

        D_ = chdt.transform(t.from_numpy(Y)).numpy()

        assert (D == D_).all(), Y

        if i % 1000 == 0:
            print("Checked", i, "problems...")

    print("All", niters, "random problems were OK!")

niters = int(sys.argv[1]) if len(sys.argv) > 1 else 1

check_simple()
check_random(niters)

