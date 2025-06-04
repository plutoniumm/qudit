import sys

sys.path.append("..")

import scipy.sparse as sp
from qudit import *
import numpy as np
import math as ma


def everything():
    D = Gategen(2)

    C = Circuit()
    L1 = C.layer(D.X, D.X, D.I, D.Z, D.Z)
    L2 = C.layer(D.H, D._, D.X, D.CX(1), D.H)
    L4 = C.barrier()
    L4 = C.layer(D.X, D.X, D.I, D.Z, D.Z)
    L4 = C.barrier()
    L3 = C.layer(D.I, D._, D.CX(3), D._, D.CX(1))

    print(C.draw())
    print("---" * 3)
    print(C.draw(output="penny"))

    # print(f"Solved to: {C.solve().shape} sparse matrix")
    print(C.solve())


if __name__ == "__main__":
    everything()
