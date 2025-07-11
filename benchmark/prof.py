import sys

sys.path.append("..")

from qudit.circuit import Circuit
import numpy as np

n = 11
C = Circuit(n, dim=4)
G = C.gates
C.gate(G.H, dits=[0])
for i in range(n - 1):
    C.gate(G.CX, dits=[i, i + 1])

_ = C.solve()
