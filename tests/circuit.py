import sys

sys.path.append("..")


# from sympy import exp, SparseMatrix, Symbol
from unittest import TestCase, main
from qudit import Circuit
import numpy as np



def mirror(n):
    C1, C2 = Circuit(2, dim=n), Circuit(2, dim=n)
    G = C1.gates

    C1.gate(G.H, dits=[0])
    C2.gate(G.H, dits=[1])
    SWAP = G.SWAP

    C1 = C1.solve()
    C2 = C2.solve()

    C1 = SWAP @ C1 @ SWAP.T

    diff = np.sum(np.abs(C1 - C2))
    print(diff)

mirror(2)

class Circuits(TestCase):
    def test_bell(self):
        HCX = np.array(
            [[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]]
        ) / np.sqrt(2)

        C = Circuit(2, dim=2)
        G = C.gates
        C.gate(G.H, dits=[0])
        C.gate(G.CX, dits=[0, 1])

        U = C.solve().todense()
        self.assertTrue(np.allclose(U, HCX, atol=1e-4))


# if __name__ == "__main__":
#     main()
