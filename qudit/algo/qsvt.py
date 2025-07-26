from scipy.linalg import fractional_matrix_power as fmp
from ..circuit import gates
import numpy as np
import torch

Cplx = torch.complex64


class QSVT:
    def __init__(self, phis, A, wires, dev):
        self.size = A.shape[0]
        self.wires = wires
        self.phis = phis
        self.dev = dev
        self.A = A

        self.U = self.block(A, wires, dev)

    def PCP(self, phi):
        ex = np.exp(1j * phi)
        arr = np.diag(
            np.concatenate([np.full(self.size, ex), np.full(self.size, ex.conj())])
        ).astype(np.complex64)

        return torch.from_numpy(arr).to(device=self.dev, dtype=Cplx)

    def block(self, A, wires, dev):
        r = lambda x: fmp(x, 0.5).astype(np.complex64)

        I = np.eye(self.size, dtype=np.complex64)

        U_A = np.block(
            [[A, r(I - A.conj().T @ A)], [r(I - A @ A.conj().T), -A.conj().T]]
        ).astype(np.complex64)
        U_A = torch.from_numpy(np.ascontiguousarray(U_A)).to(device=dev, dtype=Cplx)

        return gates.U(matrix=U_A, dim=self.size, wires=wires, index=[0, 1])
