from scipy.linalg import fractional_matrix_power as fmp
from ..circuit import gates
import torch as pt

C64 = pt.complex64


class QSVT:
    def __init__(self, phis, A, wires, dev):
        self.size = A.shape[0]
        self.wires = wires
        self.dev = dev

        if not isinstance(A, pt.Tensor):
            A = pt.from_numpy(A).to(dtype=C64, device=dev)
        self.A = A

        if not isinstance(phis, pt.Tensor):
            phis = pt.tensor(phis, dtype=C64, device=dev)
        self.phis = phis

        self.U = self.block(A, wires, dev)

    def PCP(self, phi):
        ex = pt.exp(1j * pt.tensor(phi))
        arr = pt.diag(
            pt.cat([pt.full((self.size,), ex), pt.full((self.size,), ex.conj())])
        ).to(dtype=C64, device=self.dev)

        return arr

    def block(self, A, wires, dev):
        r = lambda x: pt.from_numpy(fmp(x, 0.5)).to(dtype=C64)

        I = pt.eye(self.size, dtype=C64)

        # top-right and bottom-left
        TR = r(I - A.conj().T @ A)
        BL = r(I - A @ A.conj().T)

        T = pt.cat([A, TR], dim=1)
        B = pt.cat([BL, -A.conj().T], dim=1)
        U_A = pt.cat([T, B], dim=0).contiguous().to(device=dev, dtype=C64)

        return gates.U(matrix=U_A, dim=self.size, wires=wires, index=[0, 1])
