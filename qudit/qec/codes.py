from torch import linalg as LA
from ..circuit import Gategen
import scipy.linalg as SLA
import torch as pt
import numpy as np
import math as ma

C64 = pt.complex64

Pauli = {
    "I": pt.tensor([[1, 0], [0, 1]], dtype=C64),
    "X": pt.tensor([[0, 1], [1, 0]], dtype=C64),
    "Y": pt.tensor([[0, -1j], [1j, 0]], dtype=C64),
    "Z": pt.tensor([[1, 0], [0, -1]], dtype=C64),
}


def composite(stabilizer: list[str], ops: dict = None) -> pt.Tensor:
    if ops is None:
        ops = Pauli
    mat = ops[stabilizer[0]]
    for p in stabilizer[1:]:
        mat = pt.kron(mat, ops[p])

    return mat


def Projector(stabilizers: list[pt.Tensor]) -> pt.Tensor:
    dim = stabilizers[0].shape[0]
    I = pt.eye(dim, dtype=C64)
    P = I
    for S in stabilizers:
        P = P @ ((I + S) * 0.5)

    return P


def SVD_RRF(P: pt.Tensor, target_rank=2) -> pt.Tensor:
    Omega = pt.randn((P.shape[0], target_rank), dtype=pt.float32).to(pt.complex64)
    Y = P @ Omega
    Q = LA.qr(Y, mode="reduced")[0]

    return Q.T  # rows=vectors


def SVD_RRF_np(P, target_rank=2):
    P = P.cpu().numpy()
    Omega = np.random.normal(size=(P.shape[0], target_rank))
    Y = P @ Omega
    Q = SLA.qr(Y, mode="economic", overwrite_a=True, check_finite=False)[0]

    return pt.from_numpy(Q.T)  # rows=vectors


class Code:
    """
    A codeword is an $N \\times d$ tensor d is the dimension of the code, and N is the number computational base states, where $N = d^n$ in a system of $n$ qudits.

    For now Code only supports uniform dimension codes. So a dim of means only codewords of the form $2^n \\times 2$ will be supported. This is sufficient for many standard codes, but may need to be relaxed in the future.
    """

    codewords: pt.Tensor
    gates: Gategen
    dim: int
    dits: int
    width: int

    def __init__(self, codewords: pt.Tensor, d: int = None):
        assert codewords.ndim >= 2, "Codewords must be at least 2D tensors"
        self.codewords = codewords
        self.dim = codewords.shape[0]
        width = codewords.shape[1]
        _d = d if d is not None else self.dim
        self.dits = int(round(ma.log(width) / ma.log(_d)))
        self.gates = Gategen(_d)

    def toTensor(self):
        return self.codewords

    @staticmethod
    def isValid(code):
        norms = pt.norm(code, dim=1)
        if not pt.allclose(norms, pt.ones_like(norms)):
            print("Norms:", pt.round(norms, decimals=2))
            raise ValueError("Codewords are not normalized!")
        ortho = code @ code.T - pt.eye(code.shape[0])
        if not pt.allclose(ortho, pt.zeros_like(ortho), atol=1e-3):
            print("Orthogonality check:", pt.round(ortho, decimals=2))
            raise ValueError("Codewords are not orthogonal!")

    def __getitem__(self, idx):
        return self.codewords[idx]

    def __len__(self):
        return self.codewords.shape[0]

    @staticmethod
    def fromStabilizers(
        stabilizers: list[str] | list[list[str]], d: int = 2, method="svd", numpy=False
    ) -> "Code":
        """
        We may get `["XXY", "ZZI"]`, or `[["X", "X", "Y"], ["Z", "Z", "I"]]`. In the first case we need to split the strings into lists of characters, in the second case we can directly use the lists.

        We then use the constructed matrices to construct the projector onto the code space, and then extract the codewords using either `method="svd"` (Singular Value Decomposition, default) or `method="rrf"`(Randomized Range Finder).

        SVD is more stable but may be slower for large codes, while RRF is faster but may be less stable. For small codes, SVD is usually preferred.

        The `d` parameter sets the local dimension (default 2 for qubits). For d>2, `X` and `Z` are the generalized Weyl operators from `Gategen`.
        """
        assert method in ["svd", "rrf"]
        if isinstance(stabilizers[0], str):
            stabilizers = [list(s) for s in stabilizers]
        gg = Gategen(dim=d)
        ops = {
            "I": pt.eye(d, dtype=C64),
            "X": gg.X.tensor.to(C64),
            "Z": gg.Z.tensor.to(C64),
        }

        proj = Projector([composite(s, ops) for s in stabilizers])
        if method == "svd":
            _, S, Vh = LA.svd(proj)
            code = Vh[pt.isclose(S, pt.tensor(1.0))]
        else:  # rrf
            if numpy:
                code = SVD_RRF_np(proj)
            else:
                code = SVD_RRF(proj)
        Code.isValid(code)

        return Code(code, d=d)
