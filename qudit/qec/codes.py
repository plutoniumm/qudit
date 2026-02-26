from torch import linalg as LA
from ..circuit import Gategen
import torch as pt
import math as ma

C64 = pt.complex64

Pauli = {
    "I": pt.tensor([[1, 0], [0, 1]], dtype=C64),
    "X": pt.tensor([[0, 1], [1, 0]], dtype=C64),
    "Y": pt.tensor([[0, -1j], [1j, 0]], dtype=C64),
    "Z": pt.tensor([[1, 0], [0, -1]], dtype=C64),
}


def composite(stabilizer: list[str]) -> pt.Tensor:
    mat = Pauli[stabilizer[0]]
    for p in stabilizer[1:]:
        mat = pt.kron(mat, Pauli[p])

    return mat


def Projector(stabilizers: list[pt.Tensor]) -> pt.Tensor:
    dim = stabilizers[0].shape[0]
    I = pt.eye(dim, dtype=C64)
    P = I

    for S in stabilizers:
        P = P @ ((I + S) * 0.5)

    return P


def SVD_RRF(P: pt.Tensor, target_rank=2) -> pt.Tensor:
    # 1. Gaussian Random Matrix Omega (N x rank)
    Omega = pt.randn((P.shape[0], target_rank), dtype=pt.float32).to(pt.complex64)

    # 2. Sample the Range (Y = P @ Omega)
    # projects random vectors into codespace.
    Y = P @ Omega

    # 3. Orthonormalize Y using QR decomposition
    # Q(N, target_rank), => orthonormal basis 4 for colspace of Y
    Q = LA.qr(Y, mode="reduced")[0]

    return Q.T  # rows=vectors


def SVD_RRF_np(P, target_rank=2):
    import scipy.linalg as SLA
    import numpy as np

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

    def __init__(self, codewords: pt.Tensor):
        assert codewords.ndim >= 2, "Codewords must be at least 2D tensors"
        self.codewords = codewords

        # check that we are in a d^n, d system
        self.dim = codewords.shape[0]
        width = codewords.shape[1]

        n = int(ma.log(width) / ma.log(self.dim))
        self.dits = int(n)
        self.gates = Gategen(self.dim)

    def toTensor(self):
        return self.codewords

    @staticmethod
    def isValid(code):
        import numpy as np

        norms = pt.norm(code, dim=1)
        if not pt.allclose(norms, pt.ones_like(norms)):
            print("Norms:", np.round(norms, 2))
            raise ValueError("Codewords are not normalized!")

        # ortho for rows of a mat => A^T @ A - I = 0
        ortho = code @ code.T - pt.eye(code.shape[0])
        if not pt.allclose(ortho, pt.zeros_like(ortho), atol=1e-3):
            print("Orthogonality check:", np.round(ortho, 2))
            raise ValueError("Codewords are not orthogonal!")

    def __getitem__(self, idx):
        return self.codewords[idx]

    def __len__(self):
        return self.codewords.shape[0]

    @staticmethod
    def fromStabilizers(
        stabilizers: list[str] | list[list[str]], method="svd", numpy=False
    ) -> "Code":
        """
        We may get `["XXY", "ZZI"]`, or `[["X", "X", "Y"], ["Z", "Z", "I"]]`. In the first case we need to split the strings into lists of characters, in the second case we can directly use the lists.

        We then use the constructed matrices to construct the projector onto the code space, and then extract the codewords using either `method="svd"` (Singular Value Decomposition, default) or `method="rrf"`(Randomized Range Finder).

        SVD is more stable but may be slower for large codes, while RRF is faster but may be less stable. For small codes, SVD is usually preferred.
        """
        assert method in [
            "svd",
            "rrf",
        ], "Only 'svd' and 'rrf' methods are supported for code construction"
        if isinstance(stabilizers[0], str):
            stabilizers = [list(s) for s in stabilizers]

        proj = Projector([composite(s) for s in stabilizers])

        if method == "svd":
            _, S, Vh = LA.svd(proj)
            code = Vh[pt.isclose(S, pt.tensor(1.0))]
        else:  # rrf
            if numpy:
                code = SVD_RRF_np(proj)
            else:
                code = SVD_RRF(proj)

        Code.isValid(code)
        return Code(code)
