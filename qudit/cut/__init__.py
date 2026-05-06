from typing import List, Tuple, Callable
from ..circuit.gates import gell_mann, Operator
from .utils import stitch, _run_vector
import numpy as np
import torch

C64 = torch.complex64


def _project(d: int) -> dict:
    diags = [gell_mann(j, j, d) for j in range(d - 1)]
    I = torch.eye(d, dtype=C64)
    P = {}
    for r in range(d):
        Pr = I / d
        for L in diags:
            Pr = Pr + 0.5 * L[r, r] * L
        P[r] = Pr

    return P


def _shift(d: int, r: int) -> torch.Tensor:
    X = torch.zeros((d, d), dtype=C64)
    for i in range(d):
        X[i, (i + r) % d] = 1

    return X


def _gm_basis(d: int) -> dict:
    B = {"I": Operator(torch.eye(d, dtype=C64), "I")}
    for j in range(d):
        for k in range(d):
            if j == d - 1 and k == d - 1:
                continue
            M = gell_mann(j, k, d)
            label = f"G({j},{k})"
            B[label] = Operator(M, label)

    return B


def _decompose_basis(A: torch.Tensor, basis: dict, threshold: float) -> list:
    out = []
    for label, op in basis.items():
        B = op.tensor
        num = torch.trace(A @ B.conj().T)
        den = torch.trace(B @ B.conj().T)
        c = (num / den).item()
        if abs(c) > threshold:
            out.append((op, c))

    return out


def _decomp_svd(CX: torch.Tensor, d1: int, d2: int, threshold: float) -> list:
    M = CX.view(d1, d2, d1, d2).permute(0, 2, 1, 3).reshape(d1 * d1, d2 * d2)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    terms = []
    for i in range(len(S)):
        if S[i].item() > threshold:
            opA = Operator(U[:, i].reshape(d1, d1), f"svdA{i}")
            opB = Operator(Vh[i, :].reshape(d2, d2), f"svdB{i}")
            terms.append((opA, opB, S[i].item()))

    return terms


class Coupler:
    """
    Decomposes the cross-dimensional CX gate between a d1-system and a d2-system
    into a sum of independent local operators: CX ≈ Σ_k c_k (A_k ⊗ B_k).

    Use method='optimal' (SVD, fewest terms) or method='basis' (Gell-Mann expansion).
    Terms are always (Operator, Operator, float) regardless of method.
    """

    def __init__(
        self, d1: int, d2: int, method: str = "optimal", threshold: float = 1e-6
    ):
        assert method in ("basis", "optimal"), "method must be 'basis' or 'optimal'"

        assert d1 >= 2 and d2 >= 2, "dimensions must be >= 2"
        self.d1 = d1
        self.d2 = d2
        self.method = method
        self.threshold = threshold
        self._P1 = _project(d1)
        self._P2 = _project(d2)
        self._B1 = _gm_basis(d1)
        self._B2 = _gm_basis(d2)
        self.terms: List[Tuple[Operator, Operator, float]] = self._build()

    @property
    def CX(self) -> torch.Tensor:
        w = self.d1 * self.d2
        cx = torch.zeros((w, w), dtype=C64)
        for r in range(self.d1):
            cx += torch.kron(self._P1[r], _shift(self.d2, r))

        return cx

    def _build(self) -> list:
        if self.method == "basis":
            terms = []
            for r in range(self.d1):

                A = self._P1[r]
                B = _shift(self.d2, r)
                for opA, cA in _decompose_basis(A, self._B1, self.threshold):
                    for opB, cB in _decompose_basis(B, self._B2, self.threshold):
                        terms.append((opA, opB, cA * cB))

            return terms

        return _decomp_svd(self.CX, self.d1, self.d2, self.threshold)

    def reconstruct(self) -> torch.Tensor:
        recon = torch.zeros_like(self.CX)
        for opA, opB, c in self.terms:
            recon += c * torch.kron(opA.tensor, opB.tensor)

        return recon

    def run(self, subA: Callable, subB: Callable) -> dict:
        """
        Execute circuit cutting given two circuit factory functions.

        subA(op) and subB(op) each accept an Operator and return a Circuit
        with that operator applied at the cut wire. Loops over decomposition
        terms, runs both sub-circuits from |0⟩, and reconstructs the joint
        probability distribution via amplitude superposition.
        """
        probeA = subA(self._B1["I"])
        probeB = subB(self._B2["I"])
        bases = probeA.dims_ + probeB.dims_
        n = probeA.width * probeB.width

        amp = np.zeros(n, dtype=np.complex64)
        for opA, opB, c in self.terms:
            amp += c * np.kron(_run_vector(subA(opA)), _run_vector(subB(opB)))

        return stitch(amp, bases)
