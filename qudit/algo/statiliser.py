import math
from typing import List, Optional, Sequence, Union
from scipy.optimize import minimize
import torch as pt

from ..circuit.gates import Gategen

_paulis = {
    "X": pt.tensor([[0, 1], [1, 0]], dtype=pt.cfloat),
    "Y": pt.tensor([[0, -1j], [1j, 0]], dtype=pt.cfloat),
    "Z": pt.tensor([[1, 0], [0, -1]], dtype=pt.cfloat),
    "I": pt.eye(2, dtype=pt.cfloat),
}


def _S(*args: pt.Tensor) -> pt.Tensor:
    """
    Kron product of a sequence of operators.

    The ordering here matches the original implementation: starting from the
    first argument, each subsequent operator is kron'd on the *left*.
    """
    state = args[0]

    for d in args[1:]:
        state = pt.kron(d, state)

    return state


def S(string: str) -> pt.Tensor:
    """
    Convert a Pauli string (e.g. ``"XIIZ"``) into its operator tensor.
    """
    paulis = [_paulis[i] for i in string]

    return _S(*paulis)


def GramSchmidt(vectors: Sequence[pt.Tensor]) -> pt.Tensor:
    """
    Orthonormalise a list/sequence of vectors using Gram-Schmidt.

    Vectors with (near-)zero residual norm are dropped.
    """
    ortho: List[pt.Tensor] = []
    for v in vectors:
        w = v - sum((v @ u.conj()) * u for u in ortho)
        nrm = pt.norm(w)
        if nrm > 1e-8:
            ortho.append(w / nrm)

    return pt.stack(ortho)


class Statiliser:
    """
    Statiliser is a class that generates a basis of states stabilized by a given set of stabilizers. Stabilizers are represented as strings of Paulis (e.g., "ZZZII", "IIZZZ", "XIXXI", "IXXIX").

    `generate` uses gradient-free optimization to find states that are stabilized by the provided stabilizers, and then applies the Gram-Schmidt process to ensure the resulting states form an orthonormal basis.
    """

    stabilisers: List[pt.Tensor]
    sz: int
    num_states: int
    basis: Optional[pt.Tensor]

    def __init__(self, stabilisers: Sequence[Union[str, pt.Tensor]], d: int = 2):
        """
        Create a generator for the subspace stabilised by ``stabilisers``.
        """
        self.d = d
        gg = Gategen(dim=d)
        self._ops = {
            "I": pt.eye(d, dtype=pt.cfloat),
            "X": gg.X.tensor.to(pt.cfloat),
            "Z": gg.Z.tensor.to(pt.cfloat),
            "Y": gg.Y.tensor.to(pt.cfloat),
        }

        def _str_to_tensor(string: str) -> pt.Tensor:
            ops = [self._ops[c] for c in string]
            state = ops[0]
            for op in ops[1:]:
                state = pt.kron(op, state)

            return state

        stabilisers = [
            _str_to_tensor(s) if isinstance(s, str) else s for s in stabilisers
        ]
        self.stabilisers = list(stabilisers)
        self.sz = int(math.log(stabilisers[0].shape[0], d))
        self.num_states = d ** (self.sz - len(stabilisers))
        self.basis = None

    def _fun(self, x, mode: str = "real", minimal: int = 1) -> float:
        """
        Objective used by the optimiser.

        Combines:
        - stabiliser constraint violation: $||g|\psi\\rangle - |\psi\\rangle||$ summed over g
        - small L1 penalty to prefer sparse-ish solutions (optional) which may help in finding vectors closer to their canonical form
        - L2 penalty to encourage normalisation
        """
        vec = pt.as_tensor(x).to(pt.complex64)
        if mode != "real":
            vec = self._to_complex(vec)

        c1 = sum(pt.norm(g @ vec - vec).item() for g in self.stabilisers)
        L1 = pt.norm(vec, p=1).item()
        L2 = 2 * (1 - pt.norm(vec).item()) ** 2

        return float(c1 + (L1 + L2) * minimal)

    def _to_complex(self, vec: pt.Tensor) -> pt.Tensor:
        """
        Interpret a real tensor as concatenated real/imag parts.
        """
        l = len(vec)

        return vec[: l // 2] + 1j * vec[l // 2 :]

    def generate(
        self,
        mode: str = "real",
        tol: float = 1e-6,
        minimal: bool = True,
    ) -> pt.Tensor:
        """
        Generate an orthonormal basis for the stabilised subspace.
        """
        basis: List[pt.Tensor] = []
        factor = 2 if mode == "complex" else 1

        for _ in range(self.num_states):
            res = minimize(
                self._fun,
                x0=pt.rand(self.d**self.sz * factor).numpy(),
                args=(mode, int(minimal)),
                method="Powell",
                tol=tol,
            ).x
            state = pt.tensor(res, dtype=pt.float32 if mode == "real" else pt.cfloat)
            if mode != "real":
                state = self._to_complex(state)
            state = state / pt.norm(state)
            basis.append(state)

        basis_t = GramSchmidt(basis)
        basis_t = basis_t.to(dtype=pt.float16 if mode == "real" else pt.cfloat)
        self.basis = basis_t

        return basis_t
