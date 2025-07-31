from scipy.optimize import minimize
import torch as pt

"""
# Statiliser: Usage

statiliser = Statiliser(["ZZZII", "IIZZZ", "XIXXI", "IXXIX"])

states = statiliser.generate()
print(states)

statiliser.draw(0)
statiliser.draw(1)
"""

_paulis = {
    "X": pt.tensor([[0, 1], [1, 0]], dtype=pt.cfloat),
    "Y": pt.tensor([[0, -1j], [1j, 0]], dtype=pt.cfloat),
    "Z": pt.tensor([[1, 0], [0, -1]], dtype=pt.cfloat),
    "I": pt.eye(2, dtype=pt.cfloat),
}


def _S(*args) -> pt.Tensor:
    state = args[0]
    for d in args[1:]:
        state = pt.kron(d, state)
    return state


def S(string: str) -> pt.Tensor:
    paulis = [_paulis[i] for i in string]
    return _S(*paulis)


def GramSchmidt(vectors):
    ortho = []
    for v in vectors:
        w = v - sum((v @ u.conj()) * u for u in ortho)
        if pt.norm(w) > 1e-8:
            ortho.append(w / pt.norm(w))
    return pt.stack(ortho)


class Statiliser:
    def __init__(self, stabilisers):
        stabilisers = [S(s) for s in stabilisers]
        self.stabilisers = stabilisers
        self.sz = int(pt.log2(pt.tensor(stabilisers[0].shape[0])).item())
        self.num_states = 2 ** (self.sz - len(stabilisers))
        self.basis = None

    def _fun(self, x, mode="real", minimal=1):
        vec = pt.tensor(x, dtype=pt.complex64)
        if mode != "real":
            vec = self._to_complex(vec)

        c1 = sum(pt.norm(g @ vec - vec).item() for g in self.stabilisers)
        L1 = pt.norm(vec, p=1).item()
        L2 = 2 * (1 - pt.norm(vec).item()) ** 2

        return c1 + (L1 + L2) * minimal

    def _to_complex(self, vec):
        l = len(vec)
        return vec[: l // 2] + 1j * vec[l // 2 :]

    def generate(self, mode="real", tol=1e-6, minimal=True):
        basis = []
        factor = 2 if mode == "complex" else 1

        for _ in range(self.num_states):
            res = minimize(
                self._fun,
                x0=pt.rand(2**self.sz * factor).numpy(),
                args=(mode, int(minimal)),
                method="Powell",
                tol=tol,
            ).x
            state = pt.tensor(res, dtype=pt.float32 if mode == "real" else pt.cfloat)
            if mode != "real":
                state = self._to_complex(state)
            state = state / pt.norm(state)
            basis.append(state)

        basis = GramSchmidt(basis)
        basis = basis.to(dtype=pt.float16 if mode == "real" else pt.cfloat)
        self.basis = basis

        return basis
