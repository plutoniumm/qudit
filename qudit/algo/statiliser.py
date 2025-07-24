from scipy.optimize import minimize
import numpy as np


"""
# Statiliser: Usage

statiliser = Statiliser(["ZZZII", "IIZZZ", "XIXXI", "IXXIX"])

states = statiliser.generate()
print(states.round(3))

statiliser.draw(0)
statiliser.draw(1)
"""


_paulis = {
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
    "I": np.eye(2),
}


# Usage _S(X, I, X, X, I)
def _S(*args) -> np.ndarray:
    state = args[0]
    for d in args[1:]:
        state = np.kron(d, state)

    return state


# Usage S("XIXXI")
# literally just call S() string by string
def S(string: str) -> np.ndarray:
    paulis = [_paulis[i] for i in string]
    return _S(*paulis)


def GramSchmidt(vectors):
    ortho = []
    for v in vectors:
        w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
        if np.linalg.norm(w) > 1e-8:
            ortho.append(w / np.linalg.norm(w))

    return np.array(ortho)


class Statiliser:
    def __init__(self, stabilisers):
        stabilisers = [S(s) for s in stabilisers]
        self.stabilisers = stabilisers
        self.sz = int(np.log2(len(stabilisers[0])))
        self.num_states = 2 ** (self.sz - len(stabilisers))
        self.basis = None

    def _fun(self, x, mode="real", minimal=1):
        vec = x if mode == "real" else self._to_complex(x)
        c1 = sum(np.linalg.norm((g @ vec) - vec) for g in self.stabilisers)
        # minimise ||x||_1 s.t. ||x||_2 = 1
        L1 = np.linalg.norm(vec, 1)
        L2 = 2 * (1 - np.linalg.norm(vec, 2)) ** 2
        # L1 = L2 = 0

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
                x0=np.random.rand(2**self.sz * factor),
                args=(mode, int(minimal)),
                method="Powell",
                tol=tol,
            ).x
            state = res / np.linalg.norm(res)
            basis.append(state)

        basis = np.array(GramSchmidt(basis))
        basis = basis.astype(np.float16)
        self.basis = basis

        return basis
