import torch as pt
import math
import cmath
from .lindblad import lindblad_kraus

C128 = pt.complex128

_2Q_GATES = {"CX", "CZ", "CNOT", "SWAP", "CRX", "CRY", "CRZ"}

_DEFAULT_TIMES = {
    "1q": 50e-9,
    "2q": 200e-9,
}


class PhysicalNoise:
    """
    Noise model derived from physical T1/T2 parameters via Lindblad integration.
    """

    def __init__(self, T1, T2, method="free", gate_times=None, seed=None):
        self.T1 = T1
        self.T2 = T2
        self.method = method
        self.gate_times = gate_times or {}
        self.generator = pt.Generator().manual_seed(seed) if seed is not None else None
        self._cache = {}

    def params_for(self, wire: int) -> tuple[float, float]:
        T1 = self.T1[wire] if isinstance(self.T1, list) else self.T1
        T2 = self.T2[wire] if isinstance(self.T2, list) else self.T2

        return T1, T2

    def _gate_time(self, gate_name: str) -> float:
        if gate_name in self.gate_times:
            return self.gate_times[gate_name]

        return _DEFAULT_TIMES["2q"] if gate_name in _2Q_GATES else _DEFAULT_TIMES["1q"]

    def kraus_for(self, gate_name: str, wire: int, d: int) -> pt.Tensor:
        key = (gate_name, wire, d)
        if key in self._cache:
            return self._cache[key]

        T1, T2 = self.params_for(wire)
        t = self._gate_time(gate_name)
        result = lindblad_kraus(None, T1, T2, t, d)
        self._cache[key] = result

        return result


class WeylNoise:
    """
    Depolarising-style noise via Weyl (generalized Pauli) operators.
    """

    def __init__(self, p: float, seed=None):
        self.p = p
        self.generator = pt.Generator().manual_seed(seed) if seed is not None else None
        self._cache: dict[int, pt.Tensor] = {}

    def _build(self, d: int) -> pt.Tensor:
        w = cmath.exp(2j * cmath.pi / d)
        shift = pt.roll(pt.eye(d, dtype=C128), shifts=-1, dims=1)
        clock = pt.diag(pt.tensor([w**k for k in range(d)], dtype=C128))

        p, d2 = self.p, d * d

        ops = []
        w_I = math.sqrt(max(0.0, 1.0 - p * (d2 - 1) / d2))
        ops.append(w_I * pt.eye(d, dtype=C128))

        w_err = math.sqrt(p / d2)
        for m in range(d):
            for n in range(d):
                if m == 0 and n == 0:
                    continue
                W = pt.linalg.matrix_power(shift, m) @ pt.linalg.matrix_power(clock, n)
                ops.append(w_err * W)

        return pt.stack(ops)

    def kraus_for(self, gate_name: str, wire: int, d: int) -> pt.Tensor:
        if d not in self._cache:
            self._cache[d] = self._build(d)

        return self._cache[d]


class CoherentNoise:
    """
    Coherent (unitary) error model: perturbs a gate by a random GUE rotation.
    """

    def __init__(self, eps: float):
        self.eps = eps

    def perturb(self, U: pt.Tensor, d: int) -> pt.Tensor:
        A = pt.randn(d, d, dtype=C128)
        G = (A + A.conj().T) / 2.0
        noise = pt.linalg.matrix_exp(-1j * self.eps * G)

        return (noise @ U.to(C128)).to(U.dtype)
