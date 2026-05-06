from ..circuit.utils import to_mixed
import numpy as np
import torch


def matrix_of(gate) -> torch.Tensor:
    from ..circuit.gates import VarGate

    if isinstance(gate, VarGate):
        return gate._build_fn(gate.angle.to(torch.complex64)).detach()

    return gate.U


def _run_vector(qc) -> np.ndarray:
    ket0 = torch.zeros(qc.width, dtype=torch.complex64)
    ket0[0] = 1.0

    return qc(ket0).detach().cpu().numpy().flatten()


def stitch(amp: np.ndarray, bases: list) -> dict:
    probs = {}
    for i in range(len(amp)):
        key = "".join(str(d) for d in to_mixed(i, bases))
        probs[key] = float(np.abs(amp[i]) ** 2)

    return probs
