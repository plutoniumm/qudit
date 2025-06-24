import time
import numpy as np
from qudit.index import Gate, State,Basis
from qudit.circuit import Circuit, cfn, Layer
from qudit.gates import Gategen
from qudit.algebra import Unity, gellmann
from qudit.utils import ID, isVar, Braket, Tensor
import cirq
import qutip
from qutip import basis, tensor
from qutip.qip.operations import hadamard_transform, cnot


REPEATS = 100


def Unity(d):
    return np.exp(2j * np.pi / d)


def benchmark_custom():
    D = Gategen(2)
    C = Circuit(2)
    C.gate(D.H, dits=[0])
    C.gate(D.CX, dits=[0, 1])

    start = time.perf_counter()
    for t in range(REPEATS):
        t = C.solve()
    end = time.perf_counter()

    return (end - start) / REPEATS


def benchmark_cirq():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.Simulator()

    start = time.perf_counter()
    for _ in range(REPEATS):
        _ = simulator.simulate(circuit)
    end = time.perf_counter()
    return (end - start) / REPEATS


def benchmark_qutip():
    ket0 = basis(2, 0)
    H = hadamard_transform(1)
    I = qutip.qeye(2)
    CX = cnot()

    start = time.perf_counter()
    for _ in range(REPEATS):
        psi = tensor(ket0, ket0)
        psi = tensor(H, I) * psi
        bell = CX * psi
    end = time.perf_counter()

    return (end - start) / REPEATS


if __name__ == "__main__":
    print(f"Benchmarking over {REPEATS} runs...\n")
    print(f"Circuit     : {benchmark_custom():.6f} s")
    print(f"Cirq             : {benchmark_cirq():.6f} s")
    print(f"QuTiP            : {benchmark_qutip():.6f} s")
