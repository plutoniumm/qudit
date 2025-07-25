import sys, json

sys.path.append("..")

from qudiet.core.quantum_circuit import QuantumCircuit as Qudietc

from qudiet.core.backend.NumpyBackend import NumpyBackend
from qutip_qip.operations import hadamard_transform, cnot
from braket.circuits import Circuit as Braketc
from qiskit import QuantumCircuit as Qiskitc
from qiskit.quantum_info import Statevector
from braket.devices import LocalSimulator
from time import perf_counter as bench

from qutip import basis, tensor, qeye
from qudit.circuit import Circuit

import matplotlib.pyplot as plt
import quforge.quforge as qf
import pennylane as qml
import numpy as np
import cirq as CQ
import torch


def b_pennylane(n, repeats):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def ghz_circuit():
        qml.Hadamard(wires=0)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    start = bench()
    for _ in range(repeats):
        _ = ghz_circuit()
    return (bench() - start) / repeats


def b_braket(n, repeats):
    circ = Braketc()
    circ.h(0)
    for i in range(n - 1):
        circ.cnot(i, i + 1)
    device = LocalSimulator()

    start = bench()
    for _ in range(repeats):
        _ = device.run(circ, shots=1).result()
    return (bench() - start) / repeats


def b_qudiet(n, repeats):
    qc = Qudietc(qregs=[2] * n, backend=NumpyBackend)
    qc.h(0)
    for i in range(n - 1):
        qc.cx([i, i + 1], 2)
    qc.measure_all()

    start = bench()
    for _ in range(repeats):
        _ = qc.run()
    return (bench() - start) / repeats


def b_quforge(n, repeats):
    circ = qf.Circuit(dim=2, wires=n)
    state = qf.State("0" + "-0" * (n - 1), dim=2)

    circ.H(index=[0])
    for i in range(n - 1):
        circ.CNOT(index=[i, i + 1])

    start = bench()
    for _ in range(repeats):
        circ(state)
    return (bench() - start) / repeats


def b_qudit(n, repeats):
    circuit = Circuit(n, dim=2, device="cpu")
    CX = circuit.make([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], "CX")
    H = circuit.make([[1, 1], [1, -1]], "H")

    circuit.gate(H, [0])
    for i in range(n - 1):
        circuit.gate(CX, [i, i + 1])

    state = torch.zeros(2**n, dtype=torch.complex64)
    state[0] = 1  # |0...0>

    start = bench()
    for _ in range(repeats):
        _ = circuit(state)
    return (bench() - start) / repeats


def b_qiskit(n, repeats):
    qc = Qiskitc(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    start = bench()
    for _ in range(repeats):
        _ = Statevector(qc)

    return (bench() - start) / repeats


def b_cirq(n, repeats):
    q = CQ.LineQubit.range(n)
    ops = [CQ.H(q[0])] + [CQ.CNOT(q[i], q[i + 1]) for i in range(n - 1)]
    circuit = CQ.Circuit(ops)
    sim = CQ.Simulator()

    start = bench()
    for _ in range(repeats):
        _ = sim.simulate(circuit)
    return (bench() - start) / repeats


def b_qutip(n, repeats):
    ket0 = basis(2, 0)
    psi = tensor([ket0] * n)
    H = hadamard_transform(1)
    I = qeye(2)

    start = bench()
    for _ in range(repeats):
        state = psi
        ops = [I] * n
        ops[0] = H
        state = tensor(ops) * state
        for i in range(n - 1):
            CX = cnot(n, control=i, target=i + 1)
            state = CX * state
    return (bench() - start) / repeats


n_range = range(3, 25)
LOG_THRESHOLD = 5
repeats = 10
ms = 1e3

backends = {
    "Qudit": b_qudit,
    "Cirq": b_cirq,
    "PennyLane": b_pennylane,
    "Qudiet": b_qudiet,
    "QuForge": b_quforge,
    "Braket": b_braket,
    "Qiskit": b_qiskit,
    "QuTiP": b_qutip,
}

results = {name: [] for name in backends}

for n in n_range:
    print(f"{n}/{len(n_range) + 2}")
    for name in list(backends.keys()):
        bench_fn = backends[name]
        if bench_fn is None:
            continue

        t = bench_fn(n, repeats) * ms
        log_t = np.log(t)
        print(f"\t{name}: {t:.3f} ms")
        if log_t > LOG_THRESHOLD:
            backends[name] = None  # Mark as stopped
            continue
        results[name].append(log_t)

for name in list(backends.keys()):
    if backends[name] is None:
        del backends[name]

for name, times in results.items():
    if times:
        plt.plot(n_range[: len(times)], times, label=name, marker=".")

data = {name: times for name, times in results.items() if times}
with open("bench_n2.json", "w") as f:
    json.dump(data, f, indent=4)

plt.xlabel("Num Qubits (n)")
plt.ylabel("log (avg ms/run)")
plt.title("GHZ Circuit Benchmark")
plt.xticks(n_range)
plt.legend()
plt.axhline(LOG_THRESHOLD, color="red", linestyle="--", label="Log Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig("bench_n2.png", dpi=300)
