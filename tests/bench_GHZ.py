import sys

sys.path.append("..")

from qudiet.core.quantum_circuit import QuantumCircuit as Qudietc
from qudiet.core.backend.NumpyBackend import NumpyBackend
from qutip_qip.operations import hadamard_transform, cnot
from qiskit.quantum_info import Operator
from time import perf_counter as bench
from qutip import basis, tensor, qeye
from qiskit import QuantumCircuit as Qiskitc
from qudit.circuit import Circuit
import matplotlib.pyplot as plt
import quforge.quforge as qf
import numpy as np
import cirq as CQ

ms = 1e3

qutip_times = []
qiskit_times = []
cirq_times = []
custom_times = []
qudiet_times = []
quforge_times = []

def bench_qudiet(n, repeats):
    qc = Qudietc(
        qregs=[2]*n,
        backend=NumpyBackend
    )
    qc.h(0)
    for i in range(n - 1):
        qc.cx([i, i+1], 2)
    qc.measure_all()

    start = bench()
    for _ in range(repeats):
        _ = qc.run()
    return (bench() - start) / repeats

def bench_quforge(n, repeats):
    circ = qf.Circuit(dim=2, wires=n)
    state = qf.State('0' + '-0' * (n-1), dim=2)

    circ.H(index=[0])
    for i in range(n-1):
        circ.CNOT(index=[i, i + 1])

    start = bench()
    for _ in range(repeats):
        circ(state)
    return (bench() - start) / repeats

def bench_custom(n, repeats):
    C = Circuit(n, dim=2)
    G = C.gates
    C.gate(G.H, dits=[0])
    for i in range(n - 1):
        C.gate(G.CX, dits=[i, i + 1])
    start = bench()
    for _ in range(repeats):
        _ = C.run()
    return (bench() - start) / repeats


def bench_qiskit(n, repeats):
    start = bench()
    for _ in range(repeats):
        qc = Qiskitc(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        _ = Operator.from_circuit(qc)
    return (bench() - start) / repeats


def bench_cirq(n, repeats):
    q = CQ.LineQubit.range(n)
    ops = [CQ.H(q[0])] + [CQ.CNOT(q[i], q[i + 1]) for i in range(n - 1)]
    circuit = CQ.Circuit(ops)
    sim = CQ.Simulator()
    start = bench()
    for _ in range(repeats):
        _ = sim.simulate(circuit)
    return (bench() - start) / repeats


def bench_qutip(n, repeats):
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


n_range = range(2, 10)
for n in n_range:
    print(f"{n}/{len(n_range) + 2}")
    repeats = 10

    t_qiskit = ms * bench_qiskit(n, repeats)
    print(f"\tQiskit: {t_qiskit:.3f} ms")
    t_cirq = ms * bench_cirq(n, repeats)
    print(f"\tCirq: {t_cirq:.3f} ms")
    t_qutip = ms * bench_qutip(n, repeats)
    print(f"\tQuTiP: {t_qutip:.3f} ms")
    t_custom = ms * bench_custom(n, repeats)
    print(f"\tQudit: {t_custom:.3f} ms")
    t_qudiet = ms * bench_qudiet(n, repeats)
    print(f"\tQudiet: {t_qudiet:.3f} ms")
    t_quforge = ms * bench_quforge(n, repeats)
    print(f"\tQuForge: {t_quforge:.3f} ms")

    tot = (t_custom + t_qiskit + t_cirq + t_qutip) / ms
    print(f"\tTotal: {tot*repeats/60:.3f}")

    custom_times.append(t_custom)
    qiskit_times.append(t_qiskit)
    cirq_times.append(t_cirq)
    qutip_times.append(t_qutip)
    qudiet_times.append(t_qudiet)
    quforge_times.append(t_quforge)

custom_times = np.log(custom_times)
qiskit_times = np.log(qiskit_times)
cirq_times = np.log(cirq_times)
qutip_times = np.log(qutip_times)
qudiet_times = np.log(qudiet_times)
quforge_times = np.log(quforge_times)

plt.plot(n_range, qudiet_times, label="Qudiet", marker=".")
plt.plot(n_range, quforge_times, label="QuForge", marker=".")
plt.plot(n_range, custom_times, label="Qudit", marker=".")
plt.plot(n_range, qiskit_times, label="Qiskit", marker=".")
plt.plot(n_range, cirq_times, label="Cirq", marker=".")
plt.plot(n_range, qutip_times, label="QuTiP", marker=".")
plt.xlabel("Number of Qubits (n)")
plt.ylabel("Avg Time per Run (ms)")
plt.title("GHZ Circuit Benchmark")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
