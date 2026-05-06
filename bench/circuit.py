import sys
import time
import numpy as np

sys.path.insert(0, "..")

import torch
import cirq
import pennylane as qml
import qutip as qt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from braket.circuits import Circuit as BCircuit
from braket.devices import LocalSimulator
from qutip_qip.circuit import QubitCircuit, CircuitSimulator

from qudit.circuit import Circuit
from qudit.circuit.index import Mode

C64 = torch.complex64
WARMUP = 5
N = 50
SIZES = [2, 4, 6, 8, 10, 12]


def _ket0(n):
    x = torch.zeros(2**n, dtype=C64)
    x[0] = 1.0
    return x


def _qudit_ghz(n):
    c = Circuit(wires=n, dim=2, mode=Mode.VECTOR)
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(n - 1):
        c.gate(G.CX, [i, i + 1])
    return c


def bench_qudit(n):
    c = _qudit_ghz(n)
    ket = _ket0(n)
    for _ in range(WARMUP):
        c(ket)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        c(ket)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_qiskit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    sv = Statevector.from_int(0, 2**n)
    for _ in range(WARMUP):
        Statevector(qc)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        Statevector(qc)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_cirq(n):
    q = cirq.LineQubit.range(n)
    ops = [cirq.H(q[0])] + [cirq.CNOT(q[i], q[i + 1]) for i in range(n - 1)]
    circuit = cirq.Circuit(ops)
    sim = cirq.Simulator()
    for _ in range(WARMUP):
        sim.simulate(circuit)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        sim.simulate(circuit)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_pennylane(n):
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev)
    def ghz():
        qml.Hadamard(wires=0)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    for _ in range(WARMUP):
        ghz()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        ghz()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_numpy(n):
    H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    CX = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
    )
    I2 = np.eye(2, dtype=np.complex64)

    def run(psi):
        full = np.kron(H, np.eye(2 ** (n - 1), dtype=np.complex64))
        psi = full @ psi
        for i in range(n - 1):
            left = np.eye(2**i, dtype=np.complex64)
            right = np.eye(2 ** (n - 2 - i), dtype=np.complex64)
            psi = np.kron(np.kron(left, CX), right) @ psi
        return psi

    psi = np.zeros(2**n, dtype=np.complex64)
    psi[0] = 1.0
    for _ in range(WARMUP):
        run(psi)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        run(psi)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_braket(n):
    dev = LocalSimulator()
    c = BCircuit()
    c.h(0)
    for i in range(n - 1):
        c.cnot(i, i + 1)
    c.state_vector()

    for _ in range(WARMUP):
        dev.run(c, shots=0).result()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        dev.run(c, shots=0).result()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_qutip(n):
    qc = QubitCircuit(n)
    qc.add_gate("SNOT", targets=[0])
    for i in range(n - 1):
        qc.add_gate("CNOT", controls=[i], targets=[i + 1])
    sim = CircuitSimulator(qc)
    psi0 = qt.tensor([qt.basis(2, 0)] * n)

    for _ in range(WARMUP):
        sim.run(psi0)
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        sim.run(psi0)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_qudit_mps(n):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")
    c = Circuit(wires=n, dim=2, mode=Mode.VECTOR, device="mps")
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(n - 1):
        c.gate(G.CX, [i, i + 1])
    ket = _ket0(n).to("mps")
    for _ in range(WARMUP):
        c(ket)
    torch.mps.synchronize()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        c(ket)
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


frameworks = {
    "qudit (cpu)": bench_qudit,
    "qudit (mps)": bench_qudit_mps,
    "qiskit": bench_qiskit,
    "cirq": bench_cirq,
    "pennylane": bench_pennylane,
    "braket": bench_braket,
    "qutip": bench_qutip,
    "numpy": bench_numpy,
}

if __name__ == "__main__":
    print(f"GHZ circuit benchmark — N={N} runs, warmup={WARMUP}\n")
    header = f"{'n':>4}  " + "".join(f"{'  '+name:>14}" for name in frameworks)
    print(header)
    print("-" * len(header))

    for n in SIZES:
        row = f"{n:>4}  "
        for name, fn in frameworks.items():
            try:
                times = fn(n)
                row += f"{np.mean(times):>12.3f}  "
            except Exception:
                row += f"{'—':>12}  "
        print(row)

    print(f"\n(all times in ms, mean over {N} runs)")
