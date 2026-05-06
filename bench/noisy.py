"""
Noisy simulation benchmark — one section per noise type.

Each noise type is benchmarked independently across frameworks using their
native implementations. Sizes = number of qubits [2, 4, 6].
"""
import sys
import time
import numpy as np

sys.path.insert(0, "..")

import torch
import cirq
import pennylane as qml
import qutip as qt
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
)
from braket.circuits import Circuit as BCircuit, Noise, ResultType
from braket.devices import LocalSimulator
from qutip_qip.circuit import QubitCircuit, CircuitSimulator

from qudit.circuit import Circuit
from qudit.circuit.index import Mode
from qudit.noise import WeylNoise, PhysicalNoise

C64 = torch.complex64
WARMUP = 3
N = 30
SIZES = [2, 4, 6]


MPS = torch.backends.mps.is_available()


def _ket0(n, device="cpu"):
    x = torch.zeros(2**n, dtype=C64, device=device)
    x[0] = 1.0
    return x


def _rho0(n, device="cpu"):
    x = _ket0(n, device)
    return x.view(-1, 1) @ x.view(1, -1).conj()


def _time(fn, mps=False):
    for _ in range(WARMUP):
        fn()
        if mps:
            torch.mps.synchronize()
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        if mps:
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _qiskit_sim(wires, noise_model):
    qc = QuantumCircuit(wires)
    qc.h(0)
    for i in range(wires - 1):
        qc.cx(i, i + 1)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    tqc = transpile(qc, sim)

    def run():
        sim.run(tqc).result()

    return _time(run)


def _braket_sim(wires, noise):
    dev = LocalSimulator("braket_dm")
    c = BCircuit()
    c.h(0)
    for i in range(wires - 1):
        c.cnot(i, i + 1)
    c.apply_gate_noise(noise)
    c.density_matrix()

    def run():
        dev.run(c, shots=0).result()

    return _time(run)


def _qutip_sim(wires, kraus_fn):
    qc = QubitCircuit(wires)
    qc.add_gate("SNOT", targets=[0])
    for i in range(wires - 1):
        qc.add_gate("CNOT", controls=[i], targets=[i + 1])
    sim = CircuitSimulator(qc)
    psi0 = qt.tensor([qt.basis(2, 0)] * wires)
    I_op = qt.qeye(2)
    kraus_sets = [kraus_fn(wires, q, I_op) for q in range(wires)]

    def run():
        rho = qt.ket2dm(sim.run(psi0).final_states[0])
        for Ks in kraus_sets:
            rho = sum(K * rho * K.dag() for K in Ks)
        return float(rho.tr().real)

    return _time(run)


# ── Depolarising ──────────────────────────────────────────────────────────────

def depol_qudit(wires, p):
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=WeylNoise(p=p))
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def depol_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=WeylNoise(p=p), device="mps")
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires, "mps")
    return _time(lambda: c(rho), mps=True)


def depol_qiskit(wires, p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    return _qiskit_sim(wires, nm)


def depol_cirq(wires, p):
    q = cirq.LineQubit.range(wires)
    ops = [cirq.H(q[0])] + [cirq.CNOT(q[i], q[i + 1]) for i in range(wires - 1)]
    circuit = cirq.Circuit(ops)
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(p))
    sim = cirq.DensityMatrixSimulator(noise=noise)
    return _time(lambda: sim.simulate(circuit))


def depol_pennylane(wires, p):
    dev = qml.device("default.mixed", wires=wires)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(wires - 1):
            qml.CNOT(wires=[i, i + 1])
        for w in range(wires):
            qml.DepolarizingChannel(p, wires=w)
        return qml.density_matrix(wires=list(range(wires)))

    return _time(circuit)


def depol_braket(wires, p):
    return _braket_sim(wires, Noise.Depolarizing(probability=p))


def depol_qutip(wires, p):
    X, Y, Z_op = qt.sigmax(), qt.sigmay(), qt.sigmaz()

    def kraus_fn(n, q, I_op):
        def wrap(op):
            return qt.tensor([op if i == q else I_op for i in range(n)])
        return [
            wrap(np.sqrt(1 - p) * I_op),
            wrap(np.sqrt(p / 3) * X),
            wrap(np.sqrt(p / 3) * Y),
            wrap(np.sqrt(p / 3) * Z_op),
        ]

    return _qutip_sim(wires, kraus_fn)


# ── Amplitude Damping ─────────────────────────────────────────────────────────

def ad_qudit(wires, p):
    T1 = -50e-9 / np.log(max(1 - p, 1e-10))
    T2 = 2 * T1
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=PhysicalNoise(T1=T1, T2=T2))
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def ad_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    T1 = -50e-9 / np.log(max(1 - p, 1e-10))
    T2 = 2 * T1
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=PhysicalNoise(T1=T1, T2=T2), device="mps")
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires, "mps")
    return _time(lambda: c(rho), mps=True)


def ad_qiskit(wires, p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(amplitude_damping_error(p), ["h"])
    nm.add_all_qubit_quantum_error(amplitude_damping_error(p), ["cx"])
    return _qiskit_sim(wires, nm)


def ad_cirq(wires, p):
    q = cirq.LineQubit.range(wires)
    ops = [cirq.H(q[0])] + [cirq.CNOT(q[i], q[i + 1]) for i in range(wires - 1)]
    circuit = cirq.Circuit(ops)
    noise = cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(p))
    sim = cirq.DensityMatrixSimulator(noise=noise)
    return _time(lambda: sim.simulate(circuit))


def ad_pennylane(wires, p):
    dev = qml.device("default.mixed", wires=wires)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(wires - 1):
            qml.CNOT(wires=[i, i + 1])
        for w in range(wires):
            qml.AmplitudeDamping(p, wires=w)
        return qml.density_matrix(wires=list(range(wires)))

    return _time(circuit)


def ad_braket(wires, p):
    return _braket_sim(wires, Noise.AmplitudeDamping(gamma=p))


def ad_qutip(wires, p):
    E0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - p)]])
    E1 = qt.Qobj([[0, np.sqrt(p)], [0, 0]])

    def kraus_fn(n, q, I_op):
        def wrap(op):
            return qt.tensor([op if i == q else I_op for i in range(n)])
        return [wrap(E0), wrap(E1)]

    return _qutip_sim(wires, kraus_fn)


# ── Phase Damping ─────────────────────────────────────────────────────────────

def pd_qudit(wires, p):
    T1 = 1.0
    T2 = -50e-9 / np.log(max(1 - p, 1e-10))
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=PhysicalNoise(T1=T1, T2=T2))
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def pd_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    T1 = 1.0
    T2 = -50e-9 / np.log(max(1 - p, 1e-10))
    c = Circuit(wires=wires, dim=2, mode=Mode.NOISY, noise=PhysicalNoise(T1=T1, T2=T2), device="mps")
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    rho = _rho0(wires, "mps")
    return _time(lambda: c(rho), mps=True)


def pd_qiskit(wires, p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(phase_damping_error(p), ["h"])
    nm.add_all_qubit_quantum_error(phase_damping_error(p), ["cx"])
    return _qiskit_sim(wires, nm)


def pd_cirq(wires, p):
    q = cirq.LineQubit.range(wires)
    ops = [cirq.H(q[0])] + [cirq.CNOT(q[i], q[i + 1]) for i in range(wires - 1)]
    circuit = cirq.Circuit(ops)
    noise = cirq.ConstantQubitNoiseModel(cirq.phase_damp(p))
    sim = cirq.DensityMatrixSimulator(noise=noise)
    return _time(lambda: sim.simulate(circuit))


def pd_pennylane(wires, p):
    dev = qml.device("default.mixed", wires=wires)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(wires - 1):
            qml.CNOT(wires=[i, i + 1])
        for w in range(wires):
            qml.PhaseDamping(p, wires=w)
        return qml.density_matrix(wires=list(range(wires)))

    return _time(circuit)


def pd_braket(wires, p):
    return _braket_sim(wires, Noise.PhaseDamping(gamma=p))


def pd_qutip(wires, p):
    K0 = qt.Qobj([[1, 0], [0, np.sqrt(1 - p)]])
    K1 = qt.Qobj([[0, 0], [0, np.sqrt(p)]])

    def kraus_fn(n, q, I_op):
        def wrap(op):
            return qt.tensor([op if i == q else I_op for i in range(n)])
        return [wrap(K0), wrap(K1)]

    return _qutip_sim(wires, kraus_fn)


# ── Registry ──────────────────────────────────────────────────────────────────

NOISE_TYPES = {
    "depol": {
        "label": "Depolarising (p=0.01)",
        "p": 0.01,
        "frameworks": {
            "qudit (cpu)": depol_qudit,
            "qudit (mps)": depol_qudit_mps,
            "qiskit": depol_qiskit,
            "cirq": depol_cirq,
            "pennylane": depol_pennylane,
            "braket": depol_braket,
            "qutip": depol_qutip,
        },
    },
    "ad": {
        "label": "Amplitude Damping (γ=0.05)",
        "p": 0.05,
        "frameworks": {
            "qudit (cpu)": ad_qudit,
            "qudit (mps)": ad_qudit_mps,
            "qiskit": ad_qiskit,
            "cirq": ad_cirq,
            "pennylane": ad_pennylane,
            "braket": ad_braket,
            "qutip": ad_qutip,
        },
    },
    "phase_damp": {
        "label": "Phase Damping (γ=0.05)",
        "p": 0.05,
        "frameworks": {
            "qudit (cpu)": pd_qudit,
            "qudit (mps)": pd_qudit_mps,
            "qiskit": pd_qiskit,
            "cirq": pd_cirq,
            "pennylane": pd_pennylane,
            "braket": pd_braket,
            "qutip": pd_qutip,
        },
    },
}
