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
from qudit.noise import WeylNoise, PhysicalNoise, Channel
from qudit.noise.lib import Process

from timing import timed

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
    return timed(fn, N, WARMUP, mps=mps)


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ghz_circuit(wires, mode, noise=None, device="cpu"):
    c = Circuit(wires=wires, dim=2, mode=mode, noise=noise, device=device)
    G = c.gates[2]
    c.gate(G.H, [0])
    for i in range(wires - 1):
        c.gate(G.CX, [i, i + 1])
    return c


# ── Depolarising ──────────────────────────────────────────────────────────────

def depol_qudit(wires, p):
    c = _ghz_circuit(wires, Mode.NOISY, WeylNoise(p=p))
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def depol_qudit_det(wires, p):
    K_local = WeylNoise(p=p).kraus_for("H", 0, 2)   # (d², 2, 2) local Kraus
    c = _ghz_circuit(wires, Mode.MATRIX)
    rho0 = _rho0(wires)
    I2 = torch.eye(2, dtype=K_local.dtype)

    def _embed(k, wire):
        ops = [I2] * wires
        ops[wire] = k
        out = ops[0]
        for op in ops[1:]:
            out = torch.kron(out, op)
        return out

    K_full = [[_embed(k, w) for k in K_local] for w in range(wires)]

    def run():
        rho = c(rho0).to(K_local.dtype)
        for Kw in K_full:
            rho = sum(k @ rho @ k.conj().T for k in Kw)
        return rho

    return _time(run)


def depol_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    c = _ghz_circuit(wires, Mode.NOISY, WeylNoise(p=p), "mps")
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
    c = _ghz_circuit(wires, Mode.NOISY, PhysicalNoise(T1=T1, T2=T2))
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def ad_qudit_det(wires, p):
    noise = Process.AD(d=2, n=wires, Y=p, order=wires - 1)
    c = _ghz_circuit(wires, Mode.MATRIX)
    rho0 = _rho0(wires)
    def run():
        rho = c(rho0)
        return noise.run(rho)
    return _time(run)


def ad_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    T1 = -50e-9 / np.log(max(1 - p, 1e-10))
    T2 = 2 * T1
    c = _ghz_circuit(wires, Mode.NOISY, PhysicalNoise(T1=T1, T2=T2), "mps")
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
    c = _ghz_circuit(wires, Mode.NOISY, PhysicalNoise(T1=T1, T2=T2))
    rho = _rho0(wires)
    return _time(lambda: c(rho))


def pd_qudit_det(wires, p):
    T2 = -50e-9 / np.log(max(1 - p, 1e-10))
    K_local = PhysicalNoise(T1=1.0, T2=T2).kraus_for("H", 0, 2)   # (r, 2, 2)
    c = _ghz_circuit(wires, Mode.MATRIX)
    rho0 = _rho0(wires)
    I2 = torch.eye(2, dtype=K_local.dtype)

    def _embed(k, wire):
        ops = [I2] * wires
        ops[wire] = k
        out = ops[0]
        for op in ops[1:]:
            out = torch.kron(out, op)
        return out

    K_full = [[_embed(k, w) for k in K_local] for w in range(wires)]

    def run():
        rho = c(rho0).to(K_local.dtype)
        for Kw in K_full:
            rho = sum(k @ rho @ k.conj().T for k in Kw)
        return rho

    return _time(run)


def pd_qudit_mps(wires, p):
    if not MPS:
        raise RuntimeError("MPS not available")
    T1 = 1.0
    T2 = -50e-9 / np.log(max(1 - p, 1e-10))
    c = _ghz_circuit(wires, Mode.NOISY, PhysicalNoise(T1=T1, T2=T2), "mps")
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
            "qudit (stochastic)": depol_qudit,
            "qudit (deterministic)": depol_qudit_det,
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
            "qudit (stochastic)": ad_qudit,
            "qudit (deterministic)": ad_qudit_det,
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
            "qudit (stochastic)": pd_qudit,
            "qudit (deterministic)": pd_qudit_det,
            "qudit (mps)": pd_qudit_mps,
            "qiskit": pd_qiskit,
            "cirq": pd_cirq,
            "pennylane": pd_pennylane,
            "braket": pd_braket,
            "qutip": pd_qutip,
        },
    },
}
