from MDR import Exam, load, Question
import sys

sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from qudit import Circuit
from torch.optim import Adam

C64 = torch.complex64
dev = "cpu"


class ParametricCircuit(nn.Module):
    """Minimal single-qubit RX circuit with a trainable angle."""

    def __init__(self, init_angle: float = 0.5):
        super().__init__()
        self.c = Circuit(wires=1, dim=2, device=dev)
        G = self.c.gates[2]
        self.angle = nn.Parameter(torch.tensor(init_angle))
        self.c.gate(G.RX, [0], angle=self.angle)

    def forward(self, x):
        return self.c(x)


def ket0():
    x = torch.zeros(2, dtype=C64)
    x[0] = 1.0
    return x


class GradientDescent(Question):
    """
    Quantum gradient-descent tests: differentiability of parametric circuits,
    gate-rotation values, and training convergence.
    """

    def test_gradient_flows(self):
        """
        Parametric $R_X(\\theta)$ circuit admits gradients:
        $\\partial p_1/\\partial\\theta = \\sin(\\theta)/2 \\neq 0$ at $\\theta=1$
        where $p_1 = |\\langle 1|R_X(\\theta)|0\\rangle|^2 = \\sin^2(\\theta/2)$
        """
        model = ParametricCircuit(init_angle=1.0)
        out = model(ket0())
        # p1 = sin^2(theta/2); d(p1)/d(theta) = sin(theta)/2 ≈ 0.421 at theta=1
        loss = out.reshape(-1)[1].abs().pow(2)
        loss.backward()
        self.assertIsNotNone(model.angle.grad)
        self.assertTrue(torch.isfinite(model.angle.grad))
        # Gradient must be non-zero;  sin(1)/2 ≈ 0.421
        self.assertGreater(model.angle.grad.abs().item(), 1e-4)

    def test_rx_pi_flips_qubit(self):
        """
        $R_X(\\pi)|0\\rangle \\sim |1\\rangle$: rotation by $\\pi$ flips the qubit
        (global phase $-i$ is irrelevant)
        """
        model = ParametricCircuit(init_angle=float(np.pi))
        out = model(ket0())
        target = torch.tensor([0.0, 1.0], dtype=C64)
        self.stateEqual(target.numpy(), out.detach().numpy())

    def test_rx_zero_identity(self):
        """
        $R_X(0)|0\\rangle = |0\\rangle$: zero rotation is the identity
        """
        model = ParametricCircuit(init_angle=0.0)
        out = model(ket0())
        self.stateEqual(ket0().numpy(), out.detach().numpy())

    def test_hybrid_training_reduces_loss(self):
        """
        Classical–quantum hybrid: linear encoder + $R_X$ circuit converges,
        $\\mathcal{L}_{\\mathrm{final}} < \\mathcal{L}_{\\mathrm{initial}}$
        """
        torch.manual_seed(42)

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Linear(4, 2, dtype=torch.float32)
                self.c = Circuit(wires=1, dim=2, device=dev)
                G = self.c.gates[2]
                self.angle = nn.Parameter(torch.tensor(0.5))
                self.c.gate(G.RX, [0], angle=self.angle)

            def forward(self, x):
                x = self.enc(x).to(C64)
                return self.c(x)

        model = HybridModel()
        data = torch.ones(4)
        target = torch.zeros(2, dtype=C64)
        target[1] = 1.0

        initial_loss = torch.norm(model(data) - target).item()
        optimizer = Adam(model.parameters(), lr=0.05)
        for _ in range(30):
            loss = torch.norm(model(data) - target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        final_loss = torch.norm(model(data) - target).item()
        self.assertLess(final_loss, initial_loss)

    def test_two_qubit_gradient_flows(self):
        """
        Two-qubit $R_X \\otimes R_X$ circuit: gradients flow to both parameters.
        $\\partial p_{11}/\\partial\\theta_i \\neq 0$ where $p_{11} = |\\langle 11|\\psi\\rangle|^2$.
        """
        c = Circuit(wires=2, dim=2, device=dev)
        G = c.gates[2]
        a0 = nn.Parameter(torch.tensor(0.5))
        a1 = nn.Parameter(torch.tensor(1.0))
        c.gate(G.RX, [0], angle=a0)
        c.gate(G.RX, [1], angle=a1)

        x = torch.zeros(4, dtype=C64)
        x[0] = 1.0
        out = c(x)
        # p11 = sin^2(a0/2)*sin^2(a1/2); nonzero gradient w.r.t. both angles
        loss = out.reshape(-1)[3].abs().pow(2)
        loss.backward()

        self.assertIsNotNone(a0.grad)
        self.assertIsNotNone(a1.grad)
        self.assertGreater(a0.grad.abs().item(), 1e-4)
        self.assertGreater(a1.grad.abs().item(), 1e-4)


if __name__ == "__main__":
    runner = Exam(
        name="Qudit Gradient Descent Tests",
        desc="Validation of parametric circuit differentiability and training convergence",
        file="gd.md",
    )
    runner.run(load(GradientDescent))
