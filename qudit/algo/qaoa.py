import torch.nn as nn
import torch as pt

from ..circuit import gates as GG
from typing import Callable

devnull = lambda *args, **kwargs: None

C64 = pt.complex64


class QUBO:
    """
    QUBO (Quadratic Unconstrained Binary Optimization) problems defined by a matrix `Q` to minimize the function:

    $f(x) = \sum_{i} Q[i, i] \\times x[i] + \sum_{i < j} Q[i, j] \\times x[i] \\times x[j]$

    where x[i] are binary variables (0 or 1). The `toHamiltonian` method converts this QUBO representation into a Hamiltonian suitable for quantum algorithms like QAOA.
    """

    @staticmethod
    def toHamiltonian(Q: dict):
        """
        Convert a QUBO problem defined by matrix Q into a Hamiltonian representation.

        Function iterates over `Q` constructs Hamiltonian in terms of Pauli Z ops. Linear terms are (where $i = j$) and quadratic terms are (where $i \\neq j$). The resulting Hamiltonian is `list[tuple]`, where
        """
        ising = {}
        offset = 0.0

        for (i, j), val in Q.items():
            if i == j:
                ising[i] = ising.get(i, 0) - val / 2
                offset += val / 2
            else:
                key = tuple(sorted((i, j)))
                ising[key] = ising.get(key, 0) + val / 4
                ising[i] = ising.get(i, 0) - val / 4
                ising[j] = ising.get(j, 0) - val / 4
                offset += val / 4

        ham = []
        for keys, coeff in ising.items():
            if coeff == 0:
                continue
            if isinstance(keys, int):
                ham.append((coeff, "Z", [keys]))
            else:
                ham.append((coeff, "ZZ", list(keys)))

        return ham, offset


Energy = Callable[[dict, list], float]


class QAOA(nn.Module):
    """
    Quantum Approximate Optimization Algorithm (QAOA) implementation for qudits.

    QAOA has a known structure, and therefore instead of being built on circuit, inherits directly from `nn.Module`. The `forward` method constructs the QAOA state based on the provided Hamiltonian, and the `expectation` method computes the expectation value of the Hamiltonian with respect to the current state.
    """

    d: int
    wires: int
    qubo: dict = None
    hamiltonian: list = None
    offset: float = 0.0
    device: str

    def __init__(
        self,
        d: int,
        wires: int,
        qubo: dict = None,
        hamiltonian: list = None,
        offset: float = 0.0,
        layers=1,
        device="cpu",
    ):
        super().__init__()
        self.width = d**wires
        self.wires = wires
        self.d = d
        self.device = device
        self.layers = layers
        self.qubo = qubo

        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
            self.offset = offset
        elif qubo is not None:
            self.hamiltonian, self.offset = QUBO.toHamiltonian(qubo)
        else:
            raise ValueError("Either 'hamiltonian' or 'qubo' must be provided.")

        self._H_P_matrix = self.getMat()
        self.gammas = nn.Parameter(pt.rand(layers, device=device) * (2 * pt.pi))
        self.betas = nn.Parameter(pt.rand(layers, device=device) * pt.pi)
        self.gg = GG.Gategen(dim=d, device=device)
        self._H_factory = self.gg.U(self.gg.H, name="H")
        self._CX_factory = self.gg.U(self.gg.CX, name="CX")

        self.OpMap = {
            "Z": self._RZ,
            "ZZ": self._RZZ,
        }

    def gate(self, x, gate_class, index, **kwargs):
        """
        Helper function to apply a gate to the state `x` using the specified `gate_class` and `index`. Additional parameters for the gate can be passed via `kwargs`.
        """

        gate = gate_class(
            dim=self.d, wires=self.wires, index=index, device=self.device, **kwargs
        )

        return gate.forward(x)

    def _RZ(self, x, angle, indices):
        return self.gate(x, self.gg.RZ, index=indices, angle=angle)

    def _RZZ(self, x, angle, indices):
        x = self.gate(x, self._CX_factory, index=indices)
        x = self.gate(x, self.gg.RZ, index=[indices[1]], angle=angle)
        x = self.gate(x, self._CX_factory, index=indices)

        return x

    def forward(self):
        state = pt.zeros((self.width, 1), dtype=C64, device=self.device)
        state[0, 0] = 1.0

        for j in range(self.wires):
            state = self.gate(state, self._H_factory, index=[j])

        for i in range(self.layers):
            for coeff, gtype, indices in self.hamiltonian:
                angle = 2 * self.gammas[i] * coeff
                if gtype in self.OpMap:
                    state = self.OpMap[gtype](state, angle, indices)
                else:
                    raise NotImplementedError(f"Gate type '{gtype}' not supported.")

            for j in range(self.wires):
                state = self.gate(state, self.gg.RX, index=[j], angle=2 * self.betas[i])

        return state

    def getMat(self):
        """
        Construct Hamiltonian matrix from `self.hamiltonian`.
        """

        H_P = pt.zeros((self.width, self.width), dtype=C64, device=self.device)
        gg = GG.Gategen(dim=self.d, device=self.device)
        opmat = {
            "I": gg.I.tensor,
            "Z": gg.Z.tensor,
            "X": gg.X.tensor,
        }
        for coeff, gtype, indices in self.hamiltonian:
            oplist = []
            if gtype == "Z":
                oplist = [
                    opmat["Z"] if i == indices[0] else opmat["I"]
                    for i in range(self.wires)
                ]
            elif gtype == "ZZ":
                oplist = [
                    opmat["Z"] if i in indices else opmat["I"]
                    for i in range(self.wires)
                ]
            else:
                raise NotImplementedError(f"Matrix for '{gtype}' not defined.")
            term = oplist[0]
            for k in range(1, len(oplist)):
                term = pt.kron(term, oplist[k])
            H_P += coeff * term

        return H_P

    def expectation(self):
        """
        Compute the expectation value of the Hamiltonian with respect to the current state.

        ExpVal $\langle\psi|H|\psi\\rangle$, where $|\psi\\rangle$ is the state obtained from the `forward` method, and H is the Hamiltonian matrix constructed in `getMat`. The offset is added to the computed expectation value to account for any constant terms in the Hamiltonian.
        """
        final_state = self.forward()
        exp_val = pt.vdot(
            final_state.squeeze(), (self._H_P_matrix @ final_state).squeeze()
        ).real

        return exp_val + self.offset

    def solve(
        self, func: Energy = None, optimizer=None, steps=100, lr=0.1, hook=devnull
    ):
        if optimizer is None:
            optimizer = pt.optim.Adam(self.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.expectation()
            loss.backward()
            optimizer.step()
            hook(loss, step)

        with pt.no_grad():
            final_state = self.forward()
            Pi = (pt.abs(final_state) ** 2).squeeze()
            maxP = pt.argmax(Pi).item()
            solution = format(maxP, f"0{self.wires}b")
            solution = [int(bit) for bit in solution]

        out = {
            "solution": solution,
            "probabilities": Pi.cpu().flatten(),
        }

        if func is not None and self.qubo is not None:
            soltensr = [pt.tensor(bit) for bit in solution]
            out["value"] = func(self.qubo, soltensr)

        return out


class ClockSolver(nn.Module):
    """
    Variational qudit optimizer using direct matrix exponentiation (clock/Potts model).

    Works for any local dimension $d \geq 2$. Each layer applies a phase separator
    $U_P(\gamma) = \exp(-i\gamma H_P)$ and a mixer $U_B(\\beta) = \exp(-i\\beta H_B)$
    where $H_B = -\sum_i(X_d^{(i)} + X_d^{(i)\dagger})$.

    Initial state: $H_d|0\\rangle^{\otimes n}$. Parameters $\gamma, \\beta$ are optimized with Adam.
    """

    d: int
    wires: int
    width: int
    layers: int
    device: str
    qubo: dict
    hamiltonian: list
    offset: float

    def __init__(
        self,
        d: int,
        wires: int,
        qubo: dict = None,
        hamiltonian: list = None,
        offset: float = 0.0,
        layers: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.d = d
        self.wires = wires
        self.width = d**wires
        self.layers = layers
        self.device = device
        self.qubo = qubo

        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
            self.offset = offset
        elif qubo is not None:
            self.hamiltonian, self.offset = QUBO.toHamiltonian(qubo)
        else:
            raise ValueError("Either 'hamiltonian' or 'qubo' must be provided.")

        self.gammas = nn.Parameter(pt.rand(layers, device=device) * (2 * pt.pi))
        self.betas = nn.Parameter(pt.rand(layers, device=device) * pt.pi)

        # Pre-compute static Hamiltonians (no grad needed)
        with pt.no_grad():
            self._H_P = self._buildHP()
            self._H_B = self._buildHB()

    def _buildHP(self) -> pt.Tensor:
        """
        Build phase Hamiltonian $H_P$ from Hamiltonian terms using $Z_d$ clock operators.
        """
        gg = GG.Gategen(dim=self.d, device=self.device)
        I = gg.I.tensor
        Z = gg.Z.tensor
        H_P = pt.zeros((self.width, self.width), dtype=C64, device=self.device)
        for coeff, gtype, indices in self.hamiltonian:
            if gtype == "Z":
                ops = [Z if i == indices[0] else I for i in range(self.wires)]
            elif gtype == "ZZ":
                ops = [Z if i in indices else I for i in range(self.wires)]
            else:
                raise NotImplementedError(f"Hamiltonian term '{gtype}' not supported.")
            term = ops[0]
            for op in ops[1:]:
                term = pt.kron(term, op)
            H_P = H_P + coeff * term
        H_P = (H_P + H_P.conj().T) / 2

        return H_P

    def _buildHB(self) -> pt.Tensor:
        """
        Build mixer Hamiltonian $H_B = -\sum_i(X_d^{(i)} + X_d^{(i)\dagger})$.
        """
        gg = GG.Gategen(dim=self.d, device=self.device)
        I = gg.I.tensor
        X = gg.X.tensor
        XpXdag = X + X.conj().T

        H_B = pt.zeros((self.width, self.width), dtype=C64, device=self.device)
        for i in range(self.wires):
            ops = [XpXdag if j == i else I for j in range(self.wires)]

            term = ops[0]
            for op in ops[1:]:
                term = pt.kron(term, op)

            H_B = H_B - term

        return H_B

    def forward(self) -> pt.Tensor:
        """
        Return the QAOA state $|\psi(\gamma, \\beta)\\rangle$ as a complex vector of length $d^n$.
        """
        gg = GG.Gategen(dim=self.d, device=self.device)
        ket0 = pt.zeros(self.d, dtype=C64, device=self.device)
        ket0[0] = 1.0
        plus = gg.H.tensor @ ket0

        state = plus
        for _ in range(self.wires - 1):
            state = pt.kron(state, plus)

        for i in range(self.layers):
            U_P = pt.linalg.matrix_exp(-1j * self.gammas[i].to(C64) * self._H_P)
            state = U_P @ state
            U_B = pt.linalg.matrix_exp(-1j * self.betas[i].to(C64) * self._H_B)
            state = U_B @ state

        return state

    def expectation(self) -> pt.Tensor:
        """
        Expectation value $\langle\psi|H_P|\psi\\rangle + \mathrm{offset}$.
        """
        state = self.forward()
        exp_val = pt.vdot(state, self._H_P @ state).real

        return exp_val + self.offset

    def _decode(self, idx: int) -> list:
        """
        Decode integer index to per-wire digit list (base-$d$, big-endian).
        """
        result = []
        for _ in range(self.wires):
            result.append(idx % self.d)
            idx //= self.d

        return result[::-1]

    def solve(
        self,
        func: "Energy | None" = None,
        optimizer=None,
        steps: int = 200,
        lr: float = 0.1,
        hook: "Callable" = devnull,
    ) -> dict:
        """
        Optimise gammas/betas using PyTorch Adam (or a provided optimizer).
        """
        if optimizer is None:
            optimizer = pt.optim.Adam(self.parameters(), lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.expectation()
            loss.backward()
            optimizer.step()
            hook(loss, step)

        with pt.no_grad():
            final_state = self.forward()
            Pi = pt.abs(final_state) ** 2
            maxP = int(pt.argmax(Pi).item())
            solution = self._decode(maxP)

        out: dict = {
            "solution": solution,
            "probabilities": Pi.cpu(),
        }
        if func is not None and self.qubo is not None:
            out["value"] = func(self.qubo, [pt.tensor(x) for x in solution])

        return out
