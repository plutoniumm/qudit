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

        ```py
        # (coefficient, gate_type, indices)
        type Term = tuple[float, str, list[int]]
        ```
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
        return self.gate(x, GG.RZ, index=indices, angle=angle)

    def _RZZ(self, x, angle, indices):
        x = self.gate(x, GG.CX, index=indices)
        x = self.gate(x, GG.RZ, index=[indices[1]], angle=angle)
        x = self.gate(x, GG.CX, index=indices)

        return x

    def forward(self):
        state = pt.zeros((self.width, 1), dtype=C64, device=self.device)
        state[0, 0] = 1.0
        h_all = GG.H(
            dim=self.d,
            wires=self.wires,
            index=list(range(self.wires)),
            device=self.device,
        )

        state = h_all.forward(state)
        for i in range(self.layers):
            for coeff, gtype, indices in self.hamiltonian:
                angle = 2 * self.gammas[i] * coeff
                if gtype in self.OpMap:
                    state = self.OpMap[gtype](state, angle, indices)
                else:
                    raise NotImplementedError(f"Gate type '{gtype}' not supported.")

            for j in range(self.wires):
                state = self.gate(state, GG.RX, index=[j], angle=2 * self.betas[i])

        return state

    def getMat(self):
        """
        Construct Hamiltonian matrix from `self.hamiltonian`.
        """

        H_P = pt.zeros((self.width, self.width), dtype=C64, device=self.device)
        gg = GG.Gategen(dim=self.d, device=self.device)
        opmat = {
            "I": gg.I,
            "Z": gg.Z,
            "X": gg.X,
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

    def solve(self, func: Energy, optimizer=None, steps=100, lr=0.1, hook=devnull):
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
            soltensr = [pt.tensor(bit) for bit in solution]

            min_energy = func(self.qubo, soltensr)

        return {
            "solution": solution,
            "value": min_energy,
            "probabilities": Pi.cpu().flatten(),
        }
