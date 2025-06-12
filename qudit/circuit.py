from typing import List, Union, Callable
from sympy import SparseMatrix as Matrix
from .index import Gate, VarGate
from scipy import sparse as S
from .utils import Tensor, ID
from .gates import Gategen
import numpy as np

BARRIER = "─|─"


class Layer:
    vqc: bool = False
    data: np.ndarray
    counter: List[int]
    gates: List[Gate]
    span: int
    id: str
    d: int

    def __init__(self, size: int):
        assert size > 0, f"Size must be a >0, got: {size}"

        self.id = ID()
        self.span = size
        self.counter = list(range(size))
        self.gates = []
        self.d = -1

    def add(self, gate: Union[Gate, VarGate], dits: List[int]):
        gate.dits = dits
        self.vqc = self.vqc or gate.vqc

        for d in dits:
            if d in self.counter:
                self.counter.remove(d)

        self.gates.append(gate)
        return self

    @property
    def available(self):
        return [d for d in range(self.span) if d in self.counter]

    def open(self, *args: List[int]) -> bool:
        return all(d in self.available for d in args)

    def finalise(self):
        if self.d == -1:
            if len(self.gates) > 0:
                self.d = self.gates[0].d
            else:
                raise ValueError("Dimension not set, add a gate first")

        sublayer = self.getMat(self.gates)
        prod = sublayer[0]
        for sub in sublayer[1:]:
            prod = prod @ sub

        self.data = prod

    # return list of equal sized matrices
    def getMat(self, in_gates, compress=False) -> List[np.ndarray]:
        sublayer = [[]]
        l_gates, s_gates = [], []

        for gate in in_gates:
            if gate.span == 2:
                l_gates.append(gate)
            elif gate.span == 1:
                s_gates.append(gate)
            elif gate.span == 0:
                continue
            else:
                raise ValueError(f"Span > 2 not supported: {gate.span}")
        # endfor

        G = Gategen(self.d)
        I = G.I

        sublayer[0] = [I] * self.span
        for gate in s_gates:
            dit = gate.dits[0]
            sublayer[0][dit] = gate
        sublayer[0] = Tensor(*sublayer[0])
        # endfor

        for gate in l_gates:
            a, b = gate.dits
            name = gate.name if gate.name else f"?({a}, {b})"

            # RUN SWAPS FOR NON CONSECUTIVE DITS
            if a != 0 and b != 1:
                swap_a = G.long_swap(a, 0, width=self.span)
                swap_b = G.long_swap(b, 1, width=self.span)
                swap = swap_a @ swap_b
            elif a != 0 and b == 1:
                swap = G.long_swap(a, 0, width=self.span)
            elif b != 1 and a == 0:
                swap = G.long_swap(b, 1, width=self.span)
            else:  # a == 0 and b == 1
                swap = np.eye(self.d**self.span)

            temp = [gate] + [I] * (self.span - 2)
            temp = Tensor(*temp)
            temp = swap @ temp @ swap
            temp.name = name
            sublayer.append(temp)

        return sublayer

    def __repr__(self):
        names = [gate.name for gate in self.gates]
        return f"Layer({', '.join(names)})"

    def __getitem__(self, index):
        return self.gates[index]

    def __iter__(self):
        return iter(self.gates)


class cfn:
    def balance(strings: List[str]) -> List[str]:
        lmax = max(len(s) for s in strings)
        return [s.ljust(lmax, "─") for s in strings]

    def cx(strings: List[str], dits: List[int], name: str = "U") -> List[str]:
        ctrl, targ = dits
        name = name[1:] if name.startswith("C") else name

        if ctrl > targ:
            strings[targ] += f"╭{name}─"
            strings[ctrl] += "╰●─"
            scan = range(targ + 1, ctrl)
        else:
            strings[ctrl] += "╭●─"
            strings[targ] += f"╰{name}─"
            scan = range(ctrl + 1, targ)

        for i in scan:
            strings[i] += "│─"

        return strings


class Circuit:
    layers: List[Layer]
    vqc: bool = False
    span: int
    id: str
    d: int

    def __init__(self, size: int = 0):
        assert size >= 0, "Size must be a non-negative integer"

        self.layers = [Layer(size=size)]
        self.d = -1
        self.span = size
        self.id = ID()

    def gate(self, gate: Union[Gate, VarGate], dits: List[int]):
        layer = self.layers[-1]
        if not layer.open(*dits):
            layer.finalise()
            if layer.d != -1:
                self.d = layer.d
            layer = Layer(size=self.span)
            self.layers.append(layer)

        layer.add(gate, dits)
        return self

    def solve(self) -> np.ndarray:
        self._refresh()
        for layer in self.layers:
            if not hasattr(layer, "data"):
                layer.finalise()

        for i in range(len(self.layers)):
            if self.vqc:
                self.layers[i].data = Matrix(self.layers[i].data)
            else:
                self.layers[i].data = S.csr_matrix(self.layers[i].data)

        prod = self.layers[0].data
        for m in self.layers[1:]:
            prod = m.data @ prod

        return prod

    def draw(self):
        qudits = self.layers[0].span

        strings = ["─"] * qudits
        for l, layer in enumerate(self.layers):
            qctr = 0
            if layer[0].name == BARRIER:
                strings = cfn.balance(strings)
                strings = [s + BARRIER for s in strings]
                continue

            for gate in layer:
                if gate.span == 2:
                    strings = cfn.balance(strings)
                    strings = cfn.cx(strings, gate.dits, gate.name)
                    qctr += 2
                else:
                    g = gate.dits[0]
                    if gate.name == "I" or gate.name == "_":
                        strings[g] += "──"
                    else:
                        strings[g] += f"{gate.name}─"
                    qctr += 1
            # endfor
            strings = cfn.balance(strings)
        # endfor

        return "\n".join(strings)

    def __repr__(self):
        return self.draw()

    def __getitem__(self, index):
        return self.layers[index]

    def __setitem__(self, index: int, value: Layer):
        if not isinstance(value, Layer):
            raise TypeError(f"Expected Layer, got {type(value)}")
        if index < 0 or index >= len(self.layers):
            raise IndexError(
                f"Expected index in [0, {len(self.layers) - 1}], got {index}"
            )
        self.layers[index] = value
        self._refresh()

    def __iter__(self):
        return iter(self.layers)

    def _refresh(self):
        if self.d == -1:
            for layer in self.layers:
                if layer.d > 0:
                    self.d = layer.d
                    break

        if self.span == -1:
            span_sum = 0
            for layer in self.layers:
                if layer.span > 0:
                    span_sum += layer.span

            if span_sum > 0:
                self.span = span_sum

        if self.vqc is False:
            for layer in self.layers:
                if layer.vqc:
                    self.vqc = True
                    break

        if not self.id:
            self.id = ID()

    def barrier(self):
        if len(self.layers) < 1:
            raise ValueError("Add at least 1 layer for a barrier")
        assert self.d > 0, "Dimension Unknown, add a layer first"
        assert self.span > 0, "Span Unknown, add a layer first"

        d = self.d
        layer = Layer(size=self.span).add(
            Gate(d, np.eye(d), BARRIER), dits=list(range(self.span))
        )
        self.layers.append(layer)

        self._refresh()
        return self
