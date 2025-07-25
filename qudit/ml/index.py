from typing import List
import torch.nn as nn
import numpy as np
import torch

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'.


class Accel:
    allowed = ["cpu", "cuda", "mps", "openmp"]

    @staticmethod
    def _check(device: str):
        if device not in Accel.allowed:
            raise ValueError(
                f"Unsupported device. Allowed devices are: {Accel.allowed}"
            )

        if device == "cuda":
            return torch.cuda.is_available()
        elif device == "mps":
            return torch.backends.mps.is_available()
        elif device == "openmp":
            return torch.backends.openmp.is_available()
        elif device == "cpu":
            return True

        return True

    def check(devs: List[str]):
        print(f"Checking devices: {devs}")
        if isinstance(devs, str):
            return Accel._check(devs)

        return [Accel._check(dev) for dev in devs]

    def available():
        return [d for d in Accel.allowed if Accel._check(d)]

class Hybrid(nn.Module):
    def __init__(self, dim, wires, device="cpu"):
        super().__init__()
        self.dim = dim
        self.wires = wires
        self.circuit = None
        self.device = device

    def gate(self, *args, **kwargs):
        return self.circuit.gate(device=self.device, *args, **kwargs)

    def forward(self, x):
        return self.circuit(x)


class State(torch.Tensor):
    dev = "cpu"

    def __new__(cls, data, device=None):
        if device is not None:
            cls.dev = device

        if isinstance(data, (list, tuple)):
            data = torch.tensor(data, dtype=torch.complex64, device=cls.dev)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(dtype=torch.complex64, device=cls.dev)
        elif isinstance(data, torch.Tensor):
            data = data.to(dtype=torch.complex64, device=cls.dev)
        else:
            raise TypeError("Unsupported data type.")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        obj = super().__new__(cls, data)
        obj.dev = cls.dev
        return obj

    @property
    def device(self):
        return self.dev

    @staticmethod
    def ket_zeros(shape, device=None):
        state = torch.zeros(shape, dtype=torch.complex64, device=device)
        state[0] = 1.0
        return State(state, device=device)

    def __sub__(self, other):
        return torch.Tensor.__sub__(self, other)

    def __abs__(self):
        return torch.abs(self)
