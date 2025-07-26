from typing import List
import torch.nn as nn
import numpy as np
import torch

Hybrid = nn.Module


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
