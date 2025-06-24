from functools import cached_property, lru_cache
from ..index import Gate, State
from typing import Any, Union
import numpy as np


class Params:
    list: dict[str, float]

    def __init__(self, params: dict[str, Any]):
        self.list = params

    def __getitem__(self, key: str) -> float:
        return self.list[key]

    def __setitem__(self, key: str, value: float):
        self.list[key] = value

    def __iter__(self):
        return iter(self.list.items())

    def __contains__(self, key: str) -> bool:
        return key in self.list


class Error(Gate):
    params: Params

    def __new__(cls, d: int, O: np.ndarray = None, name: str = "Err", params={}):
        obj = super().__new__(cls, d, O, name)
        obj.params = Params(params)
        return obj


class Channel:
    operators: list[Error]

    def __init__(self, ops: list[Error]):
        self.operators = ops

    @lru_cache
    def run(self, rho: Union[State, np.ndarray]) -> np.ndarray:
        result = [O @ rho @ O.conj().T for O in self.operators]

        return sum(result)

    @cached_property
    def isTP(self) -> bool:
        I = np.eye(self.operators[0].shape[0], dtype=np.complex128)
        result = self.run(I)
        result = np.allclose(result, I, atol=1e-5)

        return result

    @cached_property
    def isCP(self) -> bool:
        I = np.eye(self.operators[0].shape[0], dtype=np.complex128)
        result = self.run(I)
        result = np.all(np.linalg.eigvals(result) >= 0)

        return result

    @cached_property
    def isCPTP(self) -> bool:
        return self.isCP and self.isTP
