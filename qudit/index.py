from typing import Any, List
import numpy.linalg as LA
import numpy as np

"""
  B = Basis(3) will create a qutrit basis
  so B("111") will return State for |111>
  or B(1, 2, 0) will return State for |120>
"""


class Basis:
    """
    Computational basis factory for qudits of local dimension $d$.

    Calling `Basis(d)(k_1,\\n\dots,\\n k_n)` returns the tensor-product basis ket
    $|k_1\\rangle \otimes \cdots \otimes |k_n\\rangle$ as a normalized `State`.
    """

    d: int
    span: int = -1

    def __init__(self, d: int):
        self.d = d

    def __call__(self, *args: Any) -> "State":
        """
        Build a computational basis ket from indices.

        If a single string is provided (e.g. "120"), it is parsed digit-wise.
        Returns $|k_1\\rangle \otimes \cdots \otimes |k_n\\rangle$.
        """
        if len(args) == 1 and isinstance(args[0], str):
            idxs: List[int] = [int(i) for i in args[0]]
        else:
            idxs = [int(k) for k in args]

        basis = np.eye(self.d, dtype=np.complex128)
        prod: np.ndarray = np.array([1.0 + 0.0j], dtype=np.complex128)
        for ket in idxs:
            if ket < 0 or ket >= self.d:
                raise ValueError(f"Index {ket} out of bounds for dimension {self.d}")
            prod = np.kron(prod, basis[int(ket)])

        return State(prod)


class State(np.ndarray):
    """
    Quantum state container backed by a NumPy array.

    - Pure state: a 1D normalized vector $|\psi\\rangle$ with $\langle\psi|\psi\\rangle = 1$.
    - Mixed state: a 2D density matrix $\\rho$ with $\mathrm{Tr}(\\rho)=1$.

    Exposes basic operations (dagger, trace, tensor product, projectors).
    """

    # Avoid overriding ndarray's own attribute stubs; keep only loose extras.

    def __new__(cls, d: Any):
        """
        Create and normalize a statevector or density matrix.

        - Vector: $|\psi\\rangle / |\psi|_2$.
        - Density: $\\rho / \mathrm{Tr}(\\rho)$.
        """
        arr = np.asarray(d, dtype=np.complex128)
        if arr.ndim == 1:
            arr /= np.linalg.norm(arr)
        elif arr.ndim == 2:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("Density matrix must be square")
            arr /= np.trace(arr).real
        else:
            raise ValueError("Input must be 1D (vector) or 2D (density matrix)")

        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return

    @property
    def isDensity(self) -> bool:
        """
        True iff this object is a density matrix $\\rho$ (i.e. 2D).
        """
        return self.ndim == 2

    def isPure(self) -> bool:
        """
        For densities, checks $\mathrm{Tr}(\\rho^2)=1$.
        For vectors, returns True.
        """
        if not self.isDensity:
            return True
        tr = np.trace(self**2).real
        return bool(np.isclose(tr, 1.0))

    @property
    def d(self) -> int:
        """
        Hilbert space dimension (vector length or matrix size).
        """
        return int(self.shape[0])

    def norm(self) -> "State":
        """
        Return a re-normalized copy: $|\psi\\rangle/\|\psi\|_2$ or $\\rho/\mathrm{Tr}(\\rho)$.
        """
        if not self.isDensity:
            return State(self / np.linalg.norm(self))
        return State(self / self.tr().real)

    def density(self) -> "State":
        """
        Return density operator.

        Vector: $\\rho = |\psi\\rangle\langle\psi|$; density: returns itself.
        """
        if not self.isDensity:
            return State(np.outer(self, self.conj()))
        return self

    @property
    def H(self) -> "State":
        return State(self.conj().T)

    def kron(self, other: Any) -> "State":
        return State(np.kron(self, other))

    def __xor__(self, other: Any) -> "State":
        return self.kron(other)

    def tr(self) -> complex:
        """
        Vector: returns $\langle\psi|\psi\\rangle$; density: returns $\mathrm{Tr}(\\rho)$.
        """
        if not self.isDensity:
            return complex(np.vdot(self, self))
        return complex(np.trace(self))

    def proj(self) -> "State":
        """
        Vector: $|\psi\\rangle\langle\psi|$.
        Density: $\sum_{\lambda_i>0} |v_i\\rangle\langle v_i|$ using eigendecomposition.
        """
        if not self.isDensity:
            return self.density()

        evals, evecs = LA.eig(self)
        matrix: np.ndarray = np.zeros_like(self, dtype=np.complex128)
        for i in range(len(evals)):
            if np.abs(evals[i]) > 1e-8:
                matrix += np.outer(evecs[:, i], evecs[:, i].conj())
        return State(matrix)

    def oproj(self) -> "State":
        """
        Orthogonal complement projector: $I - P$ where $P=\mathrm{proj}(\\rho)$.
        """
        proj = self.proj()
        perp = np.eye(proj.shape[0], dtype=np.complex128) - proj
        return State(perp)
