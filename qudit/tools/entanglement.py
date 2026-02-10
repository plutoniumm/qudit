from scipy.optimize import minimize
import numpy as np

ROUNDOFF_TOL = 1e-6


def Perp(basis: "np.ndarray | list") -> np.ndarray:
    """
    Return the orthogonal projector onto the complement of a basis span.

    Given a list/array of vectors $\{ |b_i\\rangle \}$, forms
    $P = \sum_i |b_i\\rangle\langle b_i|$ and returns $I - P$.
    """
    if not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    projector = sum(
        np.array([np.outer(basis[i], basis[i].conj().T) for i in range(len(basis))])
    )

    return np.eye(len(basis[0])) - projector


def Loss(F: "np.matrix | np.ndarray", projector: np.ndarray) -> np.ndarray:
    """
    Quadratic form $\langle F | \Pi | F \\rangle$ used as an objective.

    Computes $F^\dagger \Pi F$ (with $\Pi$ typically a projector).

    > [!NOTE]
    > for `np.ndarray`, `.H` may not exist; callers typically pass an `np.matrix`.
    """
    prod = F.H.dot(projector).dot(F)  # type: ignore[attr-defined]

    return np.real_if_close(prod, tol=ROUNDOFF_TOL)


def rank(f: object, D: int, r: int, **kwargs: object) -> float:
    """
    Multi-start minimization over a rank-parameterized family.

    Runs `scipy.optimize.minimize` from several random initial points and returns the
    smallest objective value found.
    """
    kwargs_any = dict(kwargs)  # type: ignore[arg-type]

    if "method" not in kwargs_any:
        kwargs_any["method"] = "Powell"

    if "tol" not in kwargs_any:
        kwargs_any["tol"] = ROUNDOFF_TOL

    tries = int(kwargs_any.get("tries", 1))
    rmax = float(kwargs_any.get("r_max", 2**7))
    size = int((2 * D + 1) * (r - 1))

    if "tries" in kwargs_any:
        del kwargs_any["tries"]

    minimas = np.ones(tries)
    for i in range(tries):
        try:
            minimas[i] = minimize(
                f, x0=np.random.uniform(-rmax, rmax, size), **kwargs_any
            ).fun
        except Exception:
            pass

    return float(np.min(minimas))
