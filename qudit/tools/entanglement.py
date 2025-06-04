from scipy.optimize import minimize
import numpy as np

ROUNDOFF_TOL = 1e-6


def Perp(basis):
    if not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    projector = sum(
        np.array([np.outer(basis[i], basis[i].conj().T) for i in range(len(basis))])
    )

    return np.eye(len(basis[0])) - projector


def Loss(F, projector):
    prod = F.H.dot(projector).dot(F)

    return np.real_if_close(prod, tol=ROUNDOFF_TOL)


def rank(f, D, r, **kwargs):
    if "method" not in kwargs:
        kwargs["method"] = "Powell"

    if "tol" not in kwargs:
        kwargs["tol"] = ROUNDOFF_TOL

    tries = kwargs.get("tries", 1)
    rmax = kwargs.get("r_max", 2**7)
    size = (2 * D + 1) * (r - 1)

    if "tries" in kwargs:
        del kwargs["tries"]

    minimas = np.ones(tries)
    for i in range(tries):
        try:
            minimas[i] = minimize(
                f, x0=np.random.uniform(-rmax, rmax, size), **kwargs
            ).fun
        except Exception as e:
            pass

    return np.min(minimas)
