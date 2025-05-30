import numpy as np

class GeneralizedQuditGates:
    def __init__(self, d):
        self.d = d
        self._omega = np.exp(2j * np.pi / d)

    @property
    def omega(self):
        return self._omega

    @property
    def x(self):
        X = np.zeros((self.d, self.d), dtype=complex)
        for i in range(self.d):
            X[(i + 1) % self.d, i] = 1.0
        return X

    @property
    def z(self):
        return np.diag([self.omega**j for j in range(self.d)])

    @property
    def s(self):
        omega_s = np.exp(2j * np.pi / (2 * self.d))
        return np.diag([omega_s**j for j in range(self.d)])

    @property
    def t(self):
        omega_t = np.exp(2j * np.pi / (4 * self.d))
        return np.diag([omega_t**j for j in range(self.d)])

    @property
    def h(self):
        H = np.array([[self.omega**(j * k) for k in range(self.d)] for j in range(self.d)], dtype=complex)
        return H / np.sqrt(self.d)

    @property
    def y(self):
        return 1j * (self.x - self.x.conj().T)

    def p(self, theta):
        omega_p = np.exp(1j * theta * np.pi / self.d)
        return np.diag([omega_p**j for j in range(self.d)])

    def c_gate(self, U, control_state=1):
        CU = np.zeros((self.d**2, self.d**2), dtype=complex)
        I = np.eye(self.d, dtype=complex)
        for i in range(self.d):
            proj = np.zeros((self.d, self.d), dtype=complex)
            proj[i, i] = 1
            CU += np.kron(proj, U if i == control_state else I)
        return CU
