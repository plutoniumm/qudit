from scipy import linalg as LA
import numpy.linalg as la
import numpy as np

MD = la.multi_dot


class Recovery:
    """
    Construct common approximate/analytic recovery maps for a code subspace.

    Each method returns a list of recovery Kraus operators $\{R_k\}$ intended to
    (approximately) invert a given error channel on the code projector $P$.
    """

    @staticmethod
    def leung(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Leung recovery via polar decomposition.

        Given code states $\{|\psi_i\\rangle\}$, form the projector
        $P=\sum_i |\psi_i\\rangle\langle\psi_i|$ and set
        $R_k = P\,U_k^\dagger$ where $U_k$ comes from the polar decomposition of $E_k P$.
        """
        P = sum([np.outer(state, state.conj().T) for state in codes])
        Rks = []
        for Ek in error_kraus:
            Uk, _ = LA.polar(np.dot(Ek, P), side="right")
            Rks.append(np.dot(P, Uk.conj().T))

        return Rks

    @staticmethod
    def cafaro(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Cafaro recovery using code-state normalizations.

        Builds $R_k$ as a sum over code basis states with coefficients normalized by
        $\sqrt{\langle\psi|E_k^\dagger E_k|\psi\\rangle}$.
        """
        Rks = []
        for Ek in error_kraus:
            Rks.append(
                sum(
                    [
                        np.dot(np.outer(state, state.conj().T), Ek.conj().T)
                        / np.sqrt(MD([state.conj().T, Ek.conj().T, Ek, state]))
                        for state in codes
                    ]
                )
            )
        return Rks

    @staticmethod
    def petz(kraus: list[np.ndarray], codes: list[np.ndarray]) -> list[np.ndarray]:
        """
        Petz recovery (transpose channel) restricted to the code.

        With code projector $P$, define $\mathcal{E}(P)=\sum_k E_k P E_k^\dagger$ and
        $\mathcal{R}_\\text{Petz}$ Kraus operators $R_k = P E_k^\dagger\,\mathcal{E}(P)^{-1/2}$.
        """
        P = sum([np.outer(state, state.conj().T) for state in codes])
        channel = sum([MD([Ek, P, Ek.conj().T]) for Ek in kraus])
        norm = LA.fractional_matrix_power(channel, -0.5)

        return [MD([P, Ek.conj().T, norm]) for Ek in kraus]

    @staticmethod
    def dutta(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Dutta recovery for ensembles of error operators.

        Expects a list of error-sets (each set may carry probabilities) and produces
        normalized recovery operators by averaging syndrome overlaps on the code.
        """
        Rks = []
        for Eks in error_kraus:
            Rk = []
            for i in codes:
                chis = []
                for En in Eks:
                    chis.append(
                        sum([MD([i.conj().T, Em.conj().T, En, i]) for Em in Eks])
                    )
                X_av = np.average(chis, weights=[Eks[j].P for j in range(len(chis))])

                Rk.append(
                    sum([np.outer(i, np.dot(Em, i).conj().T) for Em in Eks]) / X_av
                )

            Rk = np.sum(Rk, axis=0)
            Rks.append(Rk / np.sqrt(np.linalg.eigvalsh(np.dot(Rk.conj().T, Rk))[-1]))

        return Rks
