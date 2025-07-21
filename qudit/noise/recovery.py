from scipy import linalg as LA
import numpy.linalg as la
import numpy as np

MD = la.multi_dot


class Recovery:
    # Ak
    def leung(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
        P = sum([np.outer(state, state.conj().T) for state in codes])
        Rks = []
        for Ek in error_kraus:
            Uk, _ = LA.polar(np.dot(Ek, P), side="right")
            Rks.append(np.dot(P, Uk.conj().T))

        return Rks

    # Ak
    def cafaro(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
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

    # Ak_
    def petz(kraus: list[np.ndarray], codes: list[np.ndarray]) -> list[np.ndarray]:
        P = sum([np.outer(state, state.conj().T) for state in codes])
        channel = sum([MD([Ek, P, Ek.conj().T]) for Ek in kraus])
        norm = LA.fractional_matrix_power(channel, -0.5)

        return [MD([P, Ek.conj().T, norm]) for Ek in kraus]

    def dutta(
        error_kraus: list[np.ndarray], codes: list[np.ndarray]
    ) -> list[np.ndarray]:
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
