class Channel:
    Ek: list

    def __init__(self, Ek):
        self.Ek = Ek

    def __call__(self, state):
        if not isinstance(state, State):
            raise TypeError("Input must be a State instance")

        result = np.zeros_like(state, dtype=np.complex128)
        for k in self.Ek:
            result += k @ state @ k.conj().T
        return State(result)

    # @property
    # isCP(self) -> bool:
    #     pass

    # @property
    # isTP(self) -> bool:
    #     pass

    # @property
    # isCPTP(self) -> bool:
    #     return self.isCP and self.isTP
