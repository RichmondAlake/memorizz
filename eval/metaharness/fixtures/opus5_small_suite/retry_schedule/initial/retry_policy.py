def delays(attempts: int, base: float = 0.1, cap: float = 1.0):
    return [min(cap, base * (2**index)) for index in range(attempts + 1)]
