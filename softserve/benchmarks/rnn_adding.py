CURVATURE_FREQUENCY = 10


def stochastic_qn_accounting(budget: int, frequency: int = CURVATURE_FREQUENCY) -> tuple[int, int]:
    for steps in range(budget, 0, -1):
        endpoints = (steps - 1) // frequency
        if steps + endpoints == budget:
            return (steps, endpoints)
    raise ValueError("budget is incompatible with refresh frequency")
