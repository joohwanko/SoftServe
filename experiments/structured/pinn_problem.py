"""Use the existing experiment's problem factory, not benchmark defaults."""

from dataclasses import asdict


def reference_problem(config):
    from experiments.pinn.runner import PINNRunConfig

    return PINNRunConfig(
        pde=config["pde"],
        mode=config.get("mode", "deterministic"),
        method="adam",
        seed=config.get("seed", 0),
        lr=config.get("learning_rate", 0.001),
        width=config.get("width", 200),
        layers=config.get("layers", 4),
        num_x=config.get("num_x", 257),
        num_t=config.get("num_t", 101),
        residual_batch=config.get("residual_batch", 10000),
    ).problem()


def resolved_problem(config):
    return asdict(reference_problem(config))
