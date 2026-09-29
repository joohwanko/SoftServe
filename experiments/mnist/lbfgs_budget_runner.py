"""Exact closure-budget L-BFGS runner for the deterministic autoencoder campaign."""

from __future__ import annotations
import math
import os
import time
from dataclasses import asdict
import torch
from softserve.storage import results_root
from softserve.benchmarks.deep_autoencoder import build_model, load_data
from softserve.storage import CampaignStore
from .deterministic_runner import DeterministicRunConfig, _run_id
from .runner import _gradient, _metrics, _state_bytes, _sync


class _BudgetReached(Exception):
    pass


def run_exact_lbfgs(
    config: DeterministicRunConfig, campaign: str, worker_id: str
) -> dict[str, object]:
    config.validate()
    if config.method != "lbfgs":
        raise ValueError("this runner accepts only L-BFGS configs")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    data = load_data(config.data_path, device, include_test=config.selected)
    model = build_model(config.seed, device)
    parameters = tuple(model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=20,
        max_eval=25,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    history: list[dict[str, float | int | None]] = []
    calls = outer_steps = 0
    last_objective: float | None = None
    last_reconstruction: float | None = None
    last_evaluated = [parameter.detach().clone() for parameter in parameters]

    def record() -> None:
        model.eval()
        objective, reconstruction_bce, reconstruction_mse = _metrics(
            model, data.train, include_regularization=True
        )
        _sync(device)
        history.append(
            {
                "parameter_steps": outer_steps,
                "gradient_evaluations": calls,
                "endpoint_evaluations": 0,
                "wall_seconds": time.perf_counter() - started,
                "training_batch_objective": last_objective,
                "training_batch_reconstruction_bce": last_reconstruction,
                "training_objective": objective,
                "training_reconstruction_bce": reconstruction_bce,
                "training_reconstruction_mse": reconstruction_mse,
            }
        )
        model.train()

    record()
    next_evaluation = config.evaluation_interval
    status, reason = ("complete", None)
    try:
        while calls < config.gradient_budget:

            def closure() -> torch.Tensor:
                nonlocal calls, last_objective, last_reconstruction, last_evaluated
                if calls >= config.gradient_budget:
                    with torch.no_grad():
                        for parameter, value in zip(parameters, last_evaluated, strict=True):
                            parameter.copy_(value)
                    raise _BudgetReached
                current = _gradient(model, data.train)
                calls += 1
                last_objective = current.objective
                last_reconstruction = current.reconstruction_bce
                if not math.isfinite(current.objective):
                    raise FloatingPointError("non-finite L-BFGS objective")
                last_evaluated = [parameter.detach().clone() for parameter in parameters]
                return torch.as_tensor(current.objective, device=device)

            try:
                optimizer.step(closure)
            except _BudgetReached:
                pass
            outer_steps += 1
            if calls >= next_evaluation or calls == config.gradient_budget:
                record()
                while next_evaluation <= calls:
                    next_evaluation += config.evaluation_interval
        if calls != config.gradient_budget:
            raise RuntimeError(f"used {calls} closures, expected {config.gradient_budget}")
    except (FloatingPointError, RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["gradient_evaluations"] != calls:
        record()
    test_bce = test_mse = None
    if config.selected and status == "complete":
        if data.test is None:
            raise RuntimeError("selected run did not load the official test split")
        model.eval()
        _, test_bce, test_mse = _metrics(model, data.test, include_regularization=False)
        _sync(device)
    tail = [float(point["training_objective"]) for point in history[-5:]]
    selection_score = (
        sum(tail) / len(tail)
        if status == "complete" and tail and all((math.isfinite(value) for value in tail))
        else None
    )
    row: dict[str, object] = {
        "run_id": _run_id(config),
        "problem": "mnist_deep_autoencoder",
        "mode": "deterministic",
        "method": "lbfgs",
        "seed": config.seed,
        "learning_rate": 1.0,
        "lambda": None,
        "beta1": None,
        "nesterov": None,
        "qme_backend": None,
        "refresh_interval": None,
        "status": status,
        "reason": reason,
        "parameter_steps": outer_steps,
        "gradient_evaluations": calls,
        "endpoint_evaluations": 0,
        "curvature_updates": 0,
        "negative_secant_fraction": 0.0,
        "qme_residual_max": None,
        "selection_score": selection_score,
        "test_reconstruction_bce": test_bce,
        "test_reconstruction_mse": test_mse,
        "wall_seconds": time.perf_counter() - started,
        "optimizer_state_bytes": _state_bytes(optimizer.state_dict()),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "torch_version": torch.__version__,
        "config": asdict(config),
        "history": history,
        "implementation_patch": "exact-lbfgs-closure-budget-v2",
    }
    CampaignStore(campaign, root=results_root()).write_shard(worker_id, [row])
    return row
