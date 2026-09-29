"""Deterministic full-batch MNIST deep-autoencoder optimizer runner."""

from __future__ import annotations
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import torch
from softserve.benchmarks.deep_autoencoder import build_model, load_data
from softserve.benchmarks.rnn_adding import stochastic_qn_accounting
from softserve.optim import SoftServeKron
from softserve.secants import IntervalSecant
from softserve.storage import CampaignStore
from .runner import (
    NESTEROV_METHODS,
    QN_METHODS,
    _assign,
    _Controller,
    _gradient,
    _metrics,
    _state_bytes,
    _sync,
)

Method = Literal["softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap", "lbfgs"]
METHODS = ("softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap", "lbfgs")


@dataclass(frozen=True)
class DeterministicRunConfig:
    data_path: str
    method: Method
    seed: int
    learning_rate: float
    gradient_budget: int
    study: str = "unspecified"
    fixed_lambda: float | None = None
    refresh_interval: int = 10
    evaluation_interval: int = 50
    beta1: float = 0.95
    nesterov: bool = True
    fallback_learning_rate: float = 0.001
    device: str = "cuda"
    selected: bool = False
    qme_backend: str = "gemm"
    root_steps: int = 18
    inverse_steps: int = 10

    @classmethod
    def from_json(cls, path: str | Path) -> "DeterministicRunConfig":
        config = cls(**json.loads(Path(path).read_text()))
        config.validate()
        return config

    @property
    def parameter_steps(self) -> int | None:
        if self.method in QN_METHODS:
            return stochastic_qn_accounting(self.gradient_budget, self.refresh_interval)[0]
        if self.method == "lbfgs":
            return None
        return self.gradient_budget

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        if (self.method in QN_METHODS) != (self.fixed_lambda is not None):
            raise ValueError("fixed_lambda is defined exactly for SoftServe")
        if self.fixed_lambda is not None and (
            not math.isfinite(self.fixed_lambda) or self.fixed_lambda <= 0.0
        ):
            raise ValueError("fixed_lambda must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            min(
                self.gradient_budget,
                self.refresh_interval,
                self.evaluation_interval,
                self.root_steps,
                self.inverse_steps,
            )
            < 1
        ):
            raise ValueError("budgets, intervals, and iteration counts must be positive")
        if self.refresh_interval != 10:
            raise ValueError("the benchmark fixes K=10")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must lie in [0,1)")
        if self.method not in NESTEROV_METHODS and self.nesterov:
            raise ValueError(f"Nesterov is not defined for {self.method}")
        if self.method == "lbfgs" and self.learning_rate != 1.0:
            raise ValueError("the deterministic L-BFGS reference fixes LR=1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.method in QN_METHODS:
            stochastic_qn_accounting(self.gradient_budget, self.refresh_interval)


def _run_id(config: DeterministicRunConfig) -> str:
    payload = json.dumps(asdict(config), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    config: DeterministicRunConfig, campaign: str | None = None, worker_id: str | int = 0
) -> dict[str, Any]:
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    data = load_data(config.data_path, device, include_test=config.selected)
    model = build_model(config.seed, device)
    parameters = tuple(model.parameters())
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    controller = None if config.method == "lbfgs" else _Controller(model, config)
    lbfgs = (
        torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=20,
            max_eval=25,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        if config.method == "lbfgs"
        else None
    )
    if controller is not None:
        controller.train()
    route = parameters if controller is None else controller.main_parameters
    secant = (
        IntervalSecant(config.refresh_interval)
        if controller is not None and controller.qn is not None
        else None
    )
    history: list[dict[str, float | int | None]] = []
    calls = endpoints = curvature_updates = completed_steps = 0
    negative_pairs = pair_count = 0
    qme_residual_max = 0.0
    last_objective: float | None = None
    last_reconstruction: float | None = None
    next_evaluation = 0

    def record() -> None:
        if controller is not None:
            controller.eval()
        model.eval()
        objective, reconstruction_bce, reconstruction_mse = _metrics(
            model, data.train, include_regularization=True
        )
        _sync(device)
        history.append(
            {
                "parameter_steps": completed_steps,
                "gradient_evaluations": calls,
                "endpoint_evaluations": endpoints,
                "wall_seconds": time.perf_counter() - started,
                "training_batch_objective": last_objective,
                "training_batch_reconstruction_bce": last_reconstruction,
                "training_objective": objective,
                "training_reconstruction_bce": reconstruction_bce,
                "training_reconstruction_mse": reconstruction_mse,
            }
        )
        model.train()
        if controller is not None:
            controller.train()

    record()
    next_evaluation = config.evaluation_interval
    status, reason = ("complete", None)
    try:
        if lbfgs is not None:
            while calls < config.gradient_budget:
                remaining = config.gradient_budget - calls
                lbfgs.param_groups[0]["max_iter"] = min(20, remaining)
                lbfgs.param_groups[0]["max_eval"] = min(25, remaining)
                before = calls

                def closure() -> torch.Tensor:
                    nonlocal calls, last_objective, last_reconstruction
                    current = _gradient(model, data.train)
                    calls += 1
                    last_objective = current.objective
                    last_reconstruction = current.reconstruction_bce
                    if not math.isfinite(current.objective):
                        raise FloatingPointError("non-finite L-BFGS objective")
                    return torch.as_tensor(current.objective, device=device)

                lbfgs.step(closure)
                completed_steps += 1
                if calls == before:
                    raise RuntimeError("L-BFGS performed no closure evaluation")
                if calls >= next_evaluation or calls == config.gradient_budget:
                    record()
                    while next_evaluation <= calls:
                        next_evaluation += config.evaluation_interval
                if calls > config.gradient_budget:
                    raise RuntimeError("L-BFGS exceeded its closure-evaluation budget")
        else:
            assert controller is not None
            assert config.parameter_steps is not None
            for step in range(config.parameter_steps):
                current = _gradient(model, data.train)
                calls += 1
                last_objective = current.objective
                last_reconstruction = current.reconstruction_bce
                if not math.isfinite(current.objective):
                    raise FloatingPointError("non-finite training objective")
                _assign(parameters, current.values)
                if secant is not None and secant.starts_at(step):
                    route_gradients = tuple(
                        (current.values[parameter_indices[id(parameter)]] for parameter in route)
                    )
                    secant.start(route, route_gradients, batch_key=0)
                controller.parameter_step()
                if (
                    secant is not None
                    and secant.ends_at(step)
                    and (step + 1 < config.parameter_steps)
                ):
                    endpoint = _gradient(model, data.train)
                    calls += 1
                    endpoints += 1
                    endpoint_route = tuple(
                        (endpoint.values[parameter_indices[id(parameter)]] for parameter in route)
                    )
                    s_values, y_values = secant.pair(route, endpoint_route)
                    assert controller.qn is not None
                    controller.qn.update_curvature(s_values, y_values)
                    curvature_updates += 1
                    chi = [float(value) for value in controller.qn.last_pair_chi]
                    negative_pairs += sum((value < 0.0 for value in chi))
                    pair_count += len(chi)
                    if isinstance(controller.qn, SoftServeKron):
                        residuals = (
                            controller.qn.last_qme_residual_A + controller.qn.last_qme_residual_G
                        )
                        if residuals:
                            qme_residual_max = max(
                                qme_residual_max, max((float(value) for value in residuals))
                            )
                completed_steps = step + 1
                if calls >= next_evaluation or completed_steps == config.parameter_steps:
                    record()
                    while next_evaluation <= calls:
                        next_evaluation += config.evaluation_interval
                if calls > config.gradient_budget:
                    raise RuntimeError("gradient-call budget exceeded")
        if calls != config.gradient_budget:
            raise RuntimeError(f"used {calls} calls, expected {config.gradient_budget}")
    except (FloatingPointError, RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["parameter_steps"] != completed_steps:
        record()
    test_bce = test_mse = None
    if config.selected and status == "complete":
        if data.test is None:
            raise RuntimeError("selected run did not load the official test split")
        if controller is not None:
            controller.eval()
        model.eval()
        _, test_bce, test_mse = _metrics(model, data.test, include_regularization=False)
        _sync(device)
    tail = [float(point["training_objective"]) for point in history[-5:]]
    selection_score = (
        sum(tail) / len(tail)
        if status == "complete" and tail and all((math.isfinite(value) for value in tail))
        else None
    )
    optimizer = lbfgs if lbfgs is not None else controller
    assert optimizer is not None
    summary = {
        "run_id": _run_id(config),
        "problem": "mnist_deep_autoencoder",
        "mode": "deterministic",
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "lambda": config.fixed_lambda,
        "beta1": config.beta1 if config.method != "lbfgs" else None,
        "nesterov": config.nesterov if config.method in NESTEROV_METHODS else None,
        "qme_backend": config.qme_backend if config.method == "softserve_kron" else None,
        "refresh_interval": config.refresh_interval if config.method in QN_METHODS else None,
        "status": status,
        "reason": reason,
        "parameter_steps": completed_steps,
        "gradient_evaluations": calls,
        "endpoint_evaluations": endpoints,
        "curvature_updates": curvature_updates,
        "negative_secant_fraction": negative_pairs / pair_count if pair_count else 0.0,
        "qme_residual_max": qme_residual_max if config.method == "softserve_kron" else None,
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
    }
    if campaign is not None:
        CampaignStore(campaign).write_shard(worker_id, [summary])
    return summary
