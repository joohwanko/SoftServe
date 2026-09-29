"""K-FAC and K-BFGS(L) runner for the MNIST deep autoencoder."""

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
from torch.nn import functional as F
from softserve.baselines_kron_qn import KFAC, KBFGSL, homogeneous, homogeneous_gradients
from softserve.benchmarks.deep_autoencoder import (
    DeepAutoencoder,
    build_model,
    load_data,
    stateless_epoch_permutation,
)
from softserve.storage import CampaignStore
from .runner import L2_COEFFICIENT, _metrics, _state_bytes, _sync

Method = Literal["kfac", "kbfgs_l"]
Mode = Literal["deterministic", "stochastic"]
METHODS = ("kfac", "kbfgs_l")
MODES = ("deterministic", "stochastic")
DEFAULT_DAMPING = {"kfac": 3.0, "kbfgs_l": 0.3}


@dataclass(frozen=True)
class StructuredRunConfig:
    data_path: str
    method: Method
    mode: Mode
    seed: int
    learning_rate: float
    gradient_budget: int
    study: str = "unspecified"
    damping: float | None = None
    batch_size: int = 1000
    evaluation_interval: int = 50
    device: str = "cuda"
    selected: bool = False
    factor_decay: float = 0.9
    momentum: float = 0.9
    inverse_update_frequency: int = 20
    history_size: int = 100

    @classmethod
    def from_json(cls, path: str | Path) -> "StructuredRunConfig":
        config = cls(**json.loads(Path(path).read_text()))
        config.validate()
        return config

    @property
    def resolved_damping(self) -> float:
        return DEFAULT_DAMPING[self.method] if self.damping is None else self.damping

    @property
    def initialization_gradient_evaluations(self) -> int:
        if self.method == "kfac" and self.mode == "stochastic":
            return 60000 // self.batch_size
        return 0

    @property
    def parameter_steps(self) -> int:
        remaining = self.gradient_budget - self.initialization_gradient_evaluations
        if remaining < 2 or remaining % 2:
            raise ValueError("structured gradient budget must leave an even training budget")
        return remaining // 2

    def validate(self) -> None:
        if self.method not in METHODS or self.mode not in MODES:
            raise ValueError("unsupported structured method or mode")
        if not self.study:
            raise ValueError("study must be non-empty")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.resolved_damping) or self.resolved_damping <= 0.0:
            raise ValueError("damping must be finite and positive")
        if self.batch_size != 1000:
            raise ValueError("the stochastic benchmark fixes batch size 1000")
        if (
            min(
                self.gradient_budget,
                self.evaluation_interval,
                self.inverse_update_frequency,
                self.history_size,
            )
            < 1
        ):
            raise ValueError("budgets and frequencies must be positive")
        if not 0.0 <= self.factor_decay < 1.0 or not 0.0 <= self.momentum < 1.0:
            raise ValueError("decay and momentum must lie in [0,1)")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        _ = self.parameter_steps


@dataclass
class _Backward:
    objective: float
    reconstruction: float
    inputs: list[torch.Tensor]
    preactivations: list[torch.Tensor]
    output_gradients: list[torch.Tensor]
    parameter_gradients: list[torch.Tensor]


def _forward_with_caches(
    model: DeepAutoencoder, values: torch.Tensor, *, retain_grad: bool = True
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    inputs = []
    preactivations = []
    hidden = values
    for index, layer in enumerate(model.layers):
        inputs.append(hidden)
        preactivation = layer(hidden)
        if retain_grad:
            preactivation.retain_grad()
        preactivations.append(preactivation)
        hidden = (
            torch.relu(preactivation) if index not in {3, len(model.layers) - 1} else preactivation
        )
    return (hidden, inputs, preactivations)


def _real_backward(model: DeepAutoencoder, values: torch.Tensor) -> _Backward:
    model.zero_grad(set_to_none=True)
    logits, inputs, preactivations = _forward_with_caches(model, values)
    reconstruction = (
        F.binary_cross_entropy_with_logits(logits, values, reduction="none").sum(dim=1).mean()
    )
    penalty = torch.stack(
        [parameter.float().square().sum() for parameter in model.parameters()]
    ).sum()
    objective = reconstruction + 0.5 * L2_COEFFICIENT * penalty
    objective.backward()
    batch_size = len(values)
    output_gradients = []
    for preactivation in preactivations:
        if preactivation.grad is None:
            raise RuntimeError("missing retained preactivation gradient")
        output_gradients.append((batch_size * preactivation.grad).detach().clone())
    return _Backward(
        objective=float(objective.detach()),
        reconstruction=float(reconstruction.detach()),
        inputs=[value.detach() for value in inputs],
        preactivations=[value.detach() for value in preactivations],
        output_gradients=output_gradients,
        parameter_gradients=[
            value.detach().clone() for value in homogeneous_gradients(model.layers)
        ],
    )


def _auxiliary_backward(model: DeepAutoencoder, values: torch.Tensor, *, fisher: bool) -> _Backward:
    model.zero_grad(set_to_none=True)
    logits, inputs, preactivations = _forward_with_caches(model, values)
    targets = torch.bernoulli(torch.sigmoid(logits.detach())) if fisher else values
    reconstruction = (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(dim=1).mean()
    )
    reconstruction.backward()
    batch_size = len(values)
    output_gradients = []
    for preactivation in preactivations:
        if preactivation.grad is None:
            raise RuntimeError("missing retained preactivation gradient")
        output_gradients.append((batch_size * preactivation.grad).detach().clone())
    return _Backward(
        objective=float(reconstruction.detach()),
        reconstruction=float(reconstruction.detach()),
        inputs=[value.detach() for value in inputs],
        preactivations=[value.detach() for value in preactivations],
        output_gradients=output_gradients,
        parameter_gradients=[
            value.detach().clone() for value in homogeneous_gradients(model.layers)
        ],
    )


@torch.no_grad()
def _activation_factors(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [homogeneous(value).T @ homogeneous(value) / len(value) for value in inputs]


@torch.no_grad()
def _gradient_factors(output_gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.T @ value / len(value) for value in output_gradients]


def _run_id(config: StructuredRunConfig) -> str:
    payload = json.dumps(asdict(config), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    config: StructuredRunConfig, campaign: str | None = None, worker_id: str | int = 0
) -> dict[str, Any]:
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    data = load_data(config.data_path, device, include_test=config.selected)
    model = build_model(config.seed, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    if config.method == "kfac":
        optimizer: KFAC | KBFGSL = KFAC(
            model.layers,
            lr=config.learning_rate,
            damping=config.resolved_damping,
            momentum=config.momentum,
            factor_decay=config.factor_decay,
            inverse_update_frequency=config.inverse_update_frequency,
        )
    else:
        optimizer = KBFGSL(
            model.layers,
            lr=config.learning_rate,
            damping=config.resolved_damping,
            momentum=config.momentum,
            factor_decay=config.factor_decay,
            history_size=config.history_size,
        )
    batches_per_epoch = math.ceil(len(data.train) / config.batch_size)
    permutations: dict[int, torch.Tensor] = {}

    def stochastic_batch(step: int) -> torch.Tensor:
        epoch, batch_index = divmod(step, batches_per_epoch)
        if epoch not in permutations:
            permutations[epoch] = torch.as_tensor(
                stateless_epoch_permutation(config.seed, epoch, len(data.train)),
                dtype=torch.long,
                device=device,
            )
        start = batch_index * config.batch_size
        indices = permutations[epoch][start : min(start + config.batch_size, len(data.train))]
        return data.train[indices]

    def training_batch(step: int) -> torch.Tensor:
        return data.train if config.mode == "deterministic" else stochastic_batch(step)

    calls = auxiliary_calls = 0
    if config.mode == "stochastic":
        activation_sums: list[torch.Tensor] | None = None
        gradient_sums: list[torch.Tensor] | None = None
        for batch_index in range(batches_per_epoch):
            values = stochastic_batch(batch_index)
            if config.method == "kfac":
                auxiliary = _auxiliary_backward(model, values, fisher=True)
                calls += 1
                auxiliary_calls += 1
                gradients = _gradient_factors(auxiliary.output_gradients)
                if gradient_sums is None:
                    gradient_sums = [value.clone() for value in gradients]
                else:
                    for total, value in zip(gradient_sums, gradients, strict=True):
                        total.add_(value)
                inputs = auxiliary.inputs
            else:
                with torch.no_grad():
                    _, inputs, _ = _forward_with_caches(model, values, retain_grad=False)
            activations = _activation_factors(inputs)
            if activation_sums is None:
                activation_sums = [value.clone() for value in activations]
            else:
                for total, value in zip(activation_sums, activations, strict=True):
                    total.add_(value)
        assert activation_sums is not None
        activation_sums = [value / batches_per_epoch for value in activation_sums]
        if isinstance(optimizer, KFAC):
            assert gradient_sums is not None
            optimizer.set_initial_factors(
                activation_sums, [value / batches_per_epoch for value in gradient_sums]
            )
        else:
            optimizer.set_initial_activation_factors(activation_sums)
        model.zero_grad(set_to_none=True)
    history: list[dict[str, float | int | None]] = []
    completed_steps = 0
    last_objective: float | None = None
    last_reconstruction: float | None = None
    next_evaluation = config.evaluation_interval

    def record() -> None:
        model.eval()
        objective, reconstruction_bce, reconstruction_mse = _metrics(
            model, data.train, include_regularization=True
        )
        _sync(device)
        history.append(
            {
                "parameter_steps": completed_steps,
                "gradient_evaluations": calls,
                "endpoint_evaluations": 0,
                "auxiliary_gradient_evaluations": auxiliary_calls,
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
    status, reason = ("complete", None)
    try:
        for step in range(config.parameter_steps):
            values = training_batch(step)
            current = _real_backward(model, values)
            calls += 1
            last_objective = current.objective
            last_reconstruction = current.reconstruction
            if not math.isfinite(current.objective):
                raise FloatingPointError("non-finite training objective")
            if isinstance(optimizer, KFAC):
                auxiliary = _auxiliary_backward(model, values, fisher=True)
                calls += 1
                auxiliary_calls += 1
                optimizer.step(
                    current.parameter_gradients, auxiliary.inputs, auxiliary.output_gradients
                )
            else:
                optimizer.apply_step(current.parameter_gradients, current.inputs)
                auxiliary = _auxiliary_backward(model, values, fisher=False)
                calls += 1
                auxiliary_calls += 1
                optimizer.update_curvature(
                    current.preactivations,
                    auxiliary.preactivations,
                    current.output_gradients,
                    auxiliary.output_gradients,
                    stochastic=config.mode == "stochastic",
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
        model.eval()
        _, test_bce, test_mse = _metrics(model, data.test, include_regularization=False)
        _sync(device)
    tail = [float(point["training_objective"]) for point in history[-5:]]
    selection_score = (
        sum(tail) / len(tail)
        if status == "complete" and tail and all((math.isfinite(value) for value in tail))
        else None
    )
    summary = {
        "run_id": _run_id(config),
        "problem": "mnist_deep_autoencoder",
        "mode": config.mode,
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "lambda": None,
        "damping": config.resolved_damping,
        "beta1": config.momentum,
        "nesterov": None,
        "qme_backend": None,
        "refresh_interval": None,
        "status": status,
        "reason": reason,
        "parameter_steps": completed_steps,
        "gradient_evaluations": calls,
        "auxiliary_gradient_evaluations": auxiliary_calls,
        "endpoint_evaluations": 0,
        "curvature_updates": completed_steps,
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
        "upstream_revision": "be57dc380be1817dc9746d174fbe06e8fdeb1946",
        "config": asdict(config),
        "history": history,
    }
    if campaign is not None:
        CampaignStore(campaign).write_shard(worker_id, [summary])
    return summary
