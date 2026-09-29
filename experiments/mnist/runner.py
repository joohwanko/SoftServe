"""FP32 MNIST deep-autoencoder optimizer runner."""

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
from torch import nn
from torch.nn import functional as F
from softserve.baselines import SGDM, make_soap
from softserve.benchmarks.deep_autoencoder import (
    DeepAutoencoder,
    build_model,
    load_data,
    matrix_routes,
    stateless_epoch_permutation,
)
from softserve.benchmarks.rnn_adding import stochastic_qn_accounting
from softserve.optim import SoftServeDiag, SoftServeKron
from softserve.secants import IntervalSecant
from softserve.storage import CampaignStore

Method = Literal["softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap"]
METHODS = ("softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap")
QN_METHODS = {"softserve_kron", "softserve_diag"}
NESTEROV_METHODS = QN_METHODS | {"muon"}
MATRIX_METHODS = {"softserve_kron", "sgd_m", "muon", "soap"}
L2_COEFFICIENT = 1e-05


@dataclass(frozen=True)
class AutoencoderRunConfig:
    data_path: str
    method: Method
    seed: int
    learning_rate: float
    gradient_budget: int
    study: str = "unspecified"
    fixed_lambda: float | None = None
    batch_size: int = 1000
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
    def from_json(cls, path: str | Path) -> "AutoencoderRunConfig":
        config = cls(**json.loads(Path(path).read_text()))
        config.validate()
        return config

    @property
    def parameter_steps(self) -> int:
        if self.method in QN_METHODS:
            return stochastic_qn_accounting(self.gradient_budget, self.refresh_interval)[0]
        return self.gradient_budget

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unsupported method: {self.method}")
        if not self.study:
            raise ValueError("study must be non-empty")
        if (self.method in QN_METHODS) != (self.fixed_lambda is not None):
            raise ValueError("fixed_lambda is defined exactly for SoftServe")
        if self.fixed_lambda is not None and (
            not math.isfinite(self.fixed_lambda) or self.fixed_lambda <= 0.0
        ):
            raise ValueError("fixed_lambda must be finite and positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.fallback_learning_rate) or self.fallback_learning_rate <= 0.0:
            raise ValueError("fallback_learning_rate must be finite and positive")
        if (
            min(
                self.gradient_budget,
                self.batch_size,
                self.refresh_interval,
                self.evaluation_interval,
                self.root_steps,
                self.inverse_steps,
            )
            < 1
        ):
            raise ValueError("budgets, intervals, and iteration counts must be positive")
        if self.batch_size != 1000 or self.refresh_interval != 10:
            raise ValueError("the benchmark fixes batch=1000 and K=10")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must lie in [0,1)")
        if self.method not in NESTEROV_METHODS and self.nesterov:
            raise ValueError(f"Nesterov is not defined for {self.method}")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.method in QN_METHODS:
            stochastic_qn_accounting(self.gradient_budget, self.refresh_interval)


@dataclass
class _Gradient:
    objective: float
    reconstruction_bce: float
    values: tuple[torch.Tensor, ...]


def _losses(
    model: DeepAutoencoder, values: torch.Tensor, *, regularized: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(values)
    reconstruction = (
        F.binary_cross_entropy_with_logits(logits, values, reduction="none").sum(dim=1).mean()
    )
    objective = reconstruction
    if regularized:
        penalty = torch.stack(
            [parameter.float().square().sum() for parameter in model.parameters()]
        ).sum()
        objective = objective + 0.5 * L2_COEFFICIENT * penalty
    return (objective, reconstruction)


def _gradient(model: DeepAutoencoder, values: torch.Tensor) -> _Gradient:
    model.zero_grad(set_to_none=True)
    objective, reconstruction = _losses(model, values, regularized=True)
    objective.backward()
    parameters = tuple(model.parameters())
    gradients = tuple(
        (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in parameters
        )
    )
    return _Gradient(float(objective.detach()), float(reconstruction.detach()), gradients)


@torch.no_grad()
def _metrics(
    model: DeepAutoencoder,
    values: torch.Tensor,
    *,
    include_regularization: bool,
    chunk_size: int = 1000,
) -> tuple[float, float, float]:
    total_bce = 0.0
    total_squared = 0.0
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size]
        logits = model(batch)
        total_bce += float(
            F.binary_cross_entropy_with_logits(logits, batch, reduction="none").sum()
        )
        total_squared += float((torch.sigmoid(logits) - batch).square().sum())
    reconstruction_bce = total_bce / len(values)
    reconstruction_mse = total_squared / len(values)
    objective = reconstruction_bce
    if include_regularization:
        penalty = sum((float(parameter.float().square().sum()) for parameter in model.parameters()))
        objective += 0.5 * L2_COEFFICIENT * penalty
    return (objective, reconstruction_bce, reconstruction_mse)


def _assign(parameters: tuple[nn.Parameter, ...], gradients: tuple[torch.Tensor, ...]) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _Controller:
    def __init__(self, model: DeepAutoencoder, config: AutoencoderRunConfig) -> None:
        self.parameters = tuple(model.parameters())
        matrices, biases = matrix_routes(model)
        if config.method == "softserve_diag":
            self.main_parameters = self.parameters
            self.fallback_parameters: tuple[nn.Parameter, ...] = ()
        elif config.method == "adam":
            self.main_parameters = self.parameters
            self.fallback_parameters = ()
        else:
            self.main_parameters = tuple(matrices)
            self.fallback_parameters = tuple(biases)
        self.qn: SoftServeKron | SoftServeDiag | None = None
        self.fallback: torch.optim.Optimizer | None = None
        if config.method == "softserve_kron":
            self.qn = SoftServeKron(
                self.main_parameters,
                lr=config.learning_rate,
                lam=float(config.fixed_lambda),
                beta1=config.beta1,
                nesterov=config.nesterov,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                backend=config.qme_backend,
                root_steps=config.root_steps,
                inverse_steps=config.inverse_steps,
                normalize=True,
                gauge="balanced_trace",
                bucket_chunk_size=None,
                constrained_update=True,
                lambda_schedule="fixed",
                pair_diagnostics=True,
            )
            self.main: torch.optim.Optimizer = self.qn
        elif config.method == "softserve_diag":
            self.qn = SoftServeDiag(
                self.main_parameters,
                lr=config.learning_rate,
                lam=float(config.fixed_lambda),
                beta1=config.beta1,
                nesterov=config.nesterov,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                constrained_update=True,
                lambda_schedule="fixed",
                pair_diagnostics=True,
            )
            self.main = self.qn
        elif config.method == "sgd_m":
            self.main = SGDM(
                self.main_parameters,
                lr=config.learning_rate,
                beta=config.beta1,
                normalization="unit",
            )
        elif config.method == "adam":
            self.main = torch.optim.Adam(
                self.main_parameters,
                lr=config.learning_rate,
                betas=(config.beta1, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )
        elif config.method == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("canonical torch.optim.Muon is unavailable")
            self.main = torch.optim.Muon(
                self.main_parameters,
                lr=config.learning_rate,
                momentum=config.beta1,
                nesterov=config.nesterov,
                ns_steps=5,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
        elif config.method == "soap":
            self.main = make_soap(
                self.main_parameters,
                lr=config.learning_rate,
                betas=(config.beta1, 0.999),
                precondition_frequency=2,
                schedule_free_beta=0.99,
            )
        else:
            raise ValueError(config.method)
        if self.fallback_parameters:
            self.fallback = torch.optim.Adam(
                self.fallback_parameters,
                lr=config.fallback_learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )

    def train(self) -> None:
        function = getattr(self.main, "train", None)
        if function is not None:
            function()

    def eval(self) -> None:
        function = getattr(self.main, "eval", None)
        if function is not None:
            function()

    def parameter_step(self) -> None:
        if self.qn is None:
            self.main.step()
        else:
            self.qn.parameter_step()
        if self.fallback is not None:
            self.fallback.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "main": self.main.state_dict(),
            "fallback": None if self.fallback is None else self.fallback.state_dict(),
        }


def _state_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum((_state_bytes(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return sum((_state_bytes(item) for item in value))
    return 0


def _run_id(config: AutoencoderRunConfig) -> str:
    payload = json.dumps(asdict(config), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    config: AutoencoderRunConfig, campaign: str | None = None, worker_id: str | int = 0
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
    controller = _Controller(model, config)
    controller.train()
    route = controller.main_parameters
    secant = IntervalSecant(config.refresh_interval) if controller.qn is not None else None
    history: list[dict[str, float | int | None]] = []
    calls = endpoints = curvature_updates = completed_steps = 0
    negative_pairs = pair_count = 0
    qme_residual_max = 0.0
    last_batch_objective: float | None = None
    last_batch_reconstruction: float | None = None
    next_evaluation = 0
    batches_per_epoch = math.ceil(len(data.train) / config.batch_size)
    cached_epoch = -1
    cached_permutation: torch.Tensor | None = None

    def batch(key: int) -> torch.Tensor:
        nonlocal cached_epoch, cached_permutation
        epoch, batch_index = divmod(key, batches_per_epoch)
        if epoch != cached_epoch:
            cached_permutation = torch.as_tensor(
                stateless_epoch_permutation(config.seed, epoch, len(data.train)),
                dtype=torch.long,
                device=device,
            )
            cached_epoch = epoch
        assert cached_permutation is not None
        start = batch_index * config.batch_size
        indices = cached_permutation[start : min(start + config.batch_size, len(data.train))]
        return data.train[indices]

    def record() -> None:
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
                "training_batch_objective": last_batch_objective,
                "training_batch_reconstruction_bce": last_batch_reconstruction,
                "training_objective": objective,
                "training_reconstruction_bce": reconstruction_bce,
                "training_reconstruction_mse": reconstruction_mse,
            }
        )
        model.train()
        controller.train()

    record()
    next_evaluation = config.evaluation_interval
    status, reason = ("complete", None)
    try:
        for step in range(config.parameter_steps):
            current = _gradient(model, batch(step))
            calls += 1
            last_batch_objective = current.objective
            last_batch_reconstruction = current.reconstruction_bce
            if not math.isfinite(current.objective):
                raise FloatingPointError("non-finite training objective")
            _assign(parameters, current.values)
            if secant is not None and secant.starts_at(step):
                route_gradients = tuple(
                    (current.values[parameter_indices[id(parameter)]] for parameter in route)
                )
                secant.start(route, route_gradients, batch_key=step)
            controller.parameter_step()
            if secant is not None and secant.ends_at(step) and (step + 1 < config.parameter_steps):
                endpoint = _gradient(model, batch(int(secant.batch_key)))
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
    summary = {
        "run_id": _run_id(config),
        "problem": "mnist_deep_autoencoder",
        "mode": "stochastic",
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "lambda": config.fixed_lambda,
        "beta1": config.beta1,
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
        "optimizer_state_bytes": _state_bytes(controller.state_dict()),
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
