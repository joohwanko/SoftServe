"""Concise runner for the three Rathore et al. PINN problems.

The benchmark definition lives in :mod:`softserve.benchmarks.pinn`.  This file
only owns optimizer routing, interval secants, accounting, and storage-facing
results.  One process executes exactly one fully resolved configuration.
"""

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
from softserve.baselines import SGDM, make_soap
from softserve.benchmarks.pinn import (
    Collocation,
    PINN,
    PINNConfig,
    ResidualSampler,
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
    build_pinn,
    deterministic_sampler,
    loss_components,
    matrix_routes,
    relative_l2,
)
from softserve.optim import SoftServeDiag, SoftServeKron
from softserve.secants import IntervalSecant
from softserve.storage import CampaignStore

Mode = Literal["deterministic", "stochastic"]
LambdaSchedule = Literal["fixed"]
LambdaScope = Literal["block", "pooled"]
Method = Literal[
    "softserve_kron", "softserve_diag", "sgd_m", "adam", "soap", "muon", "lbfgs_cold", "lbfgs_warm"
]
QN_METHODS = {"softserve_kron", "softserve_diag"}
LBFGS_METHODS = {"lbfgs_cold", "lbfgs_warm"}


@dataclass(frozen=True)
class PINNRunConfig:
    """One resolved run; sweeps are represented by multiple JSON files."""

    pde: str
    mode: Mode
    method: Method
    seed: int
    lr: float
    updates: int = 200000
    gradient_budget: int | None = None
    fallback_lr: float = 0.0003
    tau: float | None = None
    beta1: float = 0.9
    optimizer_beta1: float | None = None
    nesterov: bool | None = None
    refresh_interval: int = 1
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    residual_batch: int = 10000
    sampling_namespace: int = 70000
    width: int = 200
    layers: int = 4
    num_x: int = 257
    num_t: int = 101
    device: str = "cuda"
    dtype: str = "float32"
    eval_every: int = 1000
    selected: bool = False
    qme_backend: str = "gemm"
    root_steps: int = 18
    inverse_steps: int = 10
    soap_betas: tuple[float, float] = (0.99, 0.999)
    soap_precondition_frequency: int = 2
    soap_schedule_free_beta: float | None = 0.99
    soap_gradient_clip: float | None = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: int = 25
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1e-12
    lbfgs_tolerance_change: float = 1e-12
    warmup_steps: int = 11000
    warmup_lr: float = 0.001

    @classmethod
    def from_json(cls, path: str | Path) -> "PINNRunConfig":
        payload = json.loads(Path(path).read_text())
        if "soap_betas" in payload:
            payload["soap_betas"] = tuple(payload["soap_betas"])
        config = cls(**payload)
        config.validate()
        return config

    @property
    def lam(self) -> float | None:
        return None if self.tau is None else self.tau / (1.0 - self.tau)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    @property
    def resolved_beta1(self) -> float | None:
        """Return the first-moment coefficient actually used by the method."""
        if self.method in LBFGS_METHODS:
            return None
        if self.optimizer_beta1 is not None:
            return self.optimizer_beta1
        if self.method == "muon":
            return self.muon_momentum
        if self.method == "soap":
            return self.soap_betas[0]
        return self.beta1

    @property
    def resolved_nesterov(self) -> bool | None:
        """Return the applied Nesterov switch, or None when it is inapplicable."""
        if self.method in QN_METHODS:
            return False if self.nesterov is None else self.nesterov
        if self.method == "muon":
            return self.muon_nesterov if self.nesterov is None else self.nesterov
        return None

    def problem(self) -> PINNConfig:
        return PINNConfig(
            pde=self.pde,
            width=self.width,
            layers=self.layers,
            beta=40.0 if self.pde == "convection" else 5.0,
            rho=5.0,
            num_x=self.num_x,
            num_t=self.num_t,
            residual_batch=self.residual_batch,
        )

    def validate(self) -> None:
        if self.pde not in {"wave", "convection", "reaction"}:
            raise ValueError(f"unknown PDE: {self.pde}")
        if self.mode not in {"deterministic", "stochastic"}:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.method not in QN_METHODS | LBFGS_METHODS | {"sgd_m", "adam", "soap", "muon"}:
            raise ValueError(f"unknown method: {self.method}")
        if self.updates < 1 or self.eval_every < 1 or self.refresh_interval < 1:
            raise ValueError("updates, eval_every, and refresh_interval must be positive")
        if self.lr <= 0 or self.fallback_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not math.isfinite(self.beta1) or not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be finite and lie in [0, 1)")
        if self.optimizer_beta1 is not None and (
            not math.isfinite(self.optimizer_beta1) or not 0.0 <= self.optimizer_beta1 < 1.0
        ):
            raise ValueError("optimizer_beta1 must be finite and lie in [0, 1)")
        if len(self.soap_betas) != 2 or any(
            (not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in self.soap_betas)
        ):
            raise ValueError("soap_betas must contain two finite values in [0, 1)")
        if not math.isfinite(self.muon_momentum) or not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must be finite and lie in [0, 1)")
        if self.nesterov is not None and (not isinstance(self.nesterov, bool)):
            raise ValueError("nesterov must be boolean or None")
        if not isinstance(self.muon_nesterov, bool):
            raise ValueError("muon_nesterov must be boolean")
        if self.nesterov and self.method not in QN_METHODS | {"muon"}:
            raise ValueError("Nesterov is defined only for SoftServe and Muon")
        if self.method in QN_METHODS:
            if self.tau is None or not 0 < self.tau < 1:
                raise ValueError("fixed-lambda SoftSERVE requires tau in (0, 1)")
        elif self.tau is not None:
            raise ValueError("tau is only defined for SoftSERVE")
        if self.method in LBFGS_METHODS:
            if self.mode != "deterministic":
                raise ValueError("strong-Wolfe L-BFGS is deterministic-only")
            if self.gradient_budget is None:
                raise ValueError("L-BFGS requires a gradient-call budget")
        if self.mode == "stochastic" and self.gradient_budget is not None:
            expected = self.updates
            if self.method in QN_METHODS:
                expected += self.updates // self.refresh_interval
            if expected != self.gradient_budget:
                raise ValueError(f"updates imply {expected} calls, not {self.gradient_budget}")
        self.problem().validate()


@dataclass
class _Gradient:
    loss: float
    components: dict[str, float]
    values: tuple[torch.Tensor, ...]


@dataclass
class _ReplicatedGradient:
    """Full-batch gradient plus two equal half-batch replicas."""

    full: _Gradient
    replicas: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gradient(model: PINN, points, problem: PINNConfig) -> _Gradient:
    model.zero_grad(set_to_none=True)
    terms = loss_components(model, points, problem)
    loss = sum(terms.values())
    loss.backward()
    values = tuple(
        (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in model.parameters()
        )
    )
    return _Gradient(
        float(loss.detach()), {name: float(value.detach()) for name, value in terms.items()}, values
    )


def _half_collocation(points: Collocation, start: int, stop: int) -> Collocation:
    """Slice residual points while sharing the complete boundary/initial set."""
    return Collocation(
        points.residual_x[start:stop].detach().requires_grad_(True),
        points.residual_t[start:stop].detach().requires_grad_(True),
        points.initial_x,
        points.initial_t,
        points.upper_x,
        points.upper_t,
        points.lower_x,
        points.lower_t,
    )


def _assign(parameters: tuple[nn.Parameter, ...], values: tuple[torch.Tensor, ...]) -> None:
    for parameter, value in zip(parameters, values, strict=True):
        parameter.grad = value.detach().clone()


def _fixed_metrics(
    model: PINN, sampler: ResidualSampler, problem: PINNConfig, device: torch.device
) -> dict[str, float]:
    model.eval()
    with torch.enable_grad():
        points = sampler.batch(0)
        terms = loss_components(model, points, problem)
        values = {name: float(value.detach()) for name, value in terms.items()}
    model.train()
    return {
        "training_loss": sum(values.values()),
        "residual_loss": values["residual"],
        "boundary_loss": values["boundary"],
        "initial_loss": values["initial"],
        "relative_l2": relative_l2(model, problem, device=device),
    }


def _lr_multiplier(config: PINNRunConfig, step: int) -> float:
    if config.lr_schedule == "constant" or config.updates <= 1:
        return 1.0
    progress = step / (config.updates - 1)
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


class _Controller:
    """Small adapter for the routed fixed-step methods."""

    def __init__(self, model: PINN, config: PINNRunConfig) -> None:
        self.config = config
        self.parameters = tuple(model.parameters())
        matrices, fallback = matrix_routes(model)
        self.main_parameters = tuple(matrices)
        self.fallback_parameters = tuple(fallback)
        self.qn: SoftServeKron | SoftServeDiag | None = None
        self.main: torch.optim.Optimizer
        self.fallback: torch.optim.Optimizer | None = None
        beta1 = config.resolved_beta1
        if beta1 is None:
            raise ValueError(f"{config.method} is not a fixed-step optimizer")
        if config.method == "softserve_kron":
            self.qn = SoftServeKron(
                self.main_parameters,
                lr=config.lr,
                lam=1.0 if config.lam is None else float(config.lam),
                lambda_schedule="fixed",
                beta1=beta1,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                nesterov=bool(config.resolved_nesterov),
                backend=config.qme_backend,
                root_steps=config.root_steps,
                inverse_steps=config.inverse_steps,
                normalize=True,
                gauge="balanced_trace",
                bucket_chunk_size=None,
                constrained_update=True,
                pair_diagnostics=True,
            )
            self.main = self.qn
            self.fallback = torch.optim.Adam(
                self.fallback_parameters,
                lr=config.fallback_lr,
                betas=(beta1, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )
        elif config.method == "softserve_diag":
            self.main_parameters = self.parameters
            self.fallback_parameters = ()
            self.qn = SoftServeDiag(
                self.parameters,
                lr=config.lr,
                lam=1.0 if config.lam is None else float(config.lam),
                lambda_schedule="fixed",
                beta1=beta1,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                nesterov=bool(config.resolved_nesterov),
                constrained_update=True,
                pair_diagnostics=True,
            )
            self.main = self.qn
        elif config.method == "sgd_m":
            self.main = SGDM(self.main_parameters, lr=config.lr, beta=beta1)
            self.fallback = torch.optim.Adam(
                self.fallback_parameters,
                lr=config.fallback_lr,
                betas=(beta1, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )
        elif config.method == "adam":
            self.main_parameters = self.parameters
            self.fallback_parameters = ()
            self.main = torch.optim.Adam(
                self.parameters, lr=config.lr, betas=(beta1, 0.999), eps=1e-08, weight_decay=0.0
            )
        elif config.method == "soap":
            self.main = make_soap(
                self.main_parameters,
                lr=config.lr,
                betas=(beta1, config.soap_betas[1]),
                precondition_frequency=config.soap_precondition_frequency,
                schedule_free_beta=config.soap_schedule_free_beta,
            )
            self.fallback = torch.optim.Adam(
                self.fallback_parameters,
                lr=config.fallback_lr,
                betas=(beta1, config.soap_betas[1]),
                eps=1e-08,
                weight_decay=0.0,
            )
        elif config.method == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("this experiment requires a PyTorch release with Muon")
            self.main = torch.optim.Muon(
                self.main_parameters,
                lr=config.lr,
                weight_decay=0.0,
                momentum=beta1,
                nesterov=bool(config.resolved_nesterov),
                ns_steps=config.muon_ns_steps,
                adjust_lr_fn=None,
            )
            self.fallback = torch.optim.Adam(
                self.fallback_parameters,
                lr=config.fallback_lr,
                betas=(beta1, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"unsupported fixed-step method: {config.method}")

    def set_lr(self, multiplier: float) -> None:
        for group in self.main.param_groups:
            group["lr"] = self.config.lr * multiplier
        if self.fallback is not None:
            for group in self.fallback.param_groups:
                group["lr"] = self.config.fallback_lr * multiplier

    def train(self) -> None:
        function = getattr(self.main, "train", None)
        if function is not None:
            function()

    def eval(self) -> None:
        function = getattr(self.main, "eval", None)
        if function is not None:
            function()

    def parameter_step(self) -> None:
        if self.qn is not None:
            self.qn.parameter_step()
        else:
            if self.config.method == "soap" and self.config.soap_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters, self.config.soap_gradient_clip)
            self.main.step()
        if self.fallback is not None:
            self.fallback.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "main": self.main.state_dict(),
            "fallback": None if self.fallback is None else self.fallback.state_dict(),
        }


def _run_fixed_step(config: PINNRunConfig) -> dict[str, Any]:
    problem = config.problem()
    device = torch.device(config.device)
    model = build_pinn(problem, seed=config.seed, device=device, dtype=config.torch_dtype)
    audit_sampler = deterministic_sampler(
        problem, seed=config.seed, device=device, dtype=config.torch_dtype
    )
    train_sampler = (
        audit_sampler
        if config.mode == "deterministic"
        else ResidualSampler(
            problem,
            seed=config.seed,
            device=device,
            dtype=config.torch_dtype,
            continuous=True,
            namespace=config.sampling_namespace,
        )
    )
    parameters = tuple(model.parameters())
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    controller = _Controller(model, config)
    controller.train()
    secant = IntervalSecant(config.refresh_interval) if controller.qn else None
    route = controller.main_parameters
    gradient_calls = endpoint_calls = curvature_updates = backward_passes = 0
    negative_pairs = pair_count = 0
    qme_residual_max = 0.0
    cached: _Gradient | None = None
    history: list[dict[str, Any]] = []
    scheduler_history: list[dict[str, Any]] = []
    (
        config.updates // config.refresh_interval
        if config.mode == "stochastic"
        else (config.updates - 1) // config.refresh_interval
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record(step: int) -> None:
        controller.eval()
        _sync(device)
        metrics = _fixed_metrics(model, audit_sampler, problem, device)
        _sync(device)
        controller.train()
        history.append(
            {
                "parameter_steps": step,
                "gradient_evaluations": gradient_calls,
                "endpoint_evaluations": endpoint_calls,
                "wall_seconds": time.perf_counter() - started,
                **metrics,
            }
        )

    record(0)
    status, reason = ("complete", None)
    completed_steps = 0
    try:
        for step in range(config.updates):
            batch_key = 0 if config.mode == "deterministic" else step
            if cached is None:
                points = train_sampler.batch(batch_key)
                current = _gradient(model, points, problem)
                backward_passes += 1
                gradient_calls += 1
            else:
                current, cached = (cached, None)
            if not math.isfinite(current.loss):
                raise FloatingPointError("non-finite training loss")
            _assign(parameters, current.values)
            controller.set_lr(_lr_multiplier(config, step))
            if secant is not None and secant.starts_at(step):
                raw = tuple((parameter.grad.detach() for parameter in route))
                secant.start(route, raw, batch_key=batch_key)
            controller.parameter_step()
            if secant is not None and secant.ends_at(step):
                has_future = step + 1 < config.updates
                due = config.mode == "stochastic" or has_future
                if due:
                    endpoint_key = int(secant.batch_key)
                    endpoint_points = train_sampler.batch(endpoint_key)
                    endpoint = _gradient(model, endpoint_points, problem)
                    endpoint_replicas = None
                    backward_passes += 1
                    gradient_calls += 1
                    endpoint_calls += 1
                    if endpoint_replicas is None:
                        endpoint_route = tuple(
                            (
                                endpoint.values[parameter_indices[id(parameter)]]
                                for parameter in route
                            )
                        )
                    else:
                        endpoint_route = tuple(
                            (
                                endpoint_replicas[0][parameter_indices[id(parameter)]]
                                for parameter in route
                            )
                        )
                    s_values, y_values = secant.pair(route, endpoint_route)
                    controller.qn.update_curvature(s_values, y_values)
                    curvature_updates += 1
                    chi = [float(value) for value in controller.qn.last_pair_chi]
                    negative_pairs += sum((value < 0 for value in chi))
                    pair_count += len(chi)
                    residuals = (
                        controller.qn.last_qme_residual_A + controller.qn.last_qme_residual_G
                        if isinstance(controller.qn, SoftServeKron)
                        else []
                    )
                    if residuals:
                        qme_residual_max = max(
                            qme_residual_max, max((float(value) for value in residuals))
                        )
                    completed = step + 1
                    if config.mode == "deterministic":
                        cached = endpoint
            completed = step + 1
            completed_steps = completed
            if completed % config.eval_every == 0 or completed == config.updates:
                record(completed)
            if config.gradient_budget is not None and gradient_calls > config.gradient_budget:
                raise RuntimeError("gradient-call budget exceeded")
        if config.gradient_budget is not None and gradient_calls != config.gradient_budget:
            raise RuntimeError(
                f"used {gradient_calls} gradient calls, expected {config.gradient_budget}"
            )
    except (FloatingPointError, RuntimeError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    _sync(device)
    qn = controller.qn
    scalar_diagnostic_names = (
        "action_guard_observation_count",
        "action_guard_activation_count",
        "action_guard_fallback_count",
        "action_guard_max_step_norm",
        "parameter_relative_fuse_observation_count",
        "parameter_relative_fuse_activation_count",
        "maximum_relative_update_norm",
        "maximum_initial_scale_update_norm",
    )
    scalar_diagnostics = {
        name: None if qn is None or getattr(qn, name, None) is None else float(getattr(qn, name))
        for name in scalar_diagnostic_names
    }
    summary = {
        "status": status,
        "reason": reason,
        "parameter_steps": completed_steps,
        "gradient_evaluations": gradient_calls,
        "backward_passes": backward_passes,
        "endpoint_evaluations": endpoint_calls,
        "curvature_updates": curvature_updates,
        "negative_secant_fraction": negative_pairs / pair_count if pair_count else 0.0,
        "qme_residual_max": qme_residual_max if controller.qn else None,
        "qn_skipped": None if qn is None else int(qn.skipped),
        **scalar_diagnostics,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        **{
            key: value
            for key, value in history[-1].items()
            if key not in {"parameter_steps", "gradient_evaluations", "endpoint_evaluations"}
        },
    }
    return {
        "model": model,
        "controller": controller,
        "summary": summary,
        "history": history,
        "scheduler_history": scheduler_history,
    }


def _run_lbfgs(config: PINNRunConfig) -> dict[str, Any]:
    """Full-parameter strong-Wolfe control, optionally after an Adam warmup."""
    problem, device = (config.problem(), torch.device(config.device))
    model = build_pinn(problem, seed=config.seed, device=device, dtype=config.torch_dtype)
    sampler = deterministic_sampler(
        problem, seed=config.seed, device=device, dtype=config.torch_dtype
    )
    parameters = tuple(model.parameters())
    budget = int(config.gradient_budget)
    calls = updates = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record() -> None:
        _sync(device)
        history.append(
            {
                "parameter_steps": updates,
                "gradient_evaluations": calls,
                "endpoint_evaluations": 0,
                "wall_seconds": time.perf_counter() - started,
                **_fixed_metrics(model, sampler, problem, device),
            }
        )

    record()
    if config.method == "lbfgs_warm":
        adam = torch.optim.Adam(parameters, lr=config.warmup_lr)
        for _ in range(min(config.warmup_steps, budget)):
            result = _gradient(model, sampler.batch(0), problem)
            calls += 1
            _assign(parameters, result.values)
            adam.step()
            updates += 1
            if calls % config.eval_every == 0:
                record()
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=config.lr,
        max_iter=config.lbfgs_max_iter,
        max_eval=config.lbfgs_max_eval,
        history_size=config.lbfgs_history_size,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )
    status, reason, next_log = (
        "complete",
        None,
        (calls // config.eval_every + 1) * config.eval_every,
    )
    try:
        while calls < budget:
            before = tuple((parameter.detach().clone() for parameter in parameters))
            remaining = budget - calls
            optimizer.param_groups[0]["max_eval"] = min(config.lbfgs_max_eval, remaining)
            optimizer.param_groups[0]["max_iter"] = min(config.lbfgs_max_iter, remaining)

            def closure() -> torch.Tensor:
                nonlocal calls
                if calls >= budget:
                    raise StopIteration
                result = _gradient(model, sampler.batch(0), problem)
                calls += 1
                return torch.as_tensor(result.loss, device=device, dtype=config.torch_dtype)

            try:
                optimizer.step(closure)
            except StopIteration:
                break
            updates += 1
            if calls >= next_log:
                record()
                next_log = (calls // config.eval_every + 1) * config.eval_every
            displacement = sum(
                (
                    float((parameter.detach() - old).double().square().sum())
                    for parameter, old in zip(parameters, before, strict=True)
                )
            )
            if displacement == 0.0:
                break
    except (FloatingPointError, RuntimeError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["gradient_evaluations"] != calls:
        record()
    summary = {
        "status": status,
        "reason": reason,
        "parameter_steps": updates,
        "gradient_evaluations": calls,
        "endpoint_evaluations": 0,
        "curvature_updates": 0,
        "negative_secant_fraction": None,
        "qme_residual_max": None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        **{
            key: value
            for key, value in history[-1].items()
            if key not in {"parameter_steps", "gradient_evaluations", "endpoint_evaluations"}
        },
    }
    return {"model": model, "controller": optimizer, "summary": summary, "history": history}


def _run_id(config: PINNRunConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    config: PINNRunConfig, *, campaign: str | None = None, worker_id: str | int = 0
) -> dict[str, Any]:
    """Execute and optionally commit one configuration to a CampaignStore."""
    config.validate()
    run_id = _run_id(config)
    result = _run_lbfgs(config) if config.method in LBFGS_METHODS else _run_fixed_step(config)
    row = {
        "run_id": run_id,
        "pde": config.pde,
        "mode": config.mode,
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.lr,
        "fallback_learning_rate": config.fallback_lr,
        "tau": config.tau,
        "lambda": config.lam,
        "lambda_schedule": "fixed",
        "lambda_cap": None,
        "lambda_scope": "block",
        "beta1": config.resolved_beta1,
        "nesterov": config.resolved_nesterov,
        "refresh_interval": config.refresh_interval,
        "dtype": config.dtype,
        "line_search": "strong_wolfe" if config.method in LBFGS_METHODS else None,
        "lbfgs_tolerance_grad": config.lbfgs_tolerance_grad
        if config.method in LBFGS_METHODS
        else None,
        "lbfgs_tolerance_change": config.lbfgs_tolerance_change
        if config.method in LBFGS_METHODS
        else None,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "torch_version": torch.__version__,
        "config": asdict(config),
        **result["summary"],
        "history": result["history"],
        "scheduler_history": result.get("scheduler_history", []),
    }
    if campaign is not None:
        store = CampaignStore(campaign)
        store.write_shard(worker_id, [row])
        if config.selected and row["status"] == "complete":
            store.checkpoints.mkdir(parents=True, exist_ok=True)
            path = store.checkpoints / f"{run_id}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "format": "softserve-pinn-v1",
                    "config": asdict(config),
                    "model": result["model"].state_dict(),
                    "optimizer": result["controller"].state_dict(),
                    "summary": result["summary"],
                },
                temporary,
            )
            os.replace(temporary, path)
    return row
