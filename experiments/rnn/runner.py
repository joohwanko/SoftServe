"""FP32 PyTorch implementation of the canonical RNN Adding benchmark."""

from __future__ import annotations
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from softserve.baselines import SGDM, make_soap
from softserve.benchmarks.rnn_adding import stochastic_qn_accounting
from softserve.optim import SoftServeDiag, SoftServeKron
from softserve.secants import IntervalSecant
from softserve.storage import CampaignStore

HIDDEN_SIZE = 16
INPUT_SIZE = 2
RECURRENT_COLUMNS = HIDDEN_SIZE + INPUT_SIZE + 1
OUTPUT_COLUMNS = HIDDEN_SIZE + 1
VALIDATION_SIZE = TEST_SIZE = 8192
Method = Literal["softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap"]
QN_METHODS = {"softserve_kron", "softserve_diag"}


@dataclass(frozen=True)
class RNNAddingRunConfig:
    method: Method
    seed: int
    learning_rate: float
    gradient_budget: int = 40000
    fixed_lambda: float | None = None
    sequence_length: int = 100
    batch_size: int = 128
    refresh_interval: int = 10
    evaluation_interval: int = 200
    beta: float = 0.9
    beta1: float | None = None
    nesterov: bool | None = None
    device: str = "cpu"
    dtype: str = "float32"
    selected: bool = False
    qme_backend: str = "gemm"
    root_steps: int = 18
    inverse_steps: int = 10

    @classmethod
    def from_json(cls, path: str | Path) -> "RNNAddingRunConfig":
        config = cls(**json.loads(Path(path).read_text()))
        config.validate()
        return config

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]

    @property
    def parameter_steps(self) -> int:
        if self.method in QN_METHODS:
            return stochastic_qn_accounting(self.gradient_budget, self.refresh_interval)[0]
        return self.gradient_budget

    @property
    def resolved_beta1(self) -> float:
        if self.beta1 is not None:
            return self.beta1
        if self.method == "muon":
            return 0.95
        if self.method == "soap":
            return 0.99
        if self.method == "adam":
            return 0.9
        return self.beta

    @property
    def resolved_nesterov(self) -> bool:
        if self.nesterov is not None:
            return self.nesterov
        return self.method == "muon"

    def validate(self) -> None:
        if self.method not in {"softserve_kron", "softserve_diag", "sgd_m", "adam", "muon", "soap"}:
            raise ValueError(f"unsupported method: {self.method}")
        if (self.method in QN_METHODS) != (self.fixed_lambda is not None):
            raise ValueError("fixed_lambda is defined exactly for SoftServe")
        if self.fixed_lambda is not None and (
            not math.isfinite(self.fixed_lambda) or self.fixed_lambda <= 0.0
        ):
            raise ValueError("fixed_lambda must be finite and positive")
        if self.sequence_length != 100 or self.batch_size != 128:
            raise ValueError("the paper benchmark fixes length=100 and batch=128")
        if self.gradient_budget < 2 or self.refresh_interval < 1:
            raise ValueError("invalid budget or refresh interval")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must lie in [0,1)")
        if self.beta1 is not None and (not 0.0 <= self.beta1 < 1.0):
            raise ValueError("beta1 must lie in [0,1)")
        if self.resolved_nesterov and self.method not in {
            "softserve_kron",
            "softserve_diag",
            "muon",
        }:
            raise ValueError("Nesterov is defined only for SoftServe and Muon in this benchmark")
        if self.dtype != "float32":
            raise ValueError("the main-paper RNN Adding campaign is FP32")
        if self.device != "cpu":
            raise ValueError("the small canonical benchmark is CPU-routed")


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class Problem:
    recurrent: torch.Tensor
    output: torch.Tensor
    validation: Batch
    test: Batch
    validation_mean_baseline: float
    test_mean_baseline: float


class AddingRNN(nn.Module):
    """One tanh recurrent affine matrix and one affine output matrix."""

    def __init__(self, recurrent: torch.Tensor, output: torch.Tensor) -> None:
        super().__init__()
        self.recurrent = nn.Parameter(recurrent.clone())
        self.output = nn.Parameter(output.clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(inputs.shape[0], HIDDEN_SIZE, dtype=inputs.dtype, device=inputs.device)
        ones = torch.ones(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
        for index in range(inputs.shape[1]):
            hidden = torch.tanh(
                F.linear(torch.cat((hidden, inputs[:, index], ones), dim=1), self.recurrent)
            )
        return F.linear(torch.cat((hidden, ones), dim=1), self.output).squeeze(1)

    def matrix_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.recurrent, self.output)


def _numpy_batch(length: int, size: int, seed_sequence: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence(seed_sequence))
    values = rng.random((size, length))
    markers = np.zeros((size, length), dtype=np.float64)
    halfway = length // 2
    first = rng.integers(0, halfway, size=size)
    second = rng.integers(halfway, length, size=size)
    rows = np.arange(size)
    markers[rows, first] = markers[rows, second] = 1.0
    inputs = np.stack((values, markers), axis=2).astype(np.float32)
    targets = (values[rows, first] + values[rows, second]).astype(np.float32)
    return (inputs, targets)


def _to_batch(arrays: tuple[np.ndarray, np.ndarray], device: torch.device) -> Batch:
    inputs, targets = arrays
    return Batch(
        torch.as_tensor(inputs, dtype=torch.float32, device=device),
        torch.as_tensor(targets, dtype=torch.float32, device=device),
    )


def make_problem(seed: int, device: torch.device) -> Problem:
    rng = np.random.Generator(np.random.PCG64(211000 + seed))
    recurrent = np.zeros((HIDDEN_SIZE, RECURRENT_COLUMNS), dtype=np.float32)
    recurrent[:, :HIDDEN_SIZE] = 0.95 * np.eye(HIDDEN_SIZE, dtype=np.float32)
    recurrent[:, HIDDEN_SIZE : HIDDEN_SIZE + INPUT_SIZE] = (
        0.05 * rng.standard_normal((HIDDEN_SIZE, INPUT_SIZE))
    ).astype(np.float32)
    output = np.zeros((1, OUTPUT_COLUMNS), dtype=np.float32)
    output[0, :HIDDEN_SIZE] = (0.05 * rng.standard_normal(HIDDEN_SIZE)).astype(np.float32)
    validation = _to_batch(_numpy_batch(100, VALIDATION_SIZE, [220000, seed, 100]), device)
    test = _to_batch(_numpy_batch(100, TEST_SIZE, [230000, seed, 100]), device)
    validation_baseline = float((validation.targets - validation.targets.mean()).square().mean())
    test_baseline = float((test.targets - test.targets.mean()).square().mean())
    if abs(validation_baseline - 1.0 / 6.0) > 0.005 or abs(test_baseline - 1.0 / 6.0) > 0.005:
        raise RuntimeError("predict-the-mean baseline is outside tolerance")
    return Problem(
        torch.from_numpy(recurrent).to(device),
        torch.from_numpy(output).to(device),
        validation,
        test,
        validation_baseline,
        test_baseline,
    )


def training_batch(config: RNNAddingRunConfig, step: int, device: torch.device) -> Batch:
    return _to_batch(
        _numpy_batch(
            config.sequence_length,
            config.batch_size,
            [210000, config.seed, config.sequence_length, config.batch_size, step],
        ),
        device,
    )


@dataclass
class _Gradient:
    loss: float
    values: tuple[torch.Tensor, ...]


def _gradient(model: AddingRNN, batch: Batch) -> _Gradient:
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(batch.inputs), batch.targets)
    loss.backward()
    parameters = model.matrix_parameters()
    if any((parameter.grad is None for parameter in parameters)):
        raise RuntimeError("missing RNN gradient")
    return _Gradient(
        float(loss.detach()), tuple((parameter.grad.detach().clone() for parameter in parameters))
    )


@torch.no_grad()
def _mse(model: AddingRNN, batch: Batch, chunk_size: int = 512) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(batch.targets), chunk_size):
        stop = min(start + chunk_size, len(batch.targets))
        residual = model(batch.inputs[start:stop]) - batch.targets[start:stop]
        total += residual.double().square().sum().cpu()
    return float(total / len(batch.targets))


def _assign(parameters: tuple[nn.Parameter, ...], values: tuple[torch.Tensor, ...]) -> None:
    for parameter, value in zip(parameters, values, strict=True):
        parameter.grad = value


class _Controller:
    def __init__(self, model: AddingRNN, config: RNNAddingRunConfig) -> None:
        self.parameters = model.matrix_parameters()
        self.qn: SoftServeKron | SoftServeDiag | None = None
        beta1 = config.resolved_beta1
        nesterov = config.resolved_nesterov
        if config.method == "softserve_kron":
            self.qn = SoftServeKron(
                self.parameters,
                lr=config.learning_rate,
                lam=float(config.fixed_lambda),
                beta1=beta1,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                nesterov=nesterov,
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
                self.parameters,
                lr=config.learning_rate,
                lam=float(config.fixed_lambda),
                beta1=beta1,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                nesterov=nesterov,
                constrained_update=True,
                lambda_schedule="fixed",
                pair_diagnostics=True,
            )
            self.main = self.qn
        elif config.method == "sgd_m":
            self.main = SGDM(
                self.parameters, lr=config.learning_rate, beta=beta1, normalization="unit"
            )
        elif config.method == "adam":
            self.main = torch.optim.Adam(
                self.parameters,
                lr=config.learning_rate,
                betas=(beta1, 0.999),
                eps=1e-08,
                weight_decay=0.0,
            )
        elif config.method == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("canonical torch.optim.Muon is unavailable")
            self.main = torch.optim.Muon(
                self.parameters,
                lr=config.learning_rate,
                momentum=beta1,
                nesterov=nesterov,
                ns_steps=5,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            )
        elif config.method == "soap":
            self.main = make_soap(
                self.parameters,
                lr=config.learning_rate,
                betas=(beta1, 0.999),
                precondition_frequency=2,
                schedule_free_beta=0.99,
            )
        else:
            raise ValueError(config.method)

    def train(self) -> None:
        function = getattr(self.main, "train", None)
        if function is not None:
            function()

    def eval(self) -> None:
        function = getattr(self.main, "eval", None)
        if function is not None:
            function()

    def step(self) -> None:
        if self.qn is None:
            self.main.step()
        else:
            self.qn.parameter_step()

    def state_dict(self) -> dict[str, Any]:
        return self.main.state_dict()


def _state_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum((_state_bytes(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return sum((_state_bytes(item) for item in value))
    return 0


def _run_id(config: RNNAddingRunConfig) -> str:
    payload = json.dumps(asdict(config), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    config: RNNAddingRunConfig,
    campaign: str | None = None,
    worker_id: str | int = 0,
    *,
    controller_factory: Callable[[AddingRNN, RNNAddingRunConfig], _Controller] | None = None,
) -> dict[str, Any]:
    config.validate()
    torch.set_num_threads(1)
    device = torch.device(config.device)
    problem = make_problem(config.seed, device)
    model = AddingRNN(problem.recurrent, problem.output).to(device=device, dtype=config.torch_dtype)
    parameters = model.matrix_parameters()
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    controller = (controller_factory or _Controller)(model, config)
    controller.train()
    secant = IntervalSecant(config.refresh_interval) if controller.qn else None
    history: list[dict[str, float | int]] = []
    calls = endpoints = curvature_updates = completed_steps = 0
    last_train_loss = math.nan
    started = time.perf_counter()

    def record() -> None:
        controller.eval()
        model.eval()
        validation = _mse(model, problem.validation)
        controller.train()
        model.train()
        history.append(
            {
                "parameter_steps": completed_steps,
                "gradient_evaluations": calls,
                "endpoint_evaluations": endpoints,
                "wall_seconds": time.perf_counter() - started,
                "training_batch_mse": last_train_loss,
                "validation_mse": validation,
            }
        )

    record()
    status, reason = ("complete", None)
    try:
        for step in range(config.parameter_steps):
            batch = training_batch(config, step, device)
            current = _gradient(model, batch)
            calls += 1
            last_train_loss = current.loss
            if not math.isfinite(current.loss):
                raise FloatingPointError("non-finite training loss")
            _assign(parameters, current.values)
            if secant is not None and secant.starts_at(step):
                secant.start(parameters, current.values, batch_key=step)
            controller.step()
            if secant is not None and secant.ends_at(step) and (step + 1 < config.parameter_steps):
                endpoint_batch = training_batch(config, int(secant.batch_key), device)
                endpoint = _gradient(model, endpoint_batch)
                calls += 1
                endpoints += 1
                endpoint_values = tuple(
                    (endpoint.values[parameter_indices[id(parameter)]] for parameter in parameters)
                )
                s_values, y_values = secant.pair(parameters, endpoint_values)
                controller.qn.update_curvature(s_values, y_values)
                curvature_updates += 1
            completed_steps = step + 1
            if (
                calls >= len(history) * config.evaluation_interval
                or completed_steps == config.parameter_steps
            ):
                record()
            if calls > config.gradient_budget:
                raise RuntimeError("gradient-call budget exceeded")
        if calls != config.gradient_budget:
            raise RuntimeError(f"used {calls} calls, expected {config.gradient_budget}")
    except (FloatingPointError, RuntimeError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["parameter_steps"] != completed_steps:
        record()
    tail = [float(point["validation_mse"]) for point in history[-5:]]
    test_mse = _mse(model, problem.test) if config.selected and status == "complete" else None
    summary = {
        "run_id": _run_id(config),
        "problem": "rnn_adding",
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "lambda": config.fixed_lambda,
        "beta1": config.resolved_beta1,
        "nesterov": config.resolved_nesterov,
        "qme_backend": config.qme_backend if config.method == "softserve_kron" else None,
        "refresh_interval": config.refresh_interval if config.method in QN_METHODS else None,
        "tau": None
        if config.fixed_lambda is None
        else config.fixed_lambda / (1.0 + config.fixed_lambda),
        "status": status,
        "reason": reason,
        "dtype": config.dtype,
        "parameter_steps": completed_steps,
        "gradient_evaluations": calls,
        "endpoint_evaluations": endpoints,
        "curvature_updates": curvature_updates,
        "selection_score": sum(tail) / len(tail),
        "final_validation_mse": float(history[-1]["validation_mse"]),
        "test_mse": test_mse,
        "validation_mean_baseline": problem.validation_mean_baseline,
        "test_mean_baseline": problem.test_mean_baseline if config.selected else None,
        "wall_seconds": time.perf_counter() - started,
        "optimizer_state_bytes": _state_bytes(controller.state_dict()),
        "config": asdict(config),
        "history": history,
    }
    if campaign is not None:
        CampaignStore(campaign).write_shard(worker_id, [summary])
    return summary
