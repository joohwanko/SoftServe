"""GPT-2-small/FineWeb optimizer runner with a fixed sequential token stream."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import torch
from torch import nn
from softserve.baselines import SGDM
from softserve.benchmarks.nanogpt import GPTConfig, NanoGPT, TokenFile, matrix_routes
from softserve.optim import SoftServeDiag, SoftServeKron
from softserve.secants import IntervalSecant
from softserve.storage import CampaignStore

Method = Literal["adamw", "softserve_diag", "softserve_kron", "sgd_m", "muon", "soap"]
METHODS = {"adamw", "softserve_diag", "softserve_kron", "sgd_m", "muon", "soap"}
SOFTSERVE_METHODS = {"softserve_diag", "softserve_kron"}
DirectionNormalization = Literal["unit", "rms", "none"]
NORMALIZATIONS = {"unit", "rms", "none"}
LambdaSchedule = Literal["fixed"]
LambdaScope = Literal["block", "pooled", "shrink", "shape"]
MetricControl = Literal["real", "identity", "layer_shuffled"]
SOAPRouting = Literal["canonical", "matched"]


@dataclass(frozen=True)
class GPT2RunConfig:
    train_path: str
    validation_path: str
    phase: str
    method: Method
    seed: int
    lr: float
    target_tokens: int
    schedule_steps: int = 1908
    physical_batch: int = 32
    grad_accum: int = 16
    block_size: int = 1024
    vocab_size: int = 50304
    layers: int = 12
    heads: int = 12
    width: int = 768
    fallback_lr: float = 0.0003
    tau: float | None = None
    initial_lambda: float | None = None
    beta1: float = 0.995
    optimizer_beta1: float | None = None
    softserve_nesterov: bool | None = None
    normalization: DirectionNormalization = "unit"
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-08
    muon_momentum: float = 0.95
    soap_path: str | None = None
    soap_routing: SOAPRouting | None = None
    refresh_interval: int = 10
    eval_every: int = 50
    monitor_validation_tokens: int = 50000
    final_validation_tokens: int = 5000000
    final_train_tokens: int = 5000000
    validation_batch: int = 8
    min_lr_ratio: float = 0.1
    lr_decay_fraction: float = 0.1
    checkpoint_every: int = 0
    selected: bool = False
    device: str = "cuda"

    @classmethod
    def from_json(cls, path: str | Path) -> "GPT2RunConfig":
        payload = json.loads(Path(path).read_text())
        if "lambda" in payload:
            if "initial_lambda" in payload:
                raise ValueError("use only the public lambda field")
            payload["initial_lambda"] = payload.pop("lambda")
        if "adam_betas" in payload:
            payload["adam_betas"] = tuple(payload["adam_betas"])
        config = cls(**payload)
        config.validate()
        return config

    @property
    def lam(self) -> float | None:
        if self.initial_lambda is not None:
            return self.initial_lambda
        return None if self.tau is None else self.tau / (1.0 - self.tau)

    @property
    def effective_beta1(self) -> float:
        """Return the first-moment coefficient used by the primary optimizer."""
        if self.optimizer_beta1 is not None:
            return self.optimizer_beta1
        if self.method in SOFTSERVE_METHODS or self.method == "sgd_m":
            return self.beta1
        if self.method == "adamw":
            return self.adam_betas[0]
        if self.method == "muon":
            return self.muon_momentum
        if self.method == "soap":
            return 0.95
        raise ValueError(f"unknown method: {self.method}")

    @property
    def effective_softserve_nesterov(self) -> bool:
        return False if self.softserve_nesterov is None else self.softserve_nesterov

    @property
    def effective_soap_routing(self) -> SOAPRouting:
        return "canonical" if self.soap_routing is None else self.soap_routing

    @property
    def tokens_per_update(self) -> int:
        return self.physical_batch * self.grad_accum * self.block_size

    @property
    def updates(self) -> int:
        return math.ceil(self.target_tokens / self.tokens_per_update)

    def model_config(self) -> GPTConfig:
        return GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            layers=self.layers,
            heads=self.heads,
            width=self.width,
            dropout=0.0,
            bias=True,
        )

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if (
            min(
                self.lr,
                self.fallback_lr,
                self.target_tokens,
                self.schedule_steps,
                self.physical_batch,
                self.grad_accum,
                self.block_size,
                self.eval_every,
                self.monitor_validation_tokens,
                self.final_validation_tokens,
                self.final_train_tokens,
                self.validation_batch,
                self.refresh_interval,
            )
            <= 0
        ):
            raise ValueError("learning rates, budgets, and sizes must be positive")
        if not 0.0 < self.lr_decay_fraction <= 1.0:
            raise ValueError("lr_decay_fraction must lie in (0, 1]")
        if self.tau is not None and self.initial_lambda is not None:
            raise ValueError("tau and public lambda are mutually exclusive")
        if self.initial_lambda is not None and (
            not math.isfinite(self.initial_lambda) or self.initial_lambda <= 0.0
        ):
            raise ValueError("lambda must be finite and positive")
        if self.method in SOFTSERVE_METHODS:
            literal_public_fixed = True and self.initial_lambda is not None and (self.tau is None)
            if not literal_public_fixed and (self.tau is None or not 0.0 < self.tau < 1.0):
                raise ValueError(
                    "fixed and scheduled SoftSERVE require tau in (0, 1), or a public lambda for a literal fixed schedule"
                )
        elif self.tau is not None or self.initial_lambda is not None or False or False:
            raise ValueError("lambda options are only defined for SoftSERVE")
        if self.method == "soap":
            if self.soap_path is None:
                raise ValueError("canonical SOAP requires soap_path")
            soap_path = Path(self.soap_path).expanduser()
            if not soap_path.is_file():
                raise ValueError("soap_path must name an existing file")
            if self.effective_soap_routing not in {"canonical", "matched"}:
                raise ValueError("soap_routing must be canonical or matched")
        elif self.soap_path is not None or self.soap_routing is not None:
            raise ValueError("SOAP options are only defined for SOAP")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown direction normalization: {self.normalization}")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must lie in [0, 1)")
        if self.optimizer_beta1 is not None and (not 0.0 <= self.optimizer_beta1 < 1.0):
            raise ValueError("optimizer_beta1 must lie in [0, 1)")
        if self.softserve_nesterov is not None:
            if not isinstance(self.softserve_nesterov, bool):
                raise ValueError("softserve_nesterov must be Boolean")
            if self.method not in SOFTSERVE_METHODS:
                raise ValueError("softserve_nesterov is only defined for SoftSERVE")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must lie in [0, 1)")
        if self.target_tokens > Path(self.train_path).stat().st_size // 2 - 1:
            raise ValueError("training token stream is too short")
        if self.final_validation_tokens > Path(self.validation_path).stat().st_size // 2 - 1:
            raise ValueError("validation token stream is too short")
        self.model_config()


@dataclass(frozen=True)
class Span:
    start: int
    length: int


@dataclass
class Gradient:
    loss: float
    token_count: int
    values: tuple[torch.Tensor, ...]
    replicas: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None = None


def _identity_config(config: GPT2RunConfig) -> dict[str, Any]:
    """Serialize config identity without changing legacy IDs for omitted knobs."""
    payload = asdict(config)
    for name in ("optimizer_beta1", "softserve_nesterov", "soap_routing"):
        if payload[name] is None:
            payload.pop(name)
    return payload


def _run_id(config: GPT2RunConfig) -> str:
    payload = json.dumps(_identity_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _spans(config: GPT2RunConfig, step: int) -> list[list[Span]]:
    start = step * config.tokens_per_update
    take = min(config.tokens_per_update, config.target_tokens - start)
    if take <= 0:
        raise IndexError("training token stream exhausted")
    spans, consumed = ([], 0)
    while consumed < take:
        length = min(config.block_size, take - consumed)
        spans.append(Span(start + consumed, length))
        consumed += length
    batches: list[list[Span]] = []
    current: list[Span] = []
    for span in spans:
        if current and (len(current) == config.physical_batch or current[0].length != span.length):
            batches.append(current)
            current = []
        current.append(span)
    if current:
        batches.append(current)
    return batches


def _amp(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _gradient(
    model: NanoGPT,
    data: TokenFile,
    batches: list[list[Span]],
    device: torch.device,
    *,
    replicated: bool = False,
) -> Gradient:
    if replicated and (len(batches) < 2 or len(batches) % 2):
        raise ValueError("replicated gradients require an even number of microbatches")
    model.zero_grad(set_to_none=True)
    token_count = sum((len(batch) * batch[0].length for batch in batches))
    loss_sum = 0.0
    split = len(batches) // 2
    split_values: tuple[torch.Tensor, ...] | None = None
    split_tokens = 0
    for index, batch in enumerate(batches):
        values, targets = data.batch(
            [span.start for span in batch], device=device, length=batch[0].length
        )
        with _amp(device):
            _, losses = model(values, targets, loss_reduction="sum")
        assert losses is not None
        (losses / token_count).backward()
        loss_sum += float(losses.detach())
        if replicated and index + 1 == split:
            split_tokens = sum((len(item) * item[0].length for item in batches[:split]))
            split_values = tuple(
                (
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                    for parameter in model.parameters()
                )
            )
    gradients = tuple(
        (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in model.parameters()
        )
    )
    replicas = None
    if replicated:
        assert split_values is not None and 0 < split_tokens < token_count
        right_tokens = token_count - split_tokens
        left = tuple((value * (token_count / split_tokens) for value in split_values))
        right = tuple(
            (
                (value - partial) * (token_count / right_tokens)
                for value, partial in zip(gradients, split_values, strict=True)
            )
        )
        replicas = (left, right)
    return Gradient(loss_sum / token_count, token_count, gradients, replicas)


@torch.no_grad()
def _evaluate_range(
    model: NanoGPT,
    data: TokenFile,
    *,
    start: int,
    token_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss, correct, consumed = (0.0, 0, 0)
    while consumed < token_count:
        remaining = token_count - consumed
        length = min(data.block_size, remaining)
        count = min(batch_size, remaining // length)
        starts = [start + consumed + index * length for index in range(count)]
        values, targets = data.batch(starts, device=device, length=length)
        with _amp(device):
            logits, loss = model(values, targets, loss_reduction="sum")
        assert loss is not None
        used = len(starts) * length
        total_loss += float(loss)
        correct += int((logits.argmax(dim=-1) == targets).sum())
        consumed += used
    model.train(was_training)
    return (total_loss / token_count, correct / token_count)


def _schedule(step: int, peak: float, config: GPT2RunConfig) -> float:
    warmup = math.ceil(0.02 * config.schedule_steps)
    if step < warmup:
        return peak * (step + 1) / warmup
    decay = math.ceil(config.lr_decay_fraction * config.schedule_steps)
    start = max(warmup, config.schedule_steps - decay)
    if step < start:
        return peak
    progress = min(1.0, (step - start) / max(1, decay - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)


def _perplexity(nll: float) -> float:
    try:
        return math.exp(nll)
    except OverflowError:
        return math.inf


def _assign(parameters: tuple[nn.Parameter, ...], values: tuple[torch.Tensor, ...]) -> None:
    for parameter, value in zip(parameters, values, strict=True):
        parameter.grad = value.detach().clone()


def _adamw(
    parameters,
    lr: float,
    config: GPT2RunConfig,
    device: torch.device,
    *,
    beta1: float | None = None,
):
    betas = config.adam_betas if beta1 is None else (beta1, config.adam_betas[1])
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=betas,
        eps=config.adam_epsilon,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )


def _soap_class(path: str):
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("softserve_canonical_soap", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical SOAP from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    soap = getattr(module, "SOAP", None)
    if soap is None:
        raise RuntimeError(f"canonical SOAP class is missing from {resolved}")
    return soap


class Controller:
    def __init__(self, model: NanoGPT, config: GPT2RunConfig, device: torch.device) -> None:
        self.config = config
        self.parameters = tuple(model.parameters())
        matrices, fallback = matrix_routes(model)
        self.main_parameters = tuple(matrices)
        self.fallback_parameters = tuple(fallback)
        self.qn: SoftServeDiag | SoftServeKron | None = None
        self.fallback: torch.optim.Optimizer | None = None
        if config.method == "adamw":
            self.main_parameters, self.fallback_parameters = (self.parameters, ())
            self.main = _adamw(
                self.parameters, config.lr, config, device, beta1=config.effective_beta1
            )
        elif config.method in SOFTSERVE_METHODS:
            constrained = False or config.normalization != "none"
            common_options: dict[str, Any] = {
                "lr": config.lr,
                "lam": 1.0 if config.lam is None else float(config.lam),
                "lambda_schedule": "fixed",
                "beta1": config.effective_beta1,
                "beta_sy": 0.0,
                "beta_h": 0.0,
                "T": 1,
                "nesterov": config.effective_softserve_nesterov,
                "constrained_update": constrained,
                "metric_rms_constraint": config.normalization == "rms",
                "pair_diagnostics": True,
            }
            if config.method == "softserve_kron":
                self.qn = SoftServeKron(
                    self.main_parameters,
                    backend="gemm",
                    root_steps=18,
                    inverse_steps=10,
                    normalize=True,
                    gauge="balanced_trace",
                    bucket_chunk_size=None,
                    **common_options,
                )
            else:
                self.qn = SoftServeDiag(self.main_parameters, initial_h=1.0, **common_options)
            self.main = self.qn
            self.fallback = _adamw(self.fallback_parameters, config.fallback_lr, config, device)
        elif config.method == "sgd_m":
            self.main = SGDM(
                self.main_parameters,
                lr=config.lr,
                beta=config.effective_beta1,
                normalization=config.normalization,
            )
            self.fallback = _adamw(self.fallback_parameters, config.fallback_lr, config, device)
        elif config.method == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("this experiment requires torch.optim.Muon")
            self.main = torch.optim.Muon(
                self.main_parameters,
                lr=config.lr,
                weight_decay=0.0,
                momentum=config.effective_beta1,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
            self.fallback = _adamw(self.fallback_parameters, config.fallback_lr, config, device)
        elif config.method == "soap":
            assert config.soap_path is not None
            soap = _soap_class(config.soap_path)
            if config.effective_soap_routing == "canonical":
                self.main_parameters, self.fallback_parameters = (self.parameters, ())
            self.main = soap(
                self.main_parameters,
                lr=config.lr,
                betas=(config.effective_beta1, 0.95),
                shampoo_beta=-1,
                eps=1e-08,
                weight_decay=0.0,
                precondition_frequency=config.refresh_interval,
                max_precond_dim=10000,
                merge_dims=False,
                precondition_1d=False,
                normalize_grads=False,
                correct_bias=True,
            )
            if config.effective_soap_routing == "matched":
                self.fallback = _adamw(self.fallback_parameters, config.fallback_lr, config, device)

    def set_lrs(self, step: int) -> tuple[float, float]:
        main = _schedule(step, self.config.lr, self.config)
        fallback = _schedule(step, self.config.fallback_lr, self.config)
        for group in self.main.param_groups:
            group["lr"] = main
        if self.fallback is not None:
            for group in self.fallback.param_groups:
                group["lr"] = fallback
        return (main, fallback)

    def parameter_step(self) -> None:
        if self.qn is not None:
            self.qn.parameter_step()
        else:
            self.main.step()
        if self.fallback is not None:
            self.fallback.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "main": self.main.state_dict(),
            "fallback": None if self.fallback is None else self.fallback.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.main.load_state_dict(state["main"])
        if self.fallback is not None:
            self.fallback.load_state_dict(state["fallback"])


def _state_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum((_state_bytes(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return sum((_state_bytes(item) for item in value))
    return 0


def _lambda_diagnostics(
    optimizer: SoftServeDiag | SoftServeKron,
    *,
    step: int,
    gradient_calls: int,
    unique_tokens: int,
    learning_rate: float,
) -> dict[str, Any]:
    lambdas = [float(value) for value in optimizer.last_effective_lambda]
    pair_q = [float(value) for value in optimizer.last_pair_q]
    return {
        "parameter_steps": step,
        "gradient_evaluations": gradient_calls,
        "unique_tokens": unique_tokens,
        "learning_rate": learning_rate,
        "effective_lambda": lambdas,
        "pair_q": pair_q,
        "pair_theta": [2.0 * lam * q for lam, q in zip(lambdas, pair_q, strict=True)],
        "pair_chi": [float(value) for value in optimizer.last_pair_chi],
        "pair_gate_rejected": [int(value) for value in optimizer.last_pair_gate_rejected],
        "qme_residual_A": [float(value) for value in getattr(optimizer, "last_qme_residual_A", [])],
        "qme_residual_G": [float(value) for value in getattr(optimizer, "last_qme_residual_G", [])],
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(
    config: GPT2RunConfig,
    *,
    campaign: str | None = None,
    worker_id: str | int = 0,
    observer: Any | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    model = NanoGPT(config.model_config()).to(device)
    train = TokenFile(config.train_path, config.block_size)
    validation = TokenFile(config.validation_path, config.block_size)
    parameters = tuple(model.parameters())
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    controller = Controller(model, config, device)
    route = controller.main_parameters
    secant = IntervalSecant(config.refresh_interval) if controller.qn else None
    store = CampaignStore(campaign) if campaign is not None else None
    run_id = _run_id(config)
    resume = (
        None
        if store is None or config.checkpoint_every == 0
        else store.path / "resume" / f"{run_id}.pt"
    )
    history: list[dict[str, Any]] = []
    scheduler_history: list[dict[str, Any]] = []
    step = calls = endpoints = curvature_updates = backward_tokens = 0
    accepted_curvature_updates = 0
    metric_frozen = False
    metric_control_stats: dict[str, int | str] = {
        "metric_control": "real",
        "metric_control_shuffled_matrices": 0,
        "metric_control_identity_matrices": 0,
    }
    microbatch_backwards = 0
    negative_pairs = pair_count = 0
    qme_residual_max = 0.0
    trace_lambda = False
    if observer is not None:
        observer.initialize(
            model=model,
            train=train,
            validation=validation,
            controller=controller,
            config=config,
            device=device,
        )
    expected_curvature_updates = config.updates // config.refresh_interval
    prior_wall = 0.0
    evaluation_seconds = 0.0
    checkpoint_io_seconds = 0.0
    if resume is not None and resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["config"] != _identity_config(config):
            raise RuntimeError("resume checkpoint configuration mismatch")
        model.load_state_dict(saved["model"])
        controller.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        calls = int(saved["calls"])
        endpoints = int(saved["endpoints"])
        curvature_updates = int(saved["curvature_updates"])
        accepted_curvature_updates = int(saved.get("accepted_curvature_updates", curvature_updates))
        metric_frozen = bool(saved.get("metric_frozen", False))
        metric_control_stats = dict(saved.get("metric_control_stats", metric_control_stats))
        backward_tokens = int(saved["backward_tokens"])
        microbatch_backwards = int(saved.get("microbatch_backwards", 0))
        negative_pairs = int(saved["negative_pairs"])
        pair_count = int(saved["pair_count"])
        qme_residual_max = float(saved["qme_residual_max"])
        history = saved["history"]
        scheduler_history = saved.get("scheduler_history", [])
        prior_wall = float(saved["wall_seconds"])
        if metric_frozen:
            secant = None
    setup_seconds = time.perf_counter() - run_started
    started = time.perf_counter()

    def wall() -> float:
        return setup_seconds + prior_wall + time.perf_counter() - started

    def record(training_loss: float | None) -> None:
        nonlocal evaluation_seconds
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_started = time.perf_counter()
        validation_loss, validation_accuracy = _evaluate_range(
            model,
            validation,
            start=0,
            token_count=config.monitor_validation_tokens,
            batch_size=config.validation_batch,
            device=device,
        )
        evaluation_seconds += time.perf_counter() - evaluation_started
        point = {
            "parameter_steps": step,
            "gradient_evaluations": calls,
            "endpoint_evaluations": endpoints,
            "curvature_updates": curvature_updates,
            "unique_tokens": min(step * config.tokens_per_update, config.target_tokens),
            "backward_tokens": backward_tokens,
            "microbatch_backwards": microbatch_backwards,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_perplexity": _perplexity(validation_loss),
            "validation_token_accuracy": validation_accuracy,
            "wall_seconds": wall(),
            "training_wall_seconds": max(0.0, wall() - evaluation_seconds - checkpoint_io_seconds),
        }
        if observer is not None:
            additions = observer.monitor(step=step, training_loss=training_loss)
            overlap = set(point) & set(additions)
            if overlap:
                raise RuntimeError(f"observer monitor fields overlap: {sorted(overlap)}")
            point.update(additions)
        history.append(point)

    if not history:
        record(None)
    status, reason = ("complete", None)
    last_training_loss = None
    try:
        while step < config.updates:
            batches = _spans(config, step)
            current = _gradient(model, train, batches, device, replicated=False)
            calls += 1
            backward_tokens += current.token_count
            microbatch_backwards += len(batches)
            last_training_loss = current.loss
            if not math.isfinite(current.loss):
                raise FloatingPointError("non-finite training loss")
            _assign(parameters, current.values)
            if secant is not None and secant.starts_at(step):
                secant.start(
                    route, tuple((parameter.grad.detach() for parameter in route)), batch_key=step
                )
            main_learning_rate, _ = controller.set_lrs(step)
            if observer is not None:
                observer.before_parameter_step(
                    step=step, batches=batches, gradient=current, learning_rate=main_learning_rate
                )
            controller.parameter_step()
            if observer is not None:
                observer.after_parameter_step(
                    step=step, batches=batches, gradient=current, learning_rate=main_learning_rate
                )
            if secant is not None and secant.ends_at(step):
                origin = int(secant.batch_key)
                endpoint_batches = _spans(config, origin)
                endpoint = _gradient(model, train, endpoint_batches, device, replicated=False)
                calls += 1
                endpoints += 1
                backward_tokens += endpoint.token_count
                microbatch_backwards += len(endpoint_batches)
                endpoint_values = endpoint.values
                endpoint_route = tuple(
                    (endpoint_values[parameter_indices[id(parameter)]] for parameter in route)
                )
                s_values, y_values = secant.pair(route, endpoint_route)
                controller.qn.update_curvature(s_values, y_values)
                curvature_updates += 1
                rejected = controller.qn.last_pair_gate_rejected
                accepted = not rejected or any((not bool(value) for value in rejected))
                if accepted:
                    accepted_curvature_updates += 1
                chi = [float(value) for value in controller.qn.last_pair_chi]
                negative_pairs += sum((value < 0 for value in chi))
                pair_count += len(chi)
                residuals = list(getattr(controller.qn, "last_qme_residual_A", [])) + list(
                    getattr(controller.qn, "last_qme_residual_G", [])
                )
                finite_residuals = [
                    float(value) for value in residuals if bool(torch.isfinite(value))
                ]
                if finite_residuals:
                    qme_residual_max = max(qme_residual_max, max(finite_residuals))
                completed = step + 1
                if trace_lambda and (
                    curvature_updates <= 10
                    or completed % config.eval_every == 0
                    or curvature_updates == expected_curvature_updates
                ):
                    diagnostic_args = {
                        "step": completed,
                        "gradient_calls": calls,
                        "unique_tokens": min(
                            completed * config.tokens_per_update, config.target_tokens
                        ),
                    }
                    scheduler_history.append(
                        _lambda_diagnostics(
                            controller.qn, learning_rate=main_learning_rate, **diagnostic_args
                        )
                    )
                del s_values, y_values, endpoint_route, endpoint_values, endpoint
            step += 1
            if step % config.eval_every == 0 or step == config.updates:
                record(last_training_loss)
            if (
                resume is not None
                and config.checkpoint_every
                and (step % config.checkpoint_every == 0)
            ):
                checkpoint_started = time.perf_counter()
                _atomic_save(
                    {
                        "config": _identity_config(config),
                        "model": model.state_dict(),
                        "optimizer": controller.state_dict(),
                        "step": step,
                        "calls": calls,
                        "endpoints": endpoints,
                        "curvature_updates": curvature_updates,
                        "accepted_curvature_updates": accepted_curvature_updates,
                        "metric_frozen": metric_frozen,
                        "metric_control_stats": metric_control_stats,
                        "backward_tokens": backward_tokens,
                        "microbatch_backwards": microbatch_backwards,
                        "negative_pairs": negative_pairs,
                        "pair_count": pair_count,
                        "qme_residual_max": qme_residual_max,
                        "history": history,
                        "scheduler_history": scheduler_history,
                        "wall_seconds": wall(),
                    },
                    resume,
                )
                checkpoint_io_seconds += time.perf_counter() - checkpoint_started
    except (FloatingPointError, RuntimeError, torch.linalg.LinAlgError) as error:
        status, reason = ("failed", repr(error))
    if status == "complete":
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_started = time.perf_counter()
        final_validation_loss, final_validation_accuracy = _evaluate_range(
            model,
            validation,
            start=0,
            token_count=config.final_validation_tokens,
            batch_size=config.validation_batch,
            device=device,
        )
        final_train_loss, _ = _evaluate_range(
            model,
            train,
            start=max(0, config.target_tokens - config.final_train_tokens),
            token_count=min(config.final_train_tokens, config.target_tokens),
            batch_size=config.validation_batch,
            device=device,
        )
        evaluation_seconds += time.perf_counter() - evaluation_started
    else:
        final_validation_loss = math.inf
        final_validation_accuracy = 0.0
        final_train_loss = math.inf
    persistent_state_bytes = _state_bytes(controller.state_dict())
    secant_buffer_bytes = (
        0 if secant is None else _state_bytes(secant.parameters) + _state_bytes(secant.gradients)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_to_end_seconds = time.perf_counter() - run_started
    row = {
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "problem": f"gpt2_l{config.layers}_h{config.heads}_d{config.width}_fineweb",
        "phase": config.phase,
        "method": config.method,
        "seed": config.seed,
        "learning_rate": config.lr,
        "fallback_learning_rate": config.fallback_lr,
        "direction_normalization": config.normalization,
        "beta1": config.beta1,
        "optimizer_beta1": config.optimizer_beta1,
        "effective_beta1": config.effective_beta1,
        "softserve_nesterov": config.effective_softserve_nesterov
        if config.method in SOFTSERVE_METHODS
        else None,
        "muon_momentum": config.muon_momentum,
        "soap_routing": config.effective_soap_routing if config.method == "soap" else None,
        "refresh_interval": config.refresh_interval,
        "freeze_curvature_after": None,
        "accepted_curvature_updates": accepted_curvature_updates,
        "metric_frozen": metric_frozen,
        **metric_control_stats,
        "tau": config.tau,
        "lambda": config.lam,
        "lambda_schedule": "fixed",
        "lambda_adaptive": None,
        "lambda_cap": None,
        "internal_lambda_schedule": "fixed",
        "literal_fixed_lambda": controller.qn is not None and True and True,
        "lambda_scope": "block",
        "parameter_steps": step,
        "gradient_evaluations": calls,
        "endpoint_evaluations": endpoints,
        "curvature_updates": curvature_updates,
        "unique_tokens": min(step * config.tokens_per_update, config.target_tokens),
        "backward_tokens": backward_tokens,
        "microbatch_backwards": microbatch_backwards,
        "final_train_nll": final_train_loss,
        "final_validation_nll": final_validation_loss,
        "final_validation_perplexity": _perplexity(final_validation_loss),
        "final_validation_token_accuracy": final_validation_accuracy,
        "selection_score": final_validation_loss,
        "negative_secant_fraction": negative_pairs / pair_count if pair_count else 0.0,
        "qme_residual_max": qme_residual_max if isinstance(controller.qn, SoftServeKron) else None,
        "wall_seconds": wall(),
        "end_to_end_seconds": end_to_end_seconds,
        "evaluation_seconds": evaluation_seconds,
        "setup_seconds": setup_seconds,
        "checkpoint_io_seconds": checkpoint_io_seconds,
        "non_evaluation_seconds": max(
            0.0, end_to_end_seconds - evaluation_seconds - checkpoint_io_seconds
        ),
        "parameter_count": sum((parameter.numel() for parameter in parameters)),
        "optimizer_persistent_state_bytes": persistent_state_bytes,
        "secant_buffer_bytes": secant_buffer_bytes,
        "optimizer_state_bytes": persistent_state_bytes + secant_buffer_bytes,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config),
        "history": history,
        "scheduler_history": scheduler_history,
    }
    if observer is not None:
        additions = observer.result_fields()
        overlap = set(row) & set(additions)
        if overlap:
            raise RuntimeError(f"observer result fields overlap: {sorted(overlap)}")
        row.update(additions)
    if store is not None:
        store.write_shard(worker_id, [row])
        if status == "complete" and resume is not None:
            resume.unlink(missing_ok=True)
        if config.selected and status == "complete":
            _atomic_save(
                {
                    "format": "softserve-gpt2-v1",
                    "config": _identity_config(config),
                    "model": model.state_dict(),
                    "summary": {key: value for key, value in row.items() if key != "history"},
                },
                store.checkpoints / f"{run_id}.pt",
            )
    return row
