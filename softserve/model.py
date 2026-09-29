"""Model-aware SoftSERVE with same-realization interval replay.

The numerical optimizers in optim.py remain the engines. This layer owns
parameter routing, closure replay, and the optional Adam/AdamW/Diag fallback.
"""

from contextlib import contextmanager
import copy
import inspect
import math
import random

import numpy as np
import torch
from torch.optim import Optimizer

from .optim import SoftServeDiag, SoftServeKron
from .routing import ModelRoutes


def _rng_state(device):
    return dict(
        cpu=torch.get_rng_state().clone(),
        cuda=torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
        python=random.getstate(),
        numpy=np.random.get_state(),
    )


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


class _Realization:
    def __init__(self, model, device):
        self.rng = _rng_state(device)
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        self.training = {name: module.training for name, module in model.named_modules()}
        self.autocast = torch.is_autocast_enabled(device.type)
        self.autocast_dtype = torch.get_autocast_dtype(device.type)

    @torch.no_grad()
    def install(self, model, device):
        buffers = dict(model.named_buffers())
        modules = dict(model.named_modules())
        if buffers.keys() != self.buffers.keys() or modules.keys() != self.training.keys():
            raise RuntimeError("Modules/buffers changed during the replay interval")
        for name, value in self.buffers.items():
            if buffers[name].shape != value.shape or buffers[name].dtype != value.dtype:
                raise RuntimeError("Buffer shape/dtype changed during the replay interval")
            buffers[name].copy_(value)
        for name, flag in self.training.items():
            modules[name].training = flag
        _restore_rng(self.rng, device)

    @contextmanager
    def replay(self, model, device):
        current = _Realization(model, device)
        try:
            self.install(model, device)
            with torch.autocast(
                device_type=device.type, enabled=self.autocast, dtype=self.autocast_dtype
            ):
                yield
        finally:
            current.install(model, device)


class _ModelSoftServe(Optimizer):
    def __init__(
        self,
        model,
        lr,
        lam=99.0,
        *,
        beta1=0.9,
        K=10,
        fallback="softserve-diag",
        fallback_lr=None,
        fallback_betas=(0.9, 0.999),
        fallback_weight_decay=0.0,
        max_preconditioner_dim=256,
        max_grad_norm=None,
        kind="kron",
        **core_options,
    ):
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise ValueError("K must be a positive integer")
        if fallback not in ("adam", "adamw", "softserve-diag"):
            raise ValueError("fallback must be 'adam', 'adamw', or 'softserve-diag'")
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if any(
            not math.isfinite(v) or v < 0 for v in (lr, fallback_lr, lam, fallback_weight_decay)
        ):
            raise ValueError("Learning rates, lam, and weight decay must be finite and nonnegative")
        if (
            not 0 <= beta1 < 1
            or len(fallback_betas) != 2
            or not all(0 <= b < 1 for b in fallback_betas)
        ):
            raise ValueError("Momentum coefficients must lie in [0, 1)")
        if core_options.get("T", 1) != 1:
            raise ValueError("Use K for model-mode refresh intervals; T must remain 1")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite")
        if fallback == "softserve-diag" and fallback_weight_decay:
            raise ValueError("fallback_weight_decay applies only to Adam/AdamW")
        engine = SoftServeDiag if kind == "diag" else SoftServeKron
        allowed = set(inspect.signature(engine.__init__).parameters) - {
            "self",
            "params",
            "lr",
            "lam",
            "beta1",
        }
        unknown = set(core_options) - allowed
        if unknown:
            raise TypeError(f"Unknown SoftSERVE options: {sorted(unknown)}")
        self.model, self.K, self.kind = model, K, kind
        self.fallback_name = "softserve-diag" if kind == "diag" else fallback
        self.max_grad_norm = max_grad_norm
        self.routes = ModelRoutes(model, max_preconditioner_dim, diagonal=kind == "diag")
        self.routing = copy.deepcopy(self.routes.description)
        self._positions = {id(p): i for i, p in enumerate(self.routes.parameters)}
        self._pending = None
        self._mode = None
        self.steps = self.gradient_evaluations = self.curvature_refreshes = 0
        core = dict(lr=lr, lam=lam, beta1=beta1, **core_options)
        self.kron_optimizer = (
            SoftServeKron([b.proxy for b in self.routes.blocks], **core)
            if self.routes.blocks
            else None
        )
        self.fallback_optimizer = None
        if self.routes.fallback:
            if self.fallback_name == "softserve-diag":
                shared = (
                    "beta_sy",
                    "beta_h",
                    "nesterov",
                    "constrained_update",
                    "metric_rms_constraint",
                    "update_rms",
                    "global_update_rms",
                    "pair_diagnostics",
                    "lambda_schedule",
                )
                diagonal_options = (
                    core_options
                    if kind == "diag"
                    else {k: v for k, v in core_options.items() if k in shared}
                )
                self.fallback_optimizer = SoftServeDiag(
                    self.routes.fallback,
                    lr=lr if kind == "diag" else fallback_lr,
                    lam=lam,
                    beta1=beta1,
                    **diagonal_options,
                )
            else:
                cls = torch.optim.Adam if fallback == "adam" else torch.optim.AdamW
                self.fallback_optimizer = cls(
                    self.routes.fallback,
                    lr=fallback_lr,
                    betas=fallback_betas,
                    weight_decay=fallback_weight_decay,
                )
        groups = []
        for name, child, params in (
            ("kron", self.kron_optimizer, self.routes.main),
            ("fallback", self.fallback_optimizer, self.routes.fallback),
        ):
            if child is not None:
                group = {k: v for k, v in child.param_groups[0].items() if k != "params"}
                group.update(params=params, name=name)
                groups.append(group)
        self._building_groups = True
        super().__init__(groups, dict(lr=lr))
        self._building_groups = False

    def add_param_group(self, param_group):
        if not getattr(self, "_building_groups", False):
            raise RuntimeError(
                "Model-mode routing is fixed; reconstruct the optimizer to add parameters"
            )
        super().add_param_group(param_group)

    @property
    def checkpoint_ready(self):
        """Stochastic closures cannot be serialized; save at interval boundaries."""
        return self._mode != "stochastic" or self._pending is None

    def _sync_groups(self):
        groups = {g["name"]: g for g in self.param_groups}
        if (
            self.kind == "kron"
            and self.fallback_name == "softserve-diag"
            and "kron" in groups
            and "fallback" in groups
        ):
            for key in ("lam", "beta1", "beta_sy", "beta_h", "nesterov"):
                groups["fallback"][key] = groups["kron"][key]
        for name, child in (("kron", self.kron_optimizer), ("fallback", self.fallback_optimizer)):
            if child is not None:
                for key in child.param_groups[0]:
                    if key != "params" and key in groups[name]:
                        child.param_groups[0][key] = groups[name][key]

    def _gradients(self):
        values = [p.grad if p.requires_grad else None for p in self.routes.parameters]
        present = [g for g in values if g is not None]
        if any(g.layout != torch.strided for g in present):
            raise RuntimeError(
                "Sparse gradients are unsupported in model mode; use dense embeddings or a separate optimizer"
            )
        if present and not bool(torch.stack([g.isfinite().all() for g in present]).all()):
            raise FloatingPointError("Nonfinite gradients")
        return values

    def _snapshot(self, gradients, *, closure=None, realization=None):
        qn_ids = {id(p) for p in self.routes.main}
        if self.fallback_name == "softserve-diag":
            qn_ids.update(id(p) for p in self.routes.fallback)
        entries = {}
        for i, (p, g) in enumerate(zip(self.routes.parameters, gradients, strict=True)):
            if id(p) in qn_ids and g is not None:
                entries[i] = (p.detach().clone(), g.detach().clone())
        if entries:
            self._pending = dict(
                start=self.steps, entries=entries, closure=closure, realization=realization
            )

    @torch.no_grad()
    def _refresh(self, gradients):
        entries = self._pending["entries"]

        def pair(parameter):
            i = self._positions[id(parameter)]
            if i not in entries or gradients[i] is None:
                return None
            old_p, old_g = entries[i]
            return parameter.detach() - old_p, gradients[i].detach() - old_g

        if self.kron_optimizer is not None:
            params, s, y = [], [], []
            # Form a full displacement once per real parameter, not per block.
            pairs = {id(p): pair(p) for p in self.routes.main}
            for block in self.routes.blocks:
                value = pairs[id(block.parameter)]
                if value is not None:
                    params.append(block.proxy)
                    s.append(block.view(value[0]))
                    y.append(block.view(value[1]))
            if params:
                child = self.kron_optimizer
                child._feed_pair(child.param_groups[0], params, s, y)
        if isinstance(self.fallback_optimizer, SoftServeDiag):
            values = [(p, pair(p)) for p in self.routes.fallback]
            values = [(p, v) for p, v in values if v is not None]
            if values:
                child = self.fallback_optimizer
                child._feed_pair(
                    child.param_groups[0],
                    [p for p, _ in values],
                    [v[0] for _, v in values],
                    [v[1] for _, v in values],
                )
        self.curvature_refreshes += 1

    @torch.no_grad()
    def _parameter_update(self):
        for p in self.routes.parameters:
            if not p.requires_grad:
                p.grad = None
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.routes.parameters, self.max_grad_norm, error_if_nonfinite=True
            )
        self.routes.assign_proxy_gradients()
        if self.kron_optimizer is not None:
            self.kron_optimizer.parameter_step()
        if isinstance(self.fallback_optimizer, SoftServeDiag):
            self.fallback_optimizer.parameter_step()
        elif self.fallback_optimizer is not None:
            self.fallback_optimizer.step()

    def _evaluate(self, closure):
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("closure must call backward() and return a scalar loss Tensor")
        if not bool(loss.detach().isfinite().all()):
            raise FloatingPointError("Nonfinite closure loss")
        self.gradient_evaluations += 1
        return loss.detach()

    @torch.no_grad()
    def step(self, closure=None):
        """Take one update. A closure is required for stochastic objectives.

        The closure calls backward() and binds its batch (e.g. default
        arguments). It must remain valid, with unchanged inputs/loss settings,
        until the interval ends. Standard RNGs and model buffers are replayed
        internally. Without a closure, the loss must be deterministic.
        """
        if (
            getattr(self, "grad_scale", None) is not None
            or getattr(self, "found_inf", None) is not None
        ):
            raise RuntimeError("GradScaler is not supported by model mode; use unscaled gradients")
        self.routes.validate()
        mode = "deterministic" if closure is None else "stochastic"
        if self._mode is not None and self._mode != mode:
            raise RuntimeError("Do not mix closure and closure-free steps on one optimizer")
        self._mode = mode
        self._sync_groups()
        realization = (
            _Realization(self.model, self.routes.device)
            if closure is not None and self._pending is None
            else None
        )
        loss = self._evaluate(closure) if closure is not None else None
        gradients = self._gradients()
        if not any(g is not None for g in gradients):
            if closure is not None:
                raise RuntimeError("closure produced no gradients; it must call backward()")
            return loss
        if closure is None:
            self.gradient_evaluations += 1
            if self._pending is not None and self.steps - self._pending["start"] >= self.K:
                with torch.autocast(device_type=self.routes.device.type, enabled=False):
                    self._refresh(gradients)
                self._pending = None
        if self._pending is None:
            self._snapshot(gradients, closure=closure, realization=realization)
        # Autocast may surround step(), but optimizer/QME arithmetic must
        # still use the stored parameter precision, not BF16/FP16 matmuls.
        with torch.autocast(device_type=self.routes.device.type, enabled=False):
            self._parameter_update()
        # An outer autocast context can otherwise reuse pre-update weight
        # casts for the endpoint forward pass (or the next training step).
        torch.clear_autocast_cache()
        self.steps += 1
        if (
            closure is not None
            and self._pending is not None
            and self.steps - self._pending["start"] >= self.K
        ):
            # Preserve current .grad values as well as RNG/buffers. Endpoint
            # gradients are raw, even when parameter updates use clipping.
            current_gradients = [p.grad for p in self.routes.parameters]
            try:
                with self._pending["realization"].replay(self.model, self.routes.device):
                    self._evaluate(self._pending["closure"])
                    with torch.autocast(device_type=self.routes.device.type, enabled=False):
                        self._refresh(self._gradients())
            finally:
                for p, g in zip(self.routes.parameters, current_gradients, strict=True):
                    p.grad = g
                self._pending = None
                self.routes.assign_proxy_gradients()
        return loss

    def state_dict(self):
        if not self.checkpoint_ready:
            raise RuntimeError(
                "A stochastic replay closure is pending and cannot be serialized. Save after a K-step refresh (optimizer.checkpoint_ready)."
            )
        state = super().state_dict()
        pending = None
        if self._pending is not None:
            pending = dict(start=self._pending["start"], entries=self._pending["entries"])
        state["softserve_model"] = dict(
            version=1,
            kind=self.kind,
            K=self.K,
            fallback=self.fallback_name,
            routing=self.routing,
            max_preconditioner_dim=self.routes.max_dim,
            max_grad_norm=self.max_grad_norm,
            steps=self.steps,
            gradient_evaluations=self.gradient_evaluations,
            curvature_refreshes=self.curvature_refreshes,
            mode=self._mode,
            pending=pending,
            kron=self.kron_optimizer.state_dict() if self.kron_optimizer else None,
            fallback_state=self.fallback_optimizer.state_dict()
            if self.fallback_optimizer
            else None,
        )
        return state

    def load_state_dict(self, state_dict):
        self.routes.validate()
        meta = state_dict.get("softserve_model", {})
        expected = (1, self.kind, self.K, self.fallback_name, self.routing, self.routes.max_dim)
        actual = tuple(
            meta.get(k)
            for k in ("version", "kind", "K", "fallback", "routing", "max_preconditioner_dim")
        )
        if actual != expected:
            raise ValueError(
                "Checkpoint model routing, K, or fallback does not match this optimizer"
            )
        if self._pending is not None and self._mode == "stochastic":
            raise RuntimeError("Cannot load while a stochastic replay is pending")
        super().load_state_dict(state_dict)
        for child, key in (
            (self.kron_optimizer, "kron"),
            (self.fallback_optimizer, "fallback_state"),
        ):
            if child is not None:
                child.load_state_dict(meta[key])
        self.steps = meta["steps"]
        self.gradient_evaluations = meta["gradient_evaluations"]
        self.curvature_refreshes = meta["curvature_refreshes"]
        self._mode, self.max_grad_norm = meta["mode"], meta["max_grad_norm"]
        self._pending = None
        if meta["pending"] is not None:
            pending = meta["pending"]
            entries = {
                i: tuple(
                    t.to(
                        device=self.routes.parameters[i].device,
                        dtype=self.routes.parameters[i].dtype,
                    ).clone()
                    for t in pair
                )
                for i, pair in pending["entries"].items()
            }
            self._pending = dict(
                start=pending["start"], entries=entries, closure=None, realization=None
            )
        self._sync_groups()


class ModelSoftServeKron(_ModelSoftServe):
    """Model-aware implementation selected by ``SoftServeKron(model, ...)``."""


class ModelSoftServeDiag(_ModelSoftServe):
    """Model-aware implementation selected by ``SoftServeDiag(model, ...)``."""

    def __init__(self, model, lr, lam=99.0, *, beta1=0.9, K=10, max_grad_norm=None, **core_options):
        super().__init__(
            model,
            lr,
            lam,
            beta1=beta1,
            K=K,
            max_grad_norm=max_grad_norm,
            kind="diag",
            **core_options,
        )
