"""Fixed-lambda SoftSERVE-Diag and SoftSERVE-Kron for PyTorch.

Use parameter_step() and update_curvature(s, y) for interval secants.
step() uses consecutive gradients and is intended for deterministic losses.
step(closure) evaluates both endpoints on the same cached data/randomness.
The experiment runners implement K=10 and count endpoint replays explicitly.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Sequence
import torch
from torch.optim import Optimizer
from .qme import (
    _sym,
    _eye,
    _blend,
    _rms,
    _scaled_l2_norm,
    kron_sweep,
    _root_pair_ns,
    _inverse_ns,
    gemm_kron_sweep,
    _diag_update,
    _diag_pair_metrics,
    _kron_pair_metrics,
)
from .qme import qme as qme, _gemm_qme as _gemm_qme


class _SecantOptimizer(Optimizer):
    def __init__(self, params, defaults):
        for key, value in dict(
            beta1=0.0,
            beta_sy=0.0,
            beta_h=0.0,
            T=1,
            nesterov=False,
            update_rms=None,
            global_update_rms=None,
            update_rms_eps=1e-12,
            constrained_update=False,
            metric_rms_constraint=False,
            pair_diagnostics=False,
            lambda_schedule="fixed",
        ).items():
            defaults.setdefault(key, value)
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["lr"] < 0 or group["lam"] < 0:
                raise ValueError("lr and lam must be nonnegative")
            if not all((0 <= group[k] < 1 for k in ("beta1", "beta_sy", "beta_h"))):
                raise ValueError("momentum coefficients must lie in [0, 1)")
            if group["T"] < 1:
                raise ValueError("T must be positive")
            if group["lambda_schedule"] != "fixed":
                raise ValueError("This release implements fixed lambda only")
            if group["metric_rms_constraint"] and (not group["constrained_update"]):
                raise ValueError("metric_rms_constraint requires constrained_update")
            if (
                sum(
                    [
                        group["constrained_update"],
                        group["update_rms"] is not None,
                        group["global_update_rms"] is not None,
                    ]
                )
                > 1
            ):
                raise ValueError("Choose only one update normalization")
        self.skipped = 0
        self.last_direction_identity_stats = None
        self.last_global_raw_update_rms = None
        self.last_global_scaled_update_rms = None
        self.last_one_sided_energy = self.last_one_sided_lambda = None
        self.last_one_sided_theta = None
        self.last_one_sided_capped = self.last_one_sided_skipped = False
        for name in (
            "raw_update_rms",
            "scaled_update_rms",
            "metric_denominators",
            "metric_normalization_valid",
            "pair_q",
            "pair_chi",
            "pair_gate_rejected",
            "effective_lambda",
        ):
            setattr(self, "last_" + name, [])

    def _direction(self, group, params, gradients) -> list[torch.Tensor]:
        raise NotImplementedError

    def _update_curvature(self, group, params, s_values, y_values, y_replica_values=None) -> None:
        raise NotImplementedError

    def _group_params(self, group) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        params = [parameter for parameter in group["params"] if parameter.grad is not None]
        return (params, [parameter.grad for parameter in params])

    def _momentum_gradients(self, group, params, gradients) -> list[torch.Tensor]:
        beta = group["beta1"]
        if beta == 0.0:
            self._last_step_gradients = gradients
            return gradients
        counter = self.state[group["params"][0]]
        counter["_momentum_steps"] = counter.get("_momentum_steps", 0) + 1
        correction = 1.0 - beta ** counter["_momentum_steps"]
        values = []
        for parameter, gradient in zip(params, gradients, strict=True):
            state = self.state[parameter]
            state.setdefault("momentum", torch.zeros_like(gradient))
            state["momentum"].mul_(beta).add_(gradient, alpha=1.0 - beta)
            if group["nesterov"]:
                values.append(torch.lerp(gradient, state["momentum"], beta))
            else:
                values.append(state["momentum"] / correction)
        self._last_step_gradients = values
        return values

    @staticmethod
    def _normalize_direction(
        gradient: torch.Tensor, direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        quadratic = (gradient.to(dtype) * direction.to(dtype)).sum()
        positive = torch.isfinite(quadratic) & (quadratic > 0)
        zero = torch.isfinite(quadratic) & (quadratic == 0)
        valid = positive | zero
        denominator = torch.where(
            positive,
            torch.sqrt(quadratic),
            torch.where(zero, torch.zeros_like(quadratic), torch.full_like(quadratic, torch.nan)),
        )
        inverse = torch.where(
            positive,
            torch.rsqrt(quadratic),
            torch.where(zero, torch.zeros_like(quadratic), torch.full_like(quadratic, torch.nan)),
        )
        return (direction * inverse.to(direction.dtype), denominator, valid)

    def _constrain(self, group, gradients, directions) -> list[torch.Tensor]:
        values, denominators, valid = ([], [], [])
        for gradient, direction in zip(gradients, directions, strict=True):
            value, denominator, is_valid = self._normalize_direction(gradient, direction)
            if group["metric_rms_constraint"]:
                value = value * math.sqrt(direction.numel())
            values.append(value)
            denominators.append(denominator.detach())
            valid.append(is_valid.detach())
        self.last_metric_denominators = denominators
        self.last_metric_normalization_valid = valid
        return values

    def _identity_directions(self, group, gradients) -> list[torch.Tensor]:
        if group["constrained_update"]:
            values = [self._normalize_direction(gradient, gradient)[0] for gradient in gradients]
            if group["metric_rms_constraint"]:
                values = [value * math.sqrt(value.numel()) for value in values]
            return values
        target = group["update_rms"]
        if target is None:
            return gradients
        return [
            gradient
            * (gradient.new_tensor(target) / _rms(gradient).clamp_min(group["update_rms_eps"]))
            for gradient in gradients
        ]

    def _matched_identity_directions(self, group, gradients):
        return self._identity_directions(group, gradients)

    def _record_direction_stats(self, directions, identity_directions) -> None:
        dot = torch.stack(
            [
                (direction.float() * identity.float()).sum()
                for direction, identity in zip(directions, identity_directions, strict=True)
            ]
        ).sum()
        direction_sq = torch.stack(
            [direction.float().square().sum() for direction in directions]
        ).sum()
        identity_sq = torch.stack(
            [identity.float().square().sum() for identity in identity_directions]
        ).sum()
        self.last_direction_identity_stats = (
            dot.detach(),
            direction_sq.detach(),
            identity_sq.detach(),
        )

    def _record_direction_identity_stats(self, directions, identity_directions):
        self._record_direction_stats(directions, identity_directions)

    def _step_directions(
        self, group, params, gradients, *, record_direction_stats: bool = True
    ) -> list[torch.Tensor]:
        directions = self._direction(group, params, gradients)
        if group["constrained_update"]:
            directions = self._constrain(group, self._last_step_gradients, directions)
        else:
            self.last_metric_denominators = []
            self.last_metric_normalization_valid = []
        target = group["update_rms"]
        if target is not None:
            scaled, raw_rms, scaled_rms = ([], [], [])
            for direction in directions:
                rms = _rms(direction)
                scale = rms.new_tensor(target) / rms.clamp_min(group["update_rms_eps"])
                scaled.append(direction * scale.to(direction.dtype))
                raw_rms.append(rms.detach())
                scaled_rms.append((rms * scale).detach())
            directions = scaled
            self.last_raw_update_rms = raw_rms
            self.last_scaled_update_rms = scaled_rms
        else:
            self.last_raw_update_rms = []
            self.last_scaled_update_rms = []
        global_target = group["global_update_rms"]
        if global_target is not None:
            accumulation_dtype = (
                torch.float64
                if any((direction.dtype == torch.float64 for direction in directions))
                else torch.float32
            )
            direction_norms = torch.stack(
                [_scaled_l2_norm(direction.to(accumulation_dtype)) for direction in directions]
            )
            global_norm = _scaled_l2_norm(direction_norms)
            total_numel = sum((direction.numel() for direction in directions))
            raw_global_rms = global_norm / math.sqrt(total_numel)
            scale = raw_global_rms.new_tensor(global_target) / raw_global_rms.clamp_min(
                group["update_rms_eps"]
            )
            directions = [direction * scale.to(direction.dtype) for direction in directions]
            self.last_global_raw_update_rms = raw_global_rms.detach()
            self.last_global_scaled_update_rms = (raw_global_rms * scale).detach()
        else:
            self.last_global_raw_update_rms = None
            self.last_global_scaled_update_rms = None
        if record_direction_stats:
            self._record_direction_stats(
                directions, self._identity_directions(group, self._last_step_gradients)
            )
        else:
            self.last_direction_identity_stats = None
        return directions

    def _feed_pair(self, group, params, s_values, y_values, y_replica_values=None):
        checks: dict[torch.device, list[torch.Tensor]] = {}
        replica_values = [] if y_replica_values is None else y_replica_values
        for value in (*s_values, *y_values, *replica_values):
            checks.setdefault(value.device, []).append(torch.isfinite(value).all())
        if any((not bool(torch.stack(values).all()) for values in checks.values())):
            self.skipped += 1
            return None
        state = self.state[group["params"][0]]
        state["_pairs_seen"] = state.get("_pairs_seen", 0) + 1
        beta = group["beta_sy"]
        if beta:
            correction = 1.0 - beta ** state["_pairs_seen"]
            averaged_s, averaged_y, averaged_replica = ([], [], [])
            replica_source = y_values if y_replica_values is None else y_replica_values
            for parameter, s_value, y_value, y_replica in zip(
                params, s_values, y_values, replica_source, strict=True
            ):
                item = self.state[parameter]
                item.setdefault("ema_s", torch.zeros_like(s_value))
                item.setdefault("ema_y", torch.zeros_like(y_value))
                item["ema_s"].mul_(beta).add_(s_value, alpha=1.0 - beta)
                item["ema_y"].mul_(beta).add_(y_value, alpha=1.0 - beta)
                averaged_s.append(item["ema_s"] / correction)
                averaged_y.append(item["ema_y"] / correction)
                if y_replica_values is not None:
                    item.setdefault("ema_y_replica", torch.zeros_like(y_replica))
                    item["ema_y_replica"].mul_(beta).add_(y_replica, alpha=1.0 - beta)
                    averaged_replica.append(item["ema_y_replica"] / correction)
            s_values, y_values = (averaged_s, averaged_y)
            if y_replica_values is not None:
                y_replica_values = averaged_replica
        if state["_pairs_seen"] % group["T"]:
            return None
        self._update_curvature(group, params, s_values, y_values, y_replica_values)
        return (s_values, y_values)

    @torch.no_grad()
    def update_curvature(
        self,
        s: torch.Tensor | Sequence[torch.Tensor],
        y: torch.Tensor | Sequence[torch.Tensor],
        *,
        y_replica: torch.Tensor | Sequence[torch.Tensor] | None = None,
        group_index: int = 0,
    ):
        """Apply a raw secant pair in the parameter order of the selected group."""
        group = self.param_groups[group_index]
        params = list(group["params"])
        s_values = [s] if isinstance(s, torch.Tensor) else list(s)
        y_values = [y] if isinstance(y, torch.Tensor) else list(y)
        y_replica_values = (
            None
            if y_replica is None
            else [y_replica]
            if isinstance(y_replica, torch.Tensor)
            else list(y_replica)
        )
        if len(params) != len(s_values) or len(params) != len(y_values):
            raise ValueError("one s and y tensor is required per parameter")
        if y_replica_values is not None and len(params) != len(y_replica_values):
            raise ValueError("one replicated y tensor is required per parameter")
        return self._feed_pair(group, params, s_values, y_values, y_replica_values)

    @torch.no_grad()
    def parameter_step(self) -> None:
        """Apply ``-η H g`` from current ``.grad`` without forming a pair.

        Use this with :meth:`update_curvature` when secants span several
        parameter steps or when endpoint gradients are managed externally.
        Unlike :meth:`step`, this method does not save deferred-pair snapshots.
        """
        for group in self.param_groups:
            params, gradients = self._group_params(group)
            if not params:
                continue
            directions = self._step_directions(group, params, gradients)
            for parameter, direction in zip(params, directions, strict=True):
                parameter.add_(direction, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        if closure is not None:
            return self._step_same_batch(closure)
        for group in self.param_groups:
            params, gradients = self._group_params(group)
            if not params:
                continue
            old_params = [parameter.detach().clone() for parameter in params]
            old_gradients = [gradient.detach().clone() for gradient in gradients]
            if all(("prev_param" in self.state[parameter] for parameter in params)):
                self._feed_pair(
                    group,
                    params,
                    [parameter - self.state[parameter]["prev_param"] for parameter in params],
                    [
                        gradient - self.state[parameter]["prev_grad"]
                        for parameter, gradient in zip(params, gradients, strict=True)
                    ],
                )
            directions = self._step_directions(group, params, gradients)
            for parameter, direction in zip(params, directions, strict=True):
                parameter.add_(direction, alpha=-group["lr"])
            for parameter, old_parameter, old_gradient in zip(
                params, old_params, old_gradients, strict=True
            ):
                self.state[parameter]["prev_param"] = old_parameter
                self.state[parameter]["prev_grad"] = old_gradient
        return None

    @torch.no_grad()
    def _step_same_batch(self, closure: Callable[[], torch.Tensor]):
        with torch.enable_grad():
            closure()
        cached = []
        for group in self.param_groups:
            params, gradients = self._group_params(group)
            if not params:
                continue
            gradients = [gradient.detach().clone() for gradient in gradients]
            directions = self._step_directions(group, params, gradients)
            for parameter, direction in zip(params, directions, strict=True):
                parameter.add_(direction, alpha=-group["lr"])
            cached.append((group, params, gradients, directions))
        with torch.enable_grad():
            loss = closure()
        for group, params, gradients, directions in cached:
            self._feed_pair(
                group,
                params,
                [-group["lr"] * direction for direction in directions],
                [parameter.grad - old for parameter, old in zip(params, gradients, strict=True)],
            )
        return loss


class SoftServeDiag(_SecantOptimizer):
    """Diagonal SoftSERVE (one positive inverse-metric value per parameter)."""

    def __init__(
        self,
        params,
        lr: float,
        lam: float = 99.0,
        *,
        beta1: float = 0.0,
        beta_sy: float = 0.0,
        beta_h: float = 0.0,
        T: int = 1,
        nesterov: bool = False,
        constrained_update: bool = True,
        metric_rms_constraint: bool = False,
        update_rms: float | None = None,
        global_update_rms: float | None = None,
        pair_diagnostics: bool = False,
        lambda_schedule: str = "fixed",
        initial_h: float = 1.0,
        step_damping: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                lam=lam,
                beta1=beta1,
                beta_sy=beta_sy,
                beta_h=beta_h,
                T=T,
                nesterov=nesterov,
                constrained_update=constrained_update,
                metric_rms_constraint=metric_rms_constraint,
                update_rms=update_rms,
                global_update_rms=global_update_rms,
                pair_diagnostics=pair_diagnostics,
                lambda_schedule=lambda_schedule,
                initial_h=initial_h,
                step_damping=step_damping,
            ),
        )

    def _h(self, parameter: torch.Tensor, initial_h: float | None = None) -> torch.Tensor:
        state = self.state[parameter]
        if "h" not in state:
            value = 1.0 if initial_h is None else initial_h
            state["h"] = torch.full_like(parameter, value)
            state["zero_h_bootstrap_pending"] = value == 0.0
        return state["h"]

    def _direction(self, group, params, gradients):
        gradients = self._momentum_gradients(group, params, gradients)
        directions = []
        for parameter, gradient in zip(params, gradients, strict=True):
            h = self._h(parameter, group["initial_h"])
            bootstrap = self.state[parameter].get("zero_h_bootstrap_pending", False)
            damping = group["step_damping"]
            if bootstrap:
                direction = gradient
            else:
                metric = h if damping == 0 else h / (1.0 + damping * h)
                direction = metric * gradient
            directions.append(direction)
        return directions

    def _update_curvature_fixed(self, group, params, s_values, y_values):
        self.last_pair_q, self.last_pair_chi = ([], [])
        self.last_pair_gate_rejected, self.last_effective_lambda = ([], [])
        for parameter, s_value, y_value in zip(params, s_values, y_values, strict=True):
            h = self._h(parameter, group["initial_h"])
            if group["pair_diagnostics"]:
                q, chi = _diag_pair_metrics(h, s_value, y_value)
                self.last_pair_q.append(q.detach())
                self.last_pair_chi.append(chi.detach())
                self.last_pair_gate_rejected.append(
                    torch.zeros((), dtype=torch.bool, device=s_value.device)
                )
                self.last_effective_lambda.append(q.new_tensor(group["lam"]))
            new_h = _diag_update(h, s_value, y_value, group["lam"])
            self.state[parameter]["h"] = _blend(h, new_h, group["beta_h"])
            self.state[parameter]["zero_h_bootstrap_pending"] = False

    def _update_curvature(self, group, params, s_values, y_values, y_replica_values=None):
        self._update_curvature_fixed(group, params, s_values, y_values)


@dataclass
class _KronBucket:
    params: tuple[torch.Tensor, ...]
    A: torch.Tensor
    G: torch.Tensor
    S: torch.Tensor
    Y: torch.Tensor


class SoftServeKron(_SecantOptimizer):
    """Layerwise Kronecker SoftSERVE.

    A matrix parameter has shape ``(rows, columns)`` and direction
    ``G @ gradient @ A``.  Non-matrix parameters receive the diagonal update,
    although routing biases to Adam is usually preferable for neural networks.
    """

    def __init__(
        self,
        params,
        lr: float,
        lam: float = 99.0,
        *,
        beta1: float = 0.0,
        beta_sy: float = 0.0,
        beta_h: float = 0.0,
        T: int = 1,
        nesterov: bool = False,
        backend: str = "gemm",
        root_steps: int = 18,
        inverse_steps: int = 10,
        normalize: bool = True,
        gauge: str = "balanced_trace",
        bucket_chunk_size: int | None = 1,
        constrained_update: bool = True,
        metric_rms_constraint: bool = False,
        update_rms: float | None = None,
        global_update_rms: float | None = None,
        pair_diagnostics: bool = False,
        lambda_schedule: str = "fixed",
        exact_numerical_eps: float | None = 1e-10,
    ):
        if backend not in {"exact", "gemm"}:
            raise ValueError("backend must be 'exact' or 'gemm'")
        if gauge not in {"trace_a", "balanced_trace"}:
            raise ValueError("invalid Kronecker gauge")
        if bucket_chunk_size is not None and bucket_chunk_size < 1:
            raise ValueError("bucket_chunk_size must be positive or None")
        super().__init__(
            params,
            dict(
                lr=lr,
                lam=lam,
                beta1=beta1,
                beta_sy=beta_sy,
                beta_h=beta_h,
                T=T,
                nesterov=nesterov,
                backend=backend,
                root_steps=root_steps,
                inverse_steps=inverse_steps,
                normalize=normalize,
                gauge=gauge,
                bucket_chunk_size=bucket_chunk_size,
                constrained_update=constrained_update,
                metric_rms_constraint=metric_rms_constraint,
                update_rms=update_rms,
                global_update_rms=global_update_rms,
                pair_diagnostics=pair_diagnostics,
                lambda_schedule=lambda_schedule,
                exact_numerical_eps=exact_numerical_eps,
            ),
        )
        self.last_qme_residual_A: list[torch.Tensor] = []
        self.last_qme_residual_G: list[torch.Tensor] = []
        self.last_ns_root_residual: list[torch.Tensor] = []
        self.last_ns_inverse_root_residual: list[torch.Tensor] = []
        self.last_ns_root_pair_residual: list[torch.Tensor] = []
        self.last_ns_inverse_residual: list[torch.Tensor] = []
        self._rebuild_buckets()

    def _rebuild_buckets(self) -> None:
        self._buckets_by_group: dict[int, tuple[_KronBucket, ...]] = {}
        self._bucket_by_param: dict[int, tuple[_KronBucket, int]] = {}
        for group in self.param_groups:
            grouped: dict[tuple, list[torch.Tensor]] = {}
            for parameter in group["params"]:
                if parameter.ndim == 2:
                    key = (tuple(parameter.shape), parameter.dtype, parameter.device)
                    grouped.setdefault(key, []).append(parameter)
            buckets = []
            for params in grouped.values():
                n, m = params[0].shape
                if all(("A" in self.state[p] and "G" in self.state[p] for p in params)):
                    A = torch.stack([self.state[p]["A"] for p in params])
                    G = torch.stack([self.state[p]["G"] for p in params])
                else:
                    A = _eye(m, params[0]).expand(len(params), m, m).clone()
                    G = _eye(n, params[0]).expand(len(params), n, n).clone()
                curvature_frozen = bool(group.get("_metric_frozen", False))
                bucket = _KronBucket(
                    tuple(params),
                    A,
                    G,
                    torch.empty(0, dtype=params[0].dtype, device=params[0].device)
                    if curvature_frozen
                    else torch.empty(
                        len(params), n, m, dtype=params[0].dtype, device=params[0].device
                    ),
                    torch.empty(0, dtype=params[0].dtype, device=params[0].device)
                    if curvature_frozen
                    else torch.empty(
                        len(params), n, m, dtype=params[0].dtype, device=params[0].device
                    ),
                )
                for index, parameter in enumerate(params):
                    self.state[parameter]["A"] = A[index]
                    self.state[parameter]["G"] = G[index]
                    self._bucket_by_param[id(parameter)] = (bucket, index)
                buckets.append(bucket)
            self._buckets_by_group[id(group)] = tuple(buckets)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._rebuild_buckets()
        return result

    def _factors(self, parameter: torch.Tensor):
        state = self.state[parameter]
        if parameter.ndim == 2:
            bucket, index = self._bucket_by_param[id(parameter)]
            return (bucket.A[index], bucket.G[index])
        state.setdefault("h", torch.ones_like(parameter))
        return None

    def _record_gemm_diagnostics(
        self, diagnostics: dict[str, torch.Tensor], indices: Sequence[int] | None = None
    ) -> None:
        residual_a = diagnostics["A"].reshape(-1)
        residual_g = diagnostics["G"].reshape(-1)
        if indices is None:
            self.last_qme_residual_A.extend(residual_a.unbind())
            self.last_qme_residual_G.extend(residual_g.unbind())
        else:
            if len(indices) != residual_a.numel() or len(indices) != residual_g.numel():
                raise RuntimeError("QME residual index count differs")
            for local_index, parameter_index in enumerate(indices):
                self.last_qme_residual_A[parameter_index] = residual_a[local_index]
                self.last_qme_residual_G[parameter_index] = residual_g[local_index]
        self.last_ns_root_residual.extend(diagnostics["ns_root"].reshape(-1).unbind())
        self.last_ns_inverse_root_residual.extend(
            diagnostics["ns_inverse_root"].reshape(-1).unbind()
        )
        self.last_ns_root_pair_residual.extend(diagnostics["ns_root_pair"].reshape(-1).unbind())
        self.last_ns_inverse_residual.extend(diagnostics["ns_inverse"].reshape(-1).unbind())

    def _direction(self, group, params, gradients):
        gradients = self._momentum_gradients(group, params, gradients)
        directions = []
        for parameter, gradient in zip(params, gradients, strict=True):
            factors = self._factors(parameter)
            if factors is None:
                directions.append(self.state[parameter]["h"] * gradient)
            else:
                A, G = factors
                directions.append(G @ gradient @ A)
        return directions

    def _update_curvature_fixed(self, group, params, s_values, y_values):
        self.last_pair_q, self.last_pair_chi = ([], [])
        self.last_pair_gate_rejected, self.last_effective_lambda = ([], [])
        self.last_qme_residual_A, self.last_qme_residual_G = ([], [])
        self.last_ns_root_residual, self.last_ns_inverse_root_residual = ([], [])
        self.last_ns_root_pair_residual, self.last_ns_inverse_residual = ([], [])
        lam, beta_h = (group["lam"], group["beta_h"])
        if lam == 0:
            return
        matrix_items = []
        for parameter, s_value, y_value in zip(params, s_values, y_values, strict=True):
            factors = self._factors(parameter)
            if factors is None:
                h = self.state[parameter]["h"]
                if group["pair_diagnostics"]:
                    q, chi = _diag_pair_metrics(h, s_value, y_value)
                    self.last_pair_q.append(q.detach())
                    self.last_pair_chi.append(chi.detach())
                    self.last_pair_gate_rejected.append(
                        torch.zeros((), dtype=torch.bool, device=s_value.device)
                    )
                    self.last_effective_lambda.append(q.new_tensor(lam))
                self.state[parameter]["h"] = _blend(
                    h, _diag_update(h, s_value, y_value, lam), beta_h
                )
                continue
            if group["pair_diagnostics"]:
                q, chi = _kron_pair_metrics(*factors, s_value, y_value, group["inverse_steps"])
                self.last_pair_q.append(q.detach())
                self.last_pair_chi.append(chi.detach())
                self.last_pair_gate_rejected.append(
                    torch.zeros((), dtype=torch.bool, device=s_value.device)
                )
                self.last_effective_lambda.append(q.new_tensor(lam))
            matrix_items.append((parameter, s_value, y_value, *factors))
        if group["backend"] == "exact":
            for _, s_value, y_value, A, G in matrix_items:
                diagnostics: dict[str, torch.Tensor] = {}
                A_next, G_next = kron_sweep(
                    A,
                    G,
                    s_value,
                    y_value,
                    lam,
                    normalize=group["normalize"],
                    gauge=group["gauge"],
                    eps=group["exact_numerical_eps"],
                    diagnostics=diagnostics,
                )
                self.last_qme_residual_A.append(diagnostics["A"])
                self.last_qme_residual_G.append(diagnostics["G"])
                A.copy_(_blend(A, A_next, beta_h))
                G.copy_(_blend(G, G_next, beta_h))
            return
        by_parameter = {id(item[0]): item for item in matrix_items}
        handled: set[int] = set()
        for bucket in self._buckets_by_group[id(group)]:
            items = [
                by_parameter[id(parameter)]
                for parameter in bucket.params
                if id(parameter) in by_parameter
            ]
            if not items or len(items) != len(bucket.params):
                continue
            handled.update((id(item[0]) for item in items))
            torch.stack([item[1] for item in items], out=bucket.S)
            torch.stack([item[2] for item in items], out=bucket.Y)
            chunk_size = group["bucket_chunk_size"] or len(items)
            for start in range(0, len(items), chunk_size):
                stop = min(start + chunk_size, len(items))
                diagnostics = {}
                A_next, G_next = gemm_kron_sweep(
                    bucket.A[start:stop],
                    bucket.G[start:stop],
                    bucket.S[start:stop],
                    bucket.Y[start:stop],
                    lam,
                    group["root_steps"],
                    group["inverse_steps"],
                    normalize=group["normalize"],
                    gauge=group["gauge"],
                    diagnostics=diagnostics,
                )
                self._record_gemm_diagnostics(diagnostics)
                bucket.A[start:stop].copy_(_blend(bucket.A[start:stop], A_next, beta_h))
                bucket.G[start:stop].copy_(_blend(bucket.G[start:stop], G_next, beta_h))
        remaining = [item for item in matrix_items if id(item[0]) not in handled]
        grouped: dict[tuple, list[tuple]] = {}
        for item in remaining:
            parameter = item[0]
            grouped.setdefault(
                (tuple(parameter.shape), parameter.dtype, parameter.device), []
            ).append(item)
        for items in grouped.values():
            chunk_size = group["bucket_chunk_size"] or len(items)
            for start in range(0, len(items), chunk_size):
                chunk = items[start : start + chunk_size]
                diagnostics = {}
                A_next, G_next = gemm_kron_sweep(
                    torch.stack([item[3] for item in chunk]),
                    torch.stack([item[4] for item in chunk]),
                    torch.stack([item[1] for item in chunk]),
                    torch.stack([item[2] for item in chunk]),
                    lam,
                    group["root_steps"],
                    group["inverse_steps"],
                    normalize=group["normalize"],
                    gauge=group["gauge"],
                    diagnostics=diagnostics,
                )
                self._record_gemm_diagnostics(diagnostics)
                for index, (_, _, _, A, G) in enumerate(chunk):
                    A.copy_(_blend(A, A_next[index], beta_h))
                    G.copy_(_blend(G, G_next[index], beta_h))

    def _update_curvature(self, group, params, s_values, y_values, y_replica_values=None):
        self._update_curvature_fixed(group, params, s_values, y_values)


symmetrize = _sym
root_pair_ns = _root_pair_ns
inverse_ns = _inverse_ns
tensor_rms = _rms
