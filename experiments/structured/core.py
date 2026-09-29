"""Matrix adapters for the already parity-tested MNIST baseline mechanics.

Inputs are already homogeneous where the model has a bias. For shared weights,
activation moments average over applications. K-FAC uses the expand approximation:
A = mean(a a^T), G = sum(delta delta^T), where delta is obtained from a loss-
curvature probe scaled by sqrt(1/N). K-BFGS(L) uses mean preactivation / summed
mean-loss-adjoint pairs. This latter shared-weight extension is explicitly labelled
as such; it is not a claim of reproducing a published recurrent K-BFGS algorithm.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from softserve.baselines_kron_qn import _apply_inverse_bfgs, _inverse_bfgs_update


@dataclass
class Statistics:
    activation: torch.Tensor
    input_mean: torch.Tensor
    preactivation_mean: torch.Tensor
    adjoint_sum: torch.Tensor
    fisher: torch.Tensor | None = None


class MatrixBaseline:
    def __init__(
        self,
        matrices,
        *,
        method,
        lr,
        damping,
        momentum=0.9,
        factor_decay=0.9,
        inverse_frequency=20,
        history_size=100,
    ):
        if method not in {"kfac", "kbfgs_l"}:
            raise ValueError(method)
        if not (lr > 0 and damping > 0 and (0 <= momentum < 1) and (0 <= factor_decay < 1)):
            raise ValueError("invalid optimizer constants")
        self.matrices = tuple(matrices)
        self.method, self.lr, self.damping = (method, float(lr), float(damping))
        self.momentum, self.factor_decay = (momentum, factor_decay)
        self.inverse_frequency, self.history_size = (inverse_frequency, history_size)
        self.steps, self.rejected_pairs = (0, 0)
        self.state = []
        for matrix in matrices:
            self.state.append(
                {
                    "M": torch.zeros_like(matrix),
                    "A": None,
                    "G": None,
                    "Ainv": None,
                    "Ginv": None,
                    "s": [],
                    "y": [],
                    "pair_s": torch.zeros_like(matrix[:, 0]),
                    "pair_y": torch.zeros_like(matrix[:, 0]),
                }
            )

    @torch.no_grad()
    def apply(self, gradients, statistics):
        root = math.sqrt(self.damping)
        for matrix, gradient, stat, state in zip(
            self.matrices, gradients, statistics, self.state, strict=True
        ):
            if not torch.isfinite(gradient).all() or not torch.isfinite(stat.activation).all():
                raise FloatingPointError("nonfinite gradient/activation factor")
            if state["A"] is None:
                state["A"] = stat.activation.clone()
            else:
                state["A"].lerp_(stat.activation, 1 - self.factor_decay)
            state["M"].mul_(self.momentum).add_(gradient, alpha=1 - self.momentum)
            identity = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
            state["Alm"] = state["A"] + root * identity
            state["input_mean"] = stat.input_mean
            if self.method == "kfac":
                if stat.fisher is None or not torch.isfinite(stat.fisher).all():
                    raise FloatingPointError("missing/nonfinite Fisher factor")
                if state["G"] is None:
                    state["G"] = stat.fisher.clone()
                else:
                    state["G"].lerp_(stat.fisher, 1 - self.factor_decay)
                if self.steps % self.inverse_frequency == 0 or self.steps <= 20:
                    state["Ainv"] = torch.linalg.inv(state["Alm"])
                    state["Ginv"] = torch.linalg.inv(
                        state["G"]
                        + root
                        * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
                    )
                direction = state["Ginv"] @ state["M"] @ state["Ainv"]
            else:
                if state["Ainv"] is None:
                    state["Ainv"] = torch.linalg.inv(state["Alm"])
                direction = self.inverse_output(state, state["M"]) @ state["Ainv"]
            if not torch.isfinite(direction).all():
                raise FloatingPointError("nonfinite update direction")
            matrix.add_(direction, alpha=-self.lr)
        self.steps += 1

    @staticmethod
    def inverse_output(state, value):
        if value.shape[0] == 1 and state["s"]:
            return value * (state["s"][-1][0] / state["y"][-1][0])
        return _apply_inverse_bfgs(value, state["s"], state["y"])

    @torch.no_grad()
    def update_pairs(self, before, after, *, stochastic):
        if self.method != "kbfgs_l":
            raise RuntimeError("pairs belong to K-BFGS(L)")
        decay = 0.9 if stochastic else 0.0
        root = math.sqrt(self.damping)
        for old, new, state in zip(before, after, self.state, strict=True):
            state["pair_s"].mul_(decay).add_(
                old.preactivation_mean - new.preactivation_mean, alpha=1 - decay
            )
            state["pair_y"].mul_(decay).add_(old.adjoint_sum - new.adjoint_sum, alpha=1 - decay)
            s, y = (state["pair_s"].clone(), state["pair_y"].clone())
            hy = self.inverse_output(state, y)
            sy, yhy = (s.dot(y), y.dot(hy))
            if torch.isfinite(yhy) and yhy > 0 and (sy / yhy <= 0.2):
                theta = 0.8 * yhy / (yhy - sy)
                s = theta * s + (1 - theta) * hy
            ss, sy = (s.dot(s), s.dot(y))
            if torch.isfinite(ss) and ss > 0 and (sy / ss <= root):
                theta = (1 - root) * ss / (ss - sy)
                y = theta * y + (1 - theta) * s
            curvature = s.dot(y)
            threshold = 0.0001 * s.dot(s) * torch.linalg.vector_norm(old.adjoint_sum)
            if torch.isfinite(curvature) and curvature > 0 and (curvature > threshold):
                state["s"].append(s)
                state["y"].append(y)
                if len(state["s"]) > self.history_size:
                    state["s"].pop(0)
                    state["y"].pop(0)
            else:
                self.rejected_pairs += 1
            input_s = state["Ainv"] @ state["input_mean"]
            input_y = state["Alm"] @ input_s
            state["Ainv"], accepted = _inverse_bfgs_update(
                state["Ainv"], input_s, input_y, state["input_mean"]
            )
            self.rejected_pairs += int(not accepted)

    def state_bytes(self):

        def size(obj):
            if torch.is_tensor(obj):
                return obj.numel() * obj.element_size()
            if isinstance(obj, dict):
                return sum((size(v) for v in obj.values()))
            if isinstance(obj, (list, tuple)):
                return sum((size(v) for v in obj))
            return 0

        return size(self.state)
