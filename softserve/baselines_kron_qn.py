"""Kronecker-factored baselines ported from Goldfarb et al. (2020).

The reference implementation stores a bias inside each homogeneous layer matrix.
These optimizers preserve that convention so their layer updates can be checked
directly against the public implementation.
"""

from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
import torch
from torch import nn


def homogeneous(values: torch.Tensor) -> torch.Tensor:
    """Append the constant coordinate used by the upstream implementation."""
    return torch.cat((values, torch.ones_like(values[:, :1])), dim=1)


def homogeneous_gradients(layers: Sequence[nn.Linear]) -> list[torch.Tensor]:
    gradients = []
    for layer in layers:
        if layer.weight.grad is None or layer.bias.grad is None:
            raise RuntimeError("all affine parameters must have gradients")
        gradients.append(torch.cat((layer.weight.grad, layer.bias.grad[:, None]), dim=1))
    return gradients


def _apply_inverse_bfgs(
    matrix: torch.Tensor, s_values: Sequence[torch.Tensor], y_values: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Apply an L-BFGS inverse with the reference method's identity H0."""
    if not s_values:
        return matrix
    vector_input = matrix.ndim == 1
    result = matrix[:, None] if vector_input else matrix
    q = result
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    for s_value, y_value in zip(reversed(s_values), reversed(y_values), strict=True):
        rho = y_value.dot(s_value).reciprocal()
        alpha = rho * (s_value[:, None] * q).sum(dim=0)
        q = q - y_value[:, None] * alpha[None, :]
        alphas.append(alpha)
        rhos.append(rho)
    result = q
    for s_value, y_value, alpha, rho in zip(
        s_values, y_values, reversed(alphas), reversed(rhos), strict=True
    ):
        beta = rho * (y_value[:, None] * result).sum(dim=0)
        result = result + s_value[:, None] * (alpha - beta)[None, :]
    return result[:, 0] if vector_input else result


def _inverse_bfgs_update(
    matrix: torch.Tensor, s_value: torch.Tensor, y_value: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    """Reference inverse-BFGS update, including its pair rejection test."""
    curvature = s_value.dot(y_value)
    threshold = 0.0001 * s_value.dot(s_value) * torch.linalg.vector_norm(reference)
    if not torch.isfinite(curvature) or curvature <= 0.0 or curvature <= threshold:
        return (matrix, False)
    rho = curvature.reciprocal()
    matrix_y = matrix.mv(y_value)
    updated = (
        matrix
        + (rho.square() * y_value.dot(matrix_y) + rho) * torch.outer(s_value, s_value)
        - rho * (torch.outer(s_value, matrix_y) + torch.outer(matrix_y, s_value))
    )
    if not torch.isfinite(updated).all():
        return (matrix, False)
    return (updated, True)


class KFAC(torch.optim.Optimizer):
    """Empirical-Fisher K-FAC matching the NeurIPS 2020 reference mechanics."""

    def __init__(
        self,
        layers: Sequence[nn.Linear],
        *,
        lr: float,
        damping: float = 3.0,
        momentum: float = 0.9,
        factor_decay: float = 0.9,
        inverse_update_frequency: int = 20,
    ) -> None:
        if not layers:
            raise ValueError("KFAC requires at least one layer")
        if lr <= 0.0 or damping <= 0.0:
            raise ValueError("lr and damping must be positive")
        if not 0.0 <= momentum < 1.0 or not 0.0 <= factor_decay < 1.0:
            raise ValueError("momentum and factor_decay must lie in [0,1)")
        if inverse_update_frequency < 1:
            raise ValueError("inverse_update_frequency must be positive")
        parameters = [parameter for layer in layers for parameter in layer.parameters()]
        super().__init__(parameters, {"lr": lr})
        self.layers = tuple(layers)
        self.damping = float(damping)
        self.momentum = float(momentum)
        self.factor_decay = float(factor_decay)
        self.inverse_update_frequency = int(inverse_update_frequency)
        self.steps = 0
        self.last_directions: list[torch.Tensor] = []
        for layer in self.layers:
            state = self.state[layer.weight]
            state["A"] = torch.zeros(
                layer.in_features + 1,
                layer.in_features + 1,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            state["G"] = torch.zeros(
                layer.out_features,
                layer.out_features,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            state["momentum"] = torch.zeros(
                layer.out_features,
                layer.in_features + 1,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            state["A_inv"] = None
            state["G_inv"] = None

    @torch.no_grad()
    def set_initial_factors(
        self, activation_factors: Sequence[torch.Tensor], gradient_factors: Sequence[torch.Tensor]
    ) -> None:
        if len(activation_factors) != len(self.layers) or len(gradient_factors) != len(self.layers):
            raise ValueError("one factor pair is required per layer")
        for layer, activation, gradient in zip(
            self.layers, activation_factors, gradient_factors, strict=True
        ):
            self.state[layer.weight]["A"].copy_(activation)
            self.state[layer.weight]["G"].copy_(gradient)

    @torch.no_grad()
    def step(
        self,
        actual_gradients: Sequence[torch.Tensor],
        inputs: Sequence[torch.Tensor],
        output_gradients: Sequence[torch.Tensor],
    ) -> None:
        if not len(actual_gradients) == len(inputs) == len(output_gradients) == len(self.layers):
            raise ValueError("one gradient and statistic pair is required per layer")
        root_damping = math.sqrt(self.damping)
        update_inverse = self.steps % self.inverse_update_frequency == 0 or self.steps <= 20
        self.last_directions = []
        for layer, actual, layer_input, output_gradient in zip(
            self.layers, actual_gradients, inputs, output_gradients, strict=True
        ):
            state = self.state[layer.weight]
            augmented = homogeneous(layer_input)
            activation = augmented.T @ augmented / len(augmented)
            gradient = output_gradient.T @ output_gradient / len(output_gradient)
            state["A"].mul_(self.factor_decay).add_(activation, alpha=1.0 - self.factor_decay)
            state["G"].mul_(self.factor_decay).add_(gradient, alpha=1.0 - self.factor_decay)
            state["momentum"].mul_(self.momentum).add_(actual, alpha=1.0 - self.momentum)
            if update_inverse:
                identity_a = torch.eye(
                    len(state["A"]), device=state["A"].device, dtype=state["A"].dtype
                )
                identity_g = torch.eye(
                    len(state["G"]), device=state["G"].device, dtype=state["G"].dtype
                )
                state["A_inv"] = torch.linalg.inv(state["A"] + root_damping * identity_a)
                state["G_inv"] = torch.linalg.inv(state["G"] + root_damping * identity_g)
            if state["A_inv"] is None or state["G_inv"] is None:
                raise RuntimeError("KFAC inverse factors were not initialized")
            direction = state["G_inv"] @ state["momentum"] @ state["A_inv"]
            self.last_directions.append(direction.clone())
            learning_rate = float(self.param_groups[0]["lr"])
            layer.weight.add_(direction[:, :-1], alpha=-learning_rate)
            layer.bias.add_(direction[:, -1], alpha=-learning_rate)
        self.steps += 1


class KBFGSL(torch.optim.Optimizer):
    """Kronecker-factored limited-memory BFGS from Goldfarb et al. (2020)."""

    def __init__(
        self,
        layers: Sequence[nn.Linear],
        *,
        lr: float,
        damping: float = 0.3,
        momentum: float = 0.9,
        factor_decay: float = 0.9,
        history_size: int = 100,
    ) -> None:
        if not layers:
            raise ValueError("KBFGSL requires at least one layer")
        if lr <= 0.0 or damping <= 0.0:
            raise ValueError("lr and damping must be positive")
        if not 0.0 <= momentum < 1.0 or not 0.0 <= factor_decay < 1.0:
            raise ValueError("momentum and factor_decay must lie in [0,1)")
        if history_size < 1:
            raise ValueError("history_size must be positive")
        parameters = [parameter for layer in layers for parameter in layer.parameters()]
        super().__init__(parameters, {"lr": lr})
        self.layers = tuple(layers)
        self.damping = float(damping)
        self.momentum = float(momentum)
        self.factor_decay = float(factor_decay)
        self.history_size = int(history_size)
        self.steps = 0
        self.preinitialized = False
        self.last_directions: list[torch.Tensor] = []
        for layer in self.layers:
            state = self.state[layer.weight]
            width = layer.in_features + 1
            state["A"] = torch.zeros(
                width, width, device=layer.weight.device, dtype=layer.weight.dtype
            )
            state["H_input"] = None
            state["momentum"] = torch.zeros(
                layer.out_features, width, device=layer.weight.device, dtype=layer.weight.dtype
            )
            state["pair_momentum_s"] = torch.zeros(
                layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype
            )
            state["pair_momentum_y"] = torch.zeros(
                layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype
            )
            state["s_history"] = []
            state["y_history"] = []
            state["A_lm"] = None
            state["input_mean"] = None

    @torch.no_grad()
    def set_initial_activation_factors(self, activation_factors: Sequence[torch.Tensor]) -> None:
        if len(activation_factors) != len(self.layers):
            raise ValueError("one activation factor is required per layer")
        for layer, factor in zip(self.layers, activation_factors, strict=True):
            self.state[layer.weight]["A"].copy_(factor)
        self.preinitialized = True

    @torch.no_grad()
    def apply_step(
        self, actual_gradients: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]
    ) -> None:
        """Update factors, form the direction, and move to the look-ahead point."""
        if len(actual_gradients) != len(self.layers) or len(inputs) != len(self.layers):
            raise ValueError("one gradient and activation is required per layer")
        root_damping = math.sqrt(self.damping)
        self.last_directions = []
        for layer, actual, layer_input in zip(self.layers, actual_gradients, inputs, strict=True):
            state = self.state[layer.weight]
            augmented = homogeneous(layer_input)
            activation = augmented.T @ augmented / len(augmented)
            if not (self.preinitialized and self.steps == 0):
                state["A"].mul_(self.factor_decay).add_(activation, alpha=1.0 - self.factor_decay)
            identity = torch.eye(len(state["A"]), device=state["A"].device, dtype=state["A"].dtype)
            state["A_lm"] = state["A"] + root_damping * identity
            if state["H_input"] is None:
                state["H_input"] = torch.linalg.inv(state["A_lm"])
            state["momentum"].mul_(self.momentum).add_(actual, alpha=1.0 - self.momentum)
            output_preconditioned = _apply_inverse_bfgs(
                state["momentum"], state["s_history"], state["y_history"]
            )
            direction = output_preconditioned @ state["H_input"]
            self.last_directions.append(direction.clone())
            state["input_mean"] = augmented.mean(dim=0)
            learning_rate = float(self.param_groups[0]["lr"])
            layer.weight.add_(direction[:, :-1], alpha=-learning_rate)
            layer.bias.add_(direction[:, -1], alpha=-learning_rate)

    @torch.no_grad()
    def update_curvature(
        self,
        current_preactivations: Sequence[torch.Tensor],
        next_preactivations: Sequence[torch.Tensor],
        current_output_gradients: Sequence[torch.Tensor],
        next_output_gradients: Sequence[torch.Tensor],
        *,
        stochastic: bool,
    ) -> None:
        if (
            not len(current_preactivations)
            == len(next_preactivations)
            == len(current_output_gradients)
            == len(next_output_gradients)
            == len(self.layers)
        ):
            raise ValueError("one curvature statistic is required per layer")
        pair_decay = 0.9 if stochastic else 0.0
        root_damping = math.sqrt(self.damping)
        for layer, current_a, next_a, current_g, next_g in zip(
            self.layers,
            current_preactivations,
            next_preactivations,
            current_output_gradients,
            next_output_gradients,
            strict=True,
        ):
            state = self.state[layer.weight]
            s_value = current_a.mean(dim=0) - next_a.mean(dim=0)
            y_value = current_g.mean(dim=0) - next_g.mean(dim=0)
            state["pair_momentum_s"].mul_(pair_decay).add_(s_value, alpha=1.0 - pair_decay)
            state["pair_momentum_y"].mul_(pair_decay).add_(y_value, alpha=1.0 - pair_decay)
            s_value = state["pair_momentum_s"].clone()
            y_value = state["pair_momentum_y"].clone()
            inverse_y = _apply_inverse_bfgs(y_value, state["s_history"], state["y_history"])
            sy = s_value.dot(y_value)
            yhy = y_value.dot(inverse_y)
            if torch.isfinite(yhy) and yhy > 0.0 and (sy / yhy <= 0.2):
                theta = 0.8 * yhy / (yhy - sy)
                s_value = theta * s_value + (1.0 - theta) * inverse_y
            ss = s_value.dot(s_value)
            sy = s_value.dot(y_value)
            if torch.isfinite(ss) and ss > 0.0 and (sy / ss <= root_damping):
                theta = (1.0 - root_damping) * ss / (ss - sy)
                y_value = theta * y_value + (1.0 - theta) * s_value
            threshold = (
                0.0001 * s_value.dot(s_value) * torch.linalg.vector_norm(current_g.mean(dim=0))
            )
            curvature = y_value.dot(s_value)
            if torch.isfinite(curvature) and curvature > threshold:
                state["s_history"].append(s_value)
                state["y_history"].append(y_value)
                if len(state["s_history"]) > self.history_size:
                    del state["s_history"][0]
                    del state["y_history"][0]
            input_mean = state["input_mean"]
            input_inverse = state["H_input"]
            activation_lm = state["A_lm"]
            if input_mean is None or input_inverse is None or activation_lm is None:
                raise RuntimeError("input-factor state was not initialized")
            input_s = input_inverse.mv(input_mean)
            input_y = activation_lm.mv(input_s)
            state["H_input"], _ = _inverse_bfgs_update(input_inverse, input_s, input_y, input_mean)
        self.steps += 1

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()


__all__ = [
    "KFAC",
    "KBFGSL",
    "_apply_inverse_bfgs",
    "_inverse_bfgs_update",
    "homogeneous",
    "homogeneous_gradients",
]
