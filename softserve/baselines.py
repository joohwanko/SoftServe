"""Small baseline adapters used by the routed neural-network experiments."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Literal
import torch
from torch import Tensor, nn


class SGDM(torch.optim.Optimizer):
    """The paper's matched identity-metric control.

    This is deliberately not ordinary PyTorch SGD: it uses SoftSERVE's bias-corrected
    gradient EMA and the requested tensor-direction scaling. With the default
    ``normalization="unit"`` it removes only learned curvature from the structured
    SoftSERVE route.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        lr: float,
        beta: float,
        normalization: Literal["unit", "rms", "none"] = "unit",
    ) -> None:
        if normalization not in {"unit", "rms", "none"}:
            raise ValueError(f"unknown direction normalization: {normalization}")
        super().__init__(parameters, {"lr": lr, "beta": beta, "normalization": normalization})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                beta = group["beta"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    step = state.get("step", 0) + 1
                    momentum = state.setdefault("momentum", torch.zeros_like(parameter.grad))
                    momentum.mul_(beta).add_(parameter.grad, alpha=1.0 - beta)
                    direction = momentum / (1.0 - beta**step) if beta else parameter.grad
                    normalization = group.get("normalization", "unit")
                    if normalization == "none":
                        update = direction
                    else:
                        quadratic = (direction.float() * direction.float()).sum()
                        scale = torch.where(
                            quadratic > 0, torch.rsqrt(quadratic), torch.zeros_like(quadratic)
                        )
                        if normalization == "rms":
                            scale = scale * direction.numel() ** 0.5
                        update = direction * scale.to(direction)
                    parameter.add_(update, alpha=-group["lr"])
                    state["step"] = step
        return loss


def make_soap(
    parameters: Iterable[nn.Parameter],
    *,
    lr: float,
    betas: tuple[float, float],
    precondition_frequency: int,
    schedule_free_beta: float | None = None,
):
    """Build the exact Meta DistributedShampoo SOAP variant used in the paper."""
    try:
        from distributed_shampoo import (
            DefaultSOAPConfig,
            DistributedShampoo,
            ScheduleFreeConfig,
            WeightDecayType,
        )
    except ImportError as error:
        raise ImportError("install the 'soap' experiment dependency") from error
    return DistributedShampoo(
        parameters,
        lr=lr,
        betas=betas,
        epsilon=1e-12,
        weight_decay=0.0,
        weight_decay_type=WeightDecayType.DECOUPLED,
        max_preconditioner_dim=8192,
        precondition_frequency=precondition_frequency,
        start_preconditioning_step=precondition_frequency,
        preconditioner_config=DefaultSOAPConfig,
        iterate_averaging_config=None
        if schedule_free_beta is None
        else ScheduleFreeConfig(train_interp_coeff=schedule_free_beta),
    )


class RoutedOptimizer:
    """Present a matrix optimizer and Adam fallback as one small controller."""

    def __init__(
        self,
        main: torch.optim.Optimizer,
        fallback_parameters: Iterable[nn.Parameter],
        *,
        fallback_lr: float,
        beta: float,
    ) -> None:
        self.main = main
        self.fallback = torch.optim.Adam(
            fallback_parameters, lr=fallback_lr, betas=(beta, 0.999), eps=1e-08, weight_decay=0.0
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.main.zero_grad(set_to_none=set_to_none)
        self.fallback.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.main.step()
        self.fallback.step()

    def set_lr(self, main: float, fallback: float) -> None:
        for group in self.main.param_groups:
            group["lr"] = main
        for group in self.fallback.param_groups:
            group["lr"] = fallback

    def state_dict(self) -> dict:
        return {"main": self.main.state_dict(), "fallback": self.fallback.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.main.load_state_dict(state["main"])
        self.fallback.load_state_dict(state["fallback"])


def global_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    values = [
        parameter.grad.float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)
