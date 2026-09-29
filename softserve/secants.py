"""Explicit interval secants for deterministic and same-batch stochastic loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class IntervalSecant:
    """Snapshot a raw gradient and parameter point every ``interval`` updates."""

    interval: int
    parameters: tuple[torch.Tensor, ...] | None = None
    gradients: tuple[torch.Tensor, ...] | None = None
    batch_key: int | None = None

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("interval must be positive")

    def starts_at(self, step: int) -> bool:
        return step % self.interval == 0

    def ends_at(self, step: int) -> bool:
        return (step + 1) % self.interval == 0

    @torch.no_grad()
    def start(
        self,
        parameters: Sequence[torch.Tensor],
        gradients: Sequence[torch.Tensor],
        *,
        batch_key: int,
    ) -> None:
        self.parameters = tuple(value.detach().clone() for value in parameters)
        self.gradients = tuple(value.detach().clone() for value in gradients)
        self.batch_key = batch_key

    @torch.no_grad()
    def pair(
        self,
        parameters: Sequence[torch.Tensor],
        endpoint_gradients: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.parameters is None or self.gradients is None:
            raise RuntimeError("start() must be called before pair()")
        if len(parameters) != len(self.parameters) or len(endpoint_gradients) != len(
            self.gradients
        ):
            raise ValueError("secant parameter count changed")
        s = [
            value.detach() - origin
            for value, origin in zip(parameters, self.parameters, strict=True)
        ]
        y = [
            value.detach() - origin
            for value, origin in zip(endpoint_gradients, self.gradients, strict=True)
        ]
        return s, y
