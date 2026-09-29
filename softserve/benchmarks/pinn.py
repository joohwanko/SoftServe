"""Wave, convection, and reaction PINNs from Rathore et al.

This module is a dependency-free transcription of the pinned benchmark code. It keeps
the original FP32 initialization order, collocation rules, unweighted MSE objective,
and analytic test solutions. Optimizer and cluster logic live elsewhere.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from torch import Tensor, nn

PDE = Literal["wave", "convection", "reaction"]
UPSTREAM_URL = "https://github.com/pratikrathore8/opt_for_pinns"
UPSTREAM_COMMIT = "61a90a7f2592ca819e945f48827ed623214cee1e"


@dataclass(frozen=True)
class PINNConfig:
    pde: PDE = "wave"
    width: int = 200
    layers: int = 4
    beta: float = 5.0
    rho: float = 5.0
    num_x: int = 257
    num_t: int = 101
    residual_batch: int = 10000

    @property
    def x_range(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.pde == "wave" else (0.0, 2.0 * np.pi)

    def validate(self) -> None:
        if self.pde not in {"wave", "convection", "reaction"}:
            raise ValueError(f"unknown PDE: {self.pde}")
        if self.layers < 2 or self.width < 1:
            raise ValueError("layers >= 2 and width >= 1 are required")
        if self.residual_batch < 1:
            raise ValueError("residual_batch must be positive")


class PINN(nn.Module):
    """The paper's 2 -> width -> ... -> width -> 1 tanh network."""

    def __init__(self, width: int = 200, layers: int = 4) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for layer in range(layers - 1):
            modules += [nn.Linear(2 if layer == 0 else width, width), nn.Tanh()]
        modules.append(nn.Linear(width, 1))
        self.linear = nn.Sequential(*modules)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.linear(torch.cat((x, t), dim=-1))


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


def _leaf(array: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype).detach().requires_grad_(True)


@dataclass
class Collocation:
    residual_x: Tensor
    residual_t: Tensor
    initial_x: Tensor
    initial_t: Tensor
    upper_x: Tensor
    upper_t: Tensor
    lower_x: Tensor
    lower_t: Tensor


def boundary_points(config: PINNConfig, *, device: torch.device, dtype: torch.dtype) -> Collocation:
    x = np.linspace(*config.x_range, config.num_x).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, config.num_t).reshape(-1, 1)
    empty = np.empty((0, 1))
    return Collocation(
        _leaf(empty, device=device, dtype=dtype),
        _leaf(empty, device=device, dtype=dtype),
        _leaf(x, device=device, dtype=dtype),
        _leaf(np.zeros_like(x), device=device, dtype=dtype),
        _leaf(np.full_like(t, config.x_range[1]), device=device, dtype=dtype),
        _leaf(t, device=device, dtype=dtype),
        _leaf(np.full_like(t, config.x_range[0]), device=device, dtype=dtype),
        _leaf(t, device=device, dtype=dtype),
    )


class ResidualSampler:
    """Stateless collocation sampler with exact endpoint replay.

    ``continuous=True`` is the paper's stochastic protocol: every accepted update draws
    fresh uniform coordinates. ``continuous=False`` reproduces the deterministic fixed
    10,000-point subset of the 25,500-point interior grid.
    """

    def __init__(
        self,
        config: PINNConfig,
        *,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        continuous: bool,
        namespace: int = 80000,
    ) -> None:
        config.validate()
        self.config = config
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.continuous = continuous
        self.namespace = namespace
        self.boundary = boundary_points(config, device=device, dtype=dtype)
        if not continuous:
            x = np.linspace(*config.x_range, config.num_x)[1:-1]
            t = np.linspace(0.0, 1.0, config.num_t)[1:]
            x_mesh, t_mesh = np.meshgrid(x, t)
            self.grid = np.column_stack((x_mesh.ravel(), t_mesh.ravel()))
            if config.residual_batch > len(self.grid):
                raise ValueError("fixed residual batch exceeds the interior grid")
            rng = np.random.RandomState(seed)
            self.fixed = self.grid[rng.choice(len(self.grid), config.residual_batch, replace=False)]

    def _values(self, step: int) -> np.ndarray:
        if not self.continuous:
            return self.fixed
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([self.namespace, self.seed, step]))
        )
        standard = rng.random((self.config.residual_batch, 2))
        standard[:, 0] = (
            self.config.x_range[0]
            + (self.config.x_range[1] - self.config.x_range[0]) * standard[:, 0]
        )
        return standard

    def batch(self, step: int) -> Collocation:
        points = self._values(step)
        base = self.boundary
        return Collocation(
            _leaf(points[:, :1], device=self.device, dtype=self.dtype),
            _leaf(points[:, 1:], device=self.device, dtype=self.dtype),
            base.initial_x,
            base.initial_t,
            base.upper_x,
            base.upper_t,
            base.lower_x,
            base.lower_t,
        )


def build_pinn(
    config: PINNConfig, *, seed: int, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> PINN:
    """Initialize exactly as the upstream FP32 model, then optionally promote."""
    config.validate()
    _seed_everything(seed)
    model = PINN(config.width, config.layers).to(device=device, dtype=torch.float32)
    model.apply(_initialize)
    return model.to(dtype=dtype)


def deterministic_sampler(
    config: PINNConfig, *, seed: int, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> ResidualSampler:
    """Create the upstream fixed grid subset after consuming model-init RNG.

    Call this immediately after :func:`build_pinn` to preserve exact RNG order.
    """
    sampler = ResidualSampler(
        config, seed=seed, device=torch.device(device), dtype=dtype, continuous=False
    )
    indices = np.random.choice(len(sampler.grid), config.residual_batch, replace=False)
    sampler.fixed = sampler.grid[indices]
    return sampler


def loss_components(model: PINN, points: Collocation, config: PINNConfig) -> dict[str, Tensor]:
    """Return the exact unweighted residual, boundary, and initial MSE terms."""
    x, t = (points.residual_x, points.residual_t)
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    if config.pde == "wave":
        u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True, retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, ones, create_graph=True, retain_graph=True)[0]
        residual = (u_tt - 4.0 * u_xx).square().mean()
    elif config.pde == "convection":
        u_x = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
        residual = (u_t + config.beta * u_x).square().mean()
    else:
        residual = (u_t - config.rho * u * (1.0 - u)).square().mean()
    initial = model(points.initial_x, points.initial_t)
    upper = model(points.upper_x, points.upper_t)
    lower = model(points.lower_x, points.lower_t)
    if config.pde == "wave":
        target = torch.sin(torch.pi * points.initial_x) + 0.5 * torch.sin(
            config.beta * torch.pi * points.initial_x
        )
        initial_t = torch.autograd.grad(
            initial,
            points.initial_t,
            torch.ones_like(initial),
            create_graph=True,
            retain_graph=True,
        )[0]
        boundary = upper.square().mean() + lower.square().mean()
        initial_loss = (initial - target).square().mean() + initial_t.square().mean()
    else:
        target = (
            torch.sin(points.initial_x)
            if config.pde == "convection"
            else torch.exp(-0.5 * ((points.initial_x - torch.pi) / (torch.pi / 4.0)).square())
        )
        boundary = (upper - lower).square().mean()
        initial_loss = (initial - target).square().mean()
    return {"residual": residual, "boundary": boundary, "initial": initial_loss}


def pinn_loss(model: PINN, points: Collocation, config: PINNConfig) -> Tensor:
    return sum(loss_components(model, points, config).values())


@torch.no_grad()
def relative_l2(model: PINN, config: PINNConfig, *, device: torch.device | str) -> float:
    """Official 129 x 101 relative-L2 evaluation."""
    num_x = (config.num_x - 1) // 2 + 1
    x, t = np.meshgrid(np.linspace(*config.x_range, num_x), np.linspace(0.0, 1.0, config.num_t))
    dtype = next(model.parameters()).dtype
    xt = torch.as_tensor(x.reshape(-1, 1), device=device, dtype=dtype)
    tt = torch.as_tensor(t.reshape(-1, 1), device=device, dtype=dtype)
    prediction = model(xt, tt).double()
    x64, t64 = (xt.double(), tt.double())
    if config.pde == "wave":
        target = torch.sin(torch.pi * x64) * torch.cos(2.0 * torch.pi * t64)
        target += (
            0.5
            * torch.sin(config.beta * torch.pi * x64)
            * torch.cos(2.0 * config.beta * torch.pi * t64)
        )
    elif config.pde == "convection":
        target = torch.sin(x64 - config.beta * t64)
    else:
        initial = torch.exp(-0.5 * ((x64 - torch.pi) / (torch.pi / 4.0)).square())
        growth = torch.exp(config.rho * t64)
        target = initial * growth / (initial * growth + 1.0 - initial)
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target))


def matrix_routes(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Route the two 200x200 hidden weights to matrix optimizers."""
    matrices, fallback = ([], [])
    for parameter in model.parameters():
        (
            matrices
            if parameter.ndim == 2 and parameter.shape[0] == parameter.shape[1]
            else fallback
        ).append(parameter)
    return (matrices, fallback)
