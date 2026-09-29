"""Explicit Taylor-mode PINN graph exposing derivative weight sharing.

References: Dangel et al., arXiv:2405.15603, equations 5/6 and KFAC-expand.
K-FAC preserves the sum of interior and boundary/initial Kronecker blocks.
K-BFGS uses merged shared-application moments and is labelled as an extension.
The optimized PDE loss is unchanged.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .core import Statistics


class JetPINN(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.matrices = nn.ParameterList(
            [
                nn.Parameter(torch.cat((layer.weight.detach(), layer.bias.detach()[:, None]), 1))
                for layer in original.linear
                if isinstance(layer, nn.Linear)
            ]
        )

    def forward(self, x, t):
        value = torch.cat((x, t), dim=1)
        for idx, matrix in enumerate(self.matrices):
            value = F.linear(torch.cat((value, torch.ones_like(value[:, :1])), 1), matrix)
            if idx + 1 < len(self.matrices):
                value = torch.tanh(value)
        return value

    def jet(self, x, t, kind, records):
        value = torch.cat((x, t), dim=1)
        one, zero = (torch.ones_like(x), torch.zeros_like(x))
        dx, dt = (torch.cat((one, zero), 1), torch.cat((zero, one), 1))
        if kind == "wave":
            values = [value, dx, dt, torch.zeros_like(value), torch.zeros_like(value)]
        elif kind == "xt":
            values = [value, dx, dt]
        elif kind == "t":
            values = [value, dt]
        elif kind == "value":
            values = [value]
        else:
            raise ValueError(kind)
        for idx, matrix in enumerate(self.matrices):
            augmented = torch.stack(
                [torch.cat((v, one if j == 0 else zero), 1) for j, v in enumerate(values)], 1
            )
            flat_input = augmented.reshape(-1, augmented.shape[-1])
            flat_z = F.linear(flat_input, matrix)
            records[idx].append((flat_input, flat_z))
            z = flat_z.reshape(len(x), len(values), -1)
            if idx + 1 == len(self.matrices):
                return z
            base = torch.tanh(z[:, 0])
            first = 1 - base.square()
            second = -2 * base * first
            if kind == "wave":
                values = [
                    base,
                    first * z[:, 1],
                    first * z[:, 2],
                    first * z[:, 3] + second * z[:, 1].square(),
                    first * z[:, 4] + second * z[:, 2].square(),
                ]
            else:
                values = [base] + [first * z[:, j] for j in range(1, len(values))]

    def residuals(self, points, config):
        records = [[] for _ in self.matrices]
        kind = {"wave": "wave", "convection": "xt", "reaction": "t"}[config.pde]
        values = self.jet(points.residual_x, points.residual_t, kind, records)
        u = values[:, 0]
        if config.pde == "wave":
            residual = values[:, 4] - 4 * values[:, 3]
        elif config.pde == "convection":
            residual = values[:, 2] + config.beta * values[:, 1]
        else:
            residual = values[:, 1] - config.rho * u * (1 - u)
        initial = self.jet(
            points.initial_x, points.initial_t, "t" if config.pde == "wave" else "value", records
        )
        upper = self.jet(points.upper_x, points.upper_t, "value", records)[:, 0]
        lower = self.jet(points.lower_x, points.lower_t, "value", records)[:, 0]
        if config.pde == "wave":
            target = torch.sin(torch.pi * points.initial_x) + 0.5 * torch.sin(
                config.beta * torch.pi * points.initial_x
            )
            terms = {
                "residual": [residual],
                "boundary": [upper, lower],
                "initial": [initial[:, 0] - target, initial[:, 1]],
            }
        else:
            target = (
                torch.sin(points.initial_x)
                if config.pde == "convection"
                else torch.exp(-0.5 * ((points.initial_x - torch.pi) / (torch.pi / 4)).square())
            )
            terms = {
                "residual": [residual],
                "boundary": [upper - lower],
                "initial": [initial[:, 0] - target],
            }
        return (terms, records)


def gradient_statistics(model, points, config, *, fisher, generator=None, fisher_dtype=None):
    terms, records = model.residuals(points, config)
    residuals = [r for group in terms.values() for r in group]
    loss = sum((r.square().mean() for r in residuals))
    zs = [z for group in records for _, z in group]
    values = torch.autograd.grad(loss, (*model.matrices, *zs), retain_graph=fisher)
    count = len(model.matrices)
    gradients, deltas = (values[:count], values[count:])
    probe_deltas = None
    if fisher:
        probe = sum(
            (
                math.sqrt(2 / r.numel())
                * (
                    r * (2 * torch.randint(0, 2, r.shape, device=r.device, generator=generator) - 1)
                ).sum()
                for r in residuals
            )
        )
        probe_deltas = torch.autograd.grad(probe, zs)
    stats, offset = ([], 0)
    with torch.no_grad():
        for group in records:
            n = sum((len(a) for a, _ in group))
            activation_parts = [a.detach().T @ a.detach() for a, _ in group]
            activation = sum(activation_parts) / n
            mean_input = sum((a.detach().sum(0) for a, _ in group)) / n
            mean_z = sum((z.detach().sum(0) for _, z in group)) / n
            raw_delta = deltas[offset : offset + len(group)]
            adjoint_sum = sum((d.detach().sum(0) for d in raw_delta))
            gradient_parts = None
            if fisher:
                gradient_parts = []
                for d in probe_deltas[offset : offset + len(group)]:
                    value = d.detach() if fisher_dtype is None else d.detach().to(fisher_dtype)
                    gradient_parts.append(value.T @ value)
            g = sum(gradient_parts) if fisher else None
            stat = Statistics(activation, mean_input, mean_z, adjoint_sum, g)
            if fisher:
                parts = []
                for start, stop in ((0, 1), (1, len(group))):
                    subset = group[start:stop]
                    n_part = sum((len(a) for a, _ in subset))
                    parts.append(
                        (
                            sum(activation_parts[start:stop]) / n_part,
                            sum(gradient_parts[start:stop]),
                        )
                    )
                stat.kfac_components = parts
            stats.append(stat)
            offset += len(group)
    return (float(loss.detach()), tuple((v.detach() for v in gradients)), stats)
