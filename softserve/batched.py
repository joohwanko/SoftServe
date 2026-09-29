"""Shape-batched Muon and fixed-lambda SoftSERVE updates for PIDM."""

from collections import defaultdict
import math
import torch
from softserve.optim import SoftServeKron


class BatchedMuon(torch.optim.Optimizer):
    """Canonical torch Muon NS5 arithmetic, grouped by matrix shape."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))
        groups = defaultdict(list)
        for p in self.param_groups[0]["params"]:
            groups[tuple(p.shape)].append(p)
        self.buckets = list(groups.values())
        self.buffers = [
            torch.zeros((len(ps), *ps[0].shape), device=ps[0].device) for ps in self.buckets
        ]
        for ps, buf in zip(self.buckets, self.buffers):
            for p, m in zip(ps, buf.unbind(0)):
                self.state[p]["momentum_buffer"] = m

    @torch.no_grad()
    def step(self):
        lr = self.param_groups[0]["lr"]
        for ps, buf in zip(self.buckets, self.buffers):
            gs = torch.stack([torch.zeros_like(p) if p.grad is None else p.grad for p in ps])
            buf.lerp_(gs, 0.05)
            update = gs.lerp(buf, 0.95).bfloat16()
            transpose = update.shape[-2] > update.shape[-1]
            if transpose:
                update = update.transpose(-2, -1)
            update.div_(update.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-07))
            for _ in range(5):
                gram = update @ update.transpose(-2, -1)
                poly = torch.baddbmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
                update = torch.baddbmm(update, poly, update, beta=3.4445)
            if transpose:
                update = update.transpose(-2, -1)
            scale = -lr * 0.2 * math.sqrt(max(ps[0].shape))
            torch._foreach_add_(ps, list(update.unbind(0)), alpha=scale)


class BatchedKron(SoftServeKron):
    """Vectorize the unchanged fixed-λ, bias-corrected momentum/constraint step.

    Inherit secant handling and the entire QME refresh from SoftServeKron.
    Batched versus original parameter updates are checked before training.
    """

    @torch.no_grad()
    def parameter_step(self):
        self.last_metric_denominators, self.last_metric_normalization_valid = ([], [])
        for group in self.param_groups:
            assert group["lambda_schedule"] == "fixed" and (not group["nesterov"])
            assert group["constrained_update"] and (not group["metric_rms_constraint"])
            assert group["update_rms"] is None and group["global_update_rms"] is None
            counter = self.state[group["params"][0]]
            counter["_momentum_steps"] = counter.get("_momentum_steps", 0) + 1
            beta = group["beta1"]
            correction = 1 - beta ** counter["_momentum_steps"]
            for bucket in self._buckets_by_group[id(group)]:
                gradients = torch.stack(
                    [torch.zeros_like(p) if p.grad is None else p.grad for p in bucket.params]
                )
                if not hasattr(bucket, "_campaign_momentum"):
                    momentum = torch.stack(
                        [self.state[p].get("momentum", torch.zeros_like(p)) for p in bucket.params]
                    )
                    bucket._campaign_momentum = momentum
                    for p, m in zip(bucket.params, momentum.unbind(0)):
                        self.state[p]["momentum"] = m
                momentum = bucket._campaign_momentum
                momentum.mul_(beta).add_(gradients, alpha=1 - beta)
                value = momentum / correction
                direction = bucket.G @ value @ bucket.A
                quadratic = (value * direction).sum(dim=(-2, -1), keepdim=True)
                positive = torch.isfinite(quadratic) & (quadratic > 0)
                zero = torch.isfinite(quadratic) & (quadratic == 0)
                scale = torch.where(
                    positive,
                    quadratic.rsqrt(),
                    torch.where(
                        zero, torch.zeros_like(quadratic), torch.full_like(quadratic, torch.nan)
                    ),
                )
                direction = direction * scale
                self.last_metric_denominators.extend(
                    quadratic.clamp_min(0).sqrt().flatten().unbind()
                )
                self.last_metric_normalization_valid.extend((positive | zero).flatten().unbind())
                torch._foreach_add_(
                    list(bucket.params), list(direction.unbind(0)), alpha=-group["lr"]
                )
        self.last_direction_identity_stats = None
