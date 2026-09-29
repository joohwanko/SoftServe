"""PINN K-FAC: exact per-layer solve of two damped Kronecker products.

The Taylor-mode KFAC-expand curvature follows arXiv:2405.15603. We use the
campaign's fixed first-moment convention and swept LR, not paper-specific line
search or automatic quadratic-model momentum. Auxiliary GGN VJPs count.
"""

import math
import torch
from .core import MatrixBaseline


@torch.no_grad()
def sum_kron_basis(a0, g0, a1, g1):
    """Solve G0 D A0 + G1 D A1 = M without an mn-by-mn matrix."""
    la, lg = (torch.linalg.cholesky(a0), torch.linalg.cholesky(g0))

    def generalized(left_cholesky, other):
        left = torch.linalg.solve_triangular(left_cholesky, other, upper=False)
        white = torch.linalg.solve_triangular(left_cholesky, left.T, upper=False).T
        eig, vec = torch.linalg.eigh((white + white.T) * 0.5)
        tolerance = 32 * torch.finfo(eig.dtype).eps * eig.abs().max().clamp_min(1)
        if eig.min() < -tolerance:
            raise FloatingPointError("indefinite whitened K-FAC factor")
        transformed = torch.linalg.solve_triangular(left_cholesky.T, vec, upper=True)
        return (eig.clamp_min(0), transformed)

    ea, ta = generalized(la, a1)
    eg, tg = generalized(lg, g1)
    return (ta, tg, 1 + eg[:, None] * ea[None, :])


class PINNKFAC(MatrixBaseline):
    @torch.no_grad()
    def apply(self, gradients, statistics):
        root = math.sqrt(self.damping)
        for matrix, gradient, stat, state in zip(
            self.matrices, gradients, statistics, self.state, strict=True
        ):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite gradient")
            if "parts" not in state:
                state["parts"] = [[a.clone(), g.clone()] for a, g in stat.kfac_components]
            else:
                for old, new in zip(state["parts"], stat.kfac_components, strict=True):
                    old[0].lerp_(new[0], 1 - self.factor_decay)
                    old[1].lerp_(new[1], 1 - self.factor_decay)
            state["M"].mul_(self.momentum).add_(gradient, alpha=1 - self.momentum)
            if self.steps % self.inverse_frequency == 0 or self.steps <= 20:
                (a0, g0), (a1, g1) = state["parts"]
                solve_dtype = torch.promote_types(a0.dtype, g0.dtype)
                a0, g0, a1, g1 = (value.to(solve_dtype) for value in (a0, g0, a1, g1))
                ia = torch.eye(matrix.shape[1], dtype=solve_dtype, device=matrix.device)
                ig = torch.eye(matrix.shape[0], dtype=solve_dtype, device=matrix.device)
                state["basis"] = sum_kron_basis(
                    a0 + root * ia, g0 + root * ig, a1 + root * ia, g1 + root * ig
                )
            ta, tg, denominator = state["basis"]
            direction = tg @ (tg.T @ state["M"].to(tg.dtype) @ ta / denominator) @ ta.T
            if not torch.isfinite(direction).all():
                raise FloatingPointError("nonfinite sum-Kronecker direction")
            direction = direction.to(matrix.dtype)
            if not torch.isfinite(direction).all():
                raise FloatingPointError("nonfinite sum-Kronecker direction after dtype cast")
            matrix.add_(direction, alpha=-self.lr)
        self.steps += 1
