"""Exact reference and Gram Newton--Schulz QME solvers."""

from __future__ import annotations
import math
import torch


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _eye(size: int, like: torch.Tensor) -> torch.Tensor:
    return torch.eye(size, dtype=like.dtype, device=like.device)


def _blend(old: torch.Tensor, new: torch.Tensor, beta: float) -> torch.Tensor:
    return new if beta == 0.0 else beta * old + (1.0 - beta) * new


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor if tensor.dtype in {torch.float32, torch.float64} else tensor.float()
    return torch.linalg.vector_norm(value) / math.sqrt(tensor.numel())


def _scaled_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Overflow/underflow-safe Euclidean norm in the tensor's reduction dtype."""
    absolute = tensor.abs()
    scale = absolute.amax()
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    norm = safe_scale * torch.sqrt((absolute / safe_scale).square().sum())
    return torch.where(scale == 0, torch.zeros_like(norm), norm)


def qme(quadratic: torch.Tensor, rhs: torch.Tensor, eps: float | None = 1e-10) -> torch.Tensor:
    """Positive solution of ``X @ quadratic @ X + X = rhs``.

    The eigendecompositions are evaluated in FP64, matching the reference
    implementation used for the controlled experiments.
    """
    dtype = quadratic.dtype
    U, V = (_sym(quadratic).double(), _sym(rhs).double())
    floor = torch.finfo(torch.float64).tiny if eps is None else eps
    values, vectors = torch.linalg.eigh(V)
    root_v = vectors * values.clamp_min(floor).sqrt() @ vectors.T
    inner = _sym(root_v @ U @ root_v)
    values, vectors = torch.linalg.eigh(_eye(inner.shape[0], inner) + 4.0 * inner)
    inverse_sum = vectors * (1.0 / (1.0 + values.clamp_min(floor).sqrt())) @ vectors.T
    return _sym(2.0 * root_v @ inverse_sum @ root_v).to(dtype)


def _spd_solve(matrix: torch.Tensor, rhs: torch.Tensor, eps: float | None) -> torch.Tensor:
    value = _sym(matrix).double()
    if eps is not None:
        value = value + eps * _eye(value.shape[0], value)
    return torch.linalg.solve(value, rhs.double()).to(rhs.dtype)


def _normalize_factors(
    A: torch.Tensor, G: torch.Tensor, gauge: str
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_a = A.diagonal(dim1=-2, dim2=-1).sum(-1) / A.shape[-1]
    if gauge == "trace_a":
        scale = mean_a
    elif gauge == "balanced_trace":
        mean_g = G.diagonal(dim1=-2, dim2=-1).sum(-1) / G.shape[-1]
        scale = torch.sqrt(mean_a.clamp_min(1e-12) / mean_g.clamp_min(1e-12))
    else:
        raise ValueError("gauge must be 'trace_a' or 'balanced_trace'")
    scale = scale.clamp_min(1e-12)
    return (A / scale[..., None, None], G * scale[..., None, None])


def kron_sweep(
    A: torch.Tensor,
    G: torch.Tensor,
    S: torch.Tensor,
    Y: torch.Tensor,
    lam: float,
    *,
    normalize: bool = True,
    gauge: str = "balanced_trace",
    eps: float | None = 1e-10,
    diagnostics: dict[str, torch.Tensor] | None = None,
    inverse_g_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact alternating SoftSERVE-Kron sweep.

    ``inverse_g_s`` may carry the exact ``G^-1 S`` result already computed by
    pair observation.  It is transient workspace rather than optimizer state.
    """
    n, m = S.shape

    def solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return _spd_solve(matrix, rhs, eps)

    if inverse_g_s is None:
        solved_g_s = solve(G, S)
    else:
        if (
            inverse_g_s.shape != S.shape
            or inverse_g_s.dtype != S.dtype
            or inverse_g_s.device != S.device
        ):
            raise ValueError("inverse_g_s must match S in shape, dtype, and device")
        solved_g_s = inverse_g_s
    U_A = lam / n * (Y.T @ G @ Y)
    V_A = A + lam / n * (S.T @ solved_g_s)
    A_next = qme(U_A, V_A, eps)
    solved_a_rhs = solve(A_next, torch.cat((A, S.T), dim=-1))
    alpha = torch.trace(solved_a_rhs[:, :m]) / m
    solved_a_st = solved_a_rhs[:, m:]
    U_G = lam / m * (Y @ A_next @ Y.T)
    V_G = alpha * G + lam / m * (S @ solved_a_st)
    G_next = qme(U_G, V_G, eps)
    if diagnostics is not None:
        diagnostics["A"] = (
            torch.linalg.matrix_norm(A_next @ U_A @ A_next + A_next - V_A)
            / torch.linalg.matrix_norm(V_A).clamp_min(1e-30)
        ).detach()
        diagnostics["G"] = (
            torch.linalg.matrix_norm(G_next @ U_G @ G_next + G_next - V_G)
            / torch.linalg.matrix_norm(V_G).clamp_min(1e-30)
        ).detach()
    return _normalize_factors(A_next, G_next, gauge) if normalize else (A_next, G_next)


def _root_pair_ns(
    matrix: torch.Tensor, steps: int, diagnostics: dict[str, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        matrix.square()
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(torch.finfo(matrix.dtype).tiny)
    )
    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device).expand(matrix.shape)
    root, inverse_root = (matrix / scale, eye.clone())
    for _ in range(steps):
        update = 0.5 * (3.0 * eye - inverse_root @ root)
        root, inverse_root = (_sym(root @ update), _sym(update @ inverse_root))
    root_scale = scale.sqrt()
    root, inverse_root = (root_scale * root, inverse_root / root_scale)
    if diagnostics is not None:
        denominator = matrix.square().sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
        identity_scale = math.sqrt(size)
        diagnostics["root"] = (
            (root @ root - matrix).square().sum(dim=(-2, -1)).sqrt() / denominator
        ).detach()
        diagnostics["inverse_root"] = (
            (inverse_root @ matrix @ inverse_root - eye).square().sum(dim=(-2, -1)).sqrt()
            / identity_scale
        ).detach()
        diagnostics["root_pair"] = (
            (root @ inverse_root - eye).square().sum(dim=(-2, -1)).sqrt() / identity_scale
        ).detach()
    return (root, inverse_root)


def _inverse_ns(
    matrix: torch.Tensor, steps: int, diagnostics: dict[str, torch.Tensor] | None = None
) -> torch.Tensor:
    norm = (
        matrix.square()
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(torch.finfo(matrix.dtype).tiny)
    )
    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device).expand(matrix.shape)
    inverse = eye / norm
    for _ in range(steps):
        inverse = _sym(inverse @ (2.0 * eye - matrix @ inverse))
    if diagnostics is not None:
        diagnostics["inverse"] = (
            (matrix @ inverse - eye).square().sum(dim=(-2, -1)).sqrt() / math.sqrt(size)
        ).detach()
    return inverse


def _gemm_qme(
    factor: torch.Tensor,
    rhs: torch.Tensor,
    root_steps: int,
    inverse_steps: int,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    rank = factor.shape[-1]
    eye = torch.eye(rank, dtype=factor.dtype, device=factor.device).expand(
        factor.shape[:-2] + (rank, rank)
    )
    inner = _sym(eye + 4.0 * (factor.transpose(-1, -2) @ rhs @ factor))
    root_diag: dict[str, torch.Tensor] = {}
    inverse_diag: dict[str, torch.Tensor] = {}
    root, _ = _root_pair_ns(inner, root_steps, root_diag)
    inverse = _inverse_ns(eye + root, inverse_steps, inverse_diag)
    if diagnostics is not None:
        diagnostics.update({f"qme_{key}": value for key, value in root_diag.items()})
        diagnostics.update(inverse_diag)
    product = rhs @ factor @ inverse
    return _sym(rhs - 4.0 * (product @ product.transpose(-1, -2)))


def gemm_kron_sweep(
    A: torch.Tensor,
    G: torch.Tensor,
    S: torch.Tensor,
    Y: torch.Tensor,
    lam: float | torch.Tensor,
    root_steps: int,
    inverse_steps: int,
    *,
    normalize: bool = True,
    gauge: str = "balanced_trace",
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batched, factorized Newton--Schulz Kronecker sweep."""
    lam_value = torch.as_tensor(lam, dtype=A.dtype, device=A.device)
    if lam_value.shape not in {torch.Size(), A.shape[:-2]}:
        raise ValueError("lam must be scalar or match the matrix batch shape")
    if not bool(torch.isfinite(lam_value).all()) or not bool((lam_value > 0).all()):
        raise ValueError("lam must be finite and positive")
    n, m = S.shape[-2:]
    lam_matrix = lam_value[..., None, None]
    root_G_diag: dict[str, torch.Tensor] = {}
    root_G, inverse_root_G = _root_pair_ns(G, root_steps, root_G_diag)
    transformed_S = inverse_root_G @ S
    V_A = _sym(A + lam_matrix / n * (transformed_S.transpose(-1, -2) @ transformed_S))
    R_A = torch.sqrt(lam_matrix / n) * (Y.transpose(-1, -2) @ root_G)
    U_A = _sym(R_A @ R_A.transpose(-1, -2))
    qme_A_diag: dict[str, torch.Tensor] = {}
    A_next = _gemm_qme(R_A, V_A, root_steps, inverse_steps, qme_A_diag)
    root_A_diag: dict[str, torch.Tensor] = {}
    root_A, inverse_root_A = _root_pair_ns(A_next, root_steps, root_A_diag)
    alpha = (inverse_root_A @ A @ inverse_root_A).diagonal(dim1=-2, dim2=-1).sum(-1) / m
    transformed_S = S @ inverse_root_A
    V_G = _sym(
        alpha[..., None, None] * G
        + lam_matrix / m * (transformed_S @ transformed_S.transpose(-1, -2))
    )
    R_G = torch.sqrt(lam_matrix / m) * (Y @ root_A)
    U_G = _sym(R_G @ R_G.transpose(-1, -2))
    qme_G_diag: dict[str, torch.Tensor] = {}
    G_next = _gemm_qme(R_G, V_G, root_steps, inverse_steps, qme_G_diag)
    if diagnostics is not None:
        diagnostics["A"] = (
            (A_next @ U_A @ A_next + A_next - V_A).square().sum(dim=(-2, -1)).sqrt()
            / V_A.square().sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
        ).detach()
        diagnostics["G"] = (
            (G_next @ U_G @ G_next + G_next - V_G).square().sum(dim=(-2, -1)).sqrt()
            / V_G.square().sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
        ).detach()
        diagnostics["ns_root"] = torch.stack(
            (
                root_G_diag["root"],
                root_A_diag["root"],
                qme_A_diag["qme_root"],
                qme_G_diag["qme_root"],
            )
        ).amax(dim=0)
        diagnostics["ns_inverse_root"] = torch.stack(
            (
                root_G_diag["inverse_root"],
                root_A_diag["inverse_root"],
                qme_A_diag["qme_inverse_root"],
                qme_G_diag["qme_inverse_root"],
            )
        ).amax(dim=0)
        diagnostics["ns_root_pair"] = torch.stack(
            (
                root_G_diag["root_pair"],
                root_A_diag["root_pair"],
                qme_A_diag["qme_root_pair"],
                qme_G_diag["qme_root_pair"],
            )
        ).amax(dim=0)
        diagnostics["ns_inverse"] = torch.stack(
            (qme_A_diag["inverse"], qme_G_diag["inverse"])
        ).amax(dim=0)
    return _normalize_factors(A_next, G_next, gauge) if normalize else (A_next, G_next)


def _diag_update(
    h: torch.Tensor, s: torch.Tensor, y: torch.Tensor, lam: float | torch.Tensor
) -> torch.Tensor:
    value = h + lam * s.square()
    root = (1.0 + 4.0 * lam * y.square() * value).clamp_min(1e-20).sqrt()
    return 2.0 * value / (1.0 + root)


def _diag_pair_metrics(
    h: torch.Tensor, s: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s_metric = (s.square() / h).sum()
    y_metric = (h * y.square()).sum()
    q = (s_metric + y_metric) / (2.0 * s.numel())
    denominator = torch.sqrt(s_metric.clamp_min(0.0) * y_metric.clamp_min(0.0))
    chi = ((s * y).sum() / denominator.clamp_min(1e-30)).clamp(-1.0, 1.0)
    return (q, chi)


def _kron_pair_metrics(
    A: torch.Tensor, G: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, inverse_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_A, inverse_G = (_inverse_ns(A, inverse_steps), _inverse_ns(G, inverse_steps))
    s_metric = (S * (inverse_G @ S @ inverse_A)).sum(dim=(-2, -1))
    y_metric = (Y * (G @ Y @ A)).sum(dim=(-2, -1))
    q = (s_metric + y_metric) / (2.0 * S.shape[-2] * S.shape[-1])
    denominator = torch.sqrt(s_metric.clamp_min(0.0) * y_metric.clamp_min(0.0))
    chi = ((S * Y).sum(dim=(-2, -1)) / denominator.clamp_min(1e-30)).clamp(-1, 1)
    return (q, chi)


symmetrize = _sym
root_pair_ns = _root_pair_ns
inverse_ns = _inverse_ns
gram_qme = _gemm_qme
tensor_rms = _rms
