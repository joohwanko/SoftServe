"""Algebraically identical compact inverse-L-BFGS application.

For S,Y histories and identity H0, R=triu(S^T Y), D=diag(S^T Y),
W=S R^{-T}, H=I-WY^T-YW^T+W(D+Y^T Y)W^T.
This is a different evaluation order of the reference two-loop recursion,
not an optimizer variant. Float64 assembly reduces cancellation; cache the
result until the history changes. Fall back to the original two-loop if the
compact matrix is numerically unreliable. Extra dense cache bytes are reported.
"""

import torch
from softserve.baselines_kron_qn import _apply_inverse_bfgs
from .core import MatrixBaseline


@torch.no_grad()
def compact_inverse(s_values, y_values):
    s = torch.stack(s_values, dim=1).double()
    y = torch.stack(y_values, dim=1).double()
    sty = s.T @ y
    r = torch.triu(sty)
    w = torch.linalg.solve_triangular(r, s.T, upper=True).T
    wy = w @ y.T
    h = (
        torch.eye(s.shape[0], dtype=s.dtype, device=s.device)
        - wy
        - wy.T
        + w @ (torch.diag(sty.diagonal()) + y.T @ y) @ w.T
    )
    return (h + h.T) * 0.5


class CompactKBFGSL(MatrixBaseline):
    @staticmethod
    def inverse_output(state, value):
        if not state["s"] or value.shape[0] == 1:
            return MatrixBaseline.inverse_output(state, value)
        version = (len(state["s"]), id(state["s"][-1]), id(state["y"][-1]))
        if state.get("compact_version") != version:
            state["compact_version"] = version
            state["compact_updates"] = state.get("compact_updates", 0) + 1
            state["compact_H"] = None
            try:
                h = compact_inverse(state["s"], state["y"])
                _, info = torch.linalg.cholesky_ex(h)
                if int(info) == 0 and torch.isfinite(h).all():
                    s, y = (state["s"][-1].double(), state["y"][-1].double())
                    relative = torch.linalg.vector_norm(h @ y - s) / torch.linalg.vector_norm(
                        s
                    ).clamp_min(1e-30)
                    if state["compact_updates"] <= 2 or state["compact_updates"] % 20 == 0:
                        probe = torch.stack((s, y), dim=1)
                        reference = _apply_inverse_bfgs(
                            probe,
                            [v.double() for v in state["s"]],
                            [v.double() for v in state["y"]],
                        )
                        predicted = h @ probe
                        relative = torch.maximum(
                            relative,
                            torch.linalg.vector_norm(predicted - reference)
                            / torch.linalg.vector_norm(reference).clamp_min(1e-30),
                        )
                    if relative <= 1e-08:
                        converted = h.to(value.dtype)
                        _, converted_info = torch.linalg.cholesky_ex(converted)
                        if int(converted_info) == 0:
                            state["compact_H"] = converted
            except RuntimeError:
                pass
            if state["compact_H"] is None:
                state["compact_fallbacks"] = state.get("compact_fallbacks", 0) + 1
        if state["compact_H"] is None:
            return MatrixBaseline.inverse_output(state, value)
        return state["compact_H"] @ value
