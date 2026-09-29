"""Standalone QME solution-error ablation from the appendix."""

import argparse
import csv
import math
from pathlib import Path
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
from softserve import optim

OUT = Path("outputs/qme")


def norm(x):
    return torch.linalg.vector_norm(x.double())


def relative(x, ref):
    return float(norm(x.double() - ref.double()) / norm(ref).clamp_min(1e-300))


def residual(q, u, v):
    q, u, v = (q.double(), u.double(), v.double())
    return float(norm(q @ u @ q + q - v) / norm(v).clamp_min(1e-300))


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


ROOT_STEPS = (4, 8, 12, 18, 24)
INVERSE_STEPS = (2, 4, 6, 10, 16)
CONDITIONS = (10, 1000, 100000)
SETTINGS = [(r, i) for r in ROOT_STEPS for i in INVERSE_STEPS] + [(5, 5)]


@torch.no_grad()
def compute():
    rows = []
    n = 256
    for condition in CONDITIONS:
        for seed in range(3):
            gen = torch.Generator().manual_seed(92100 + 100 * n + seed)
            ov = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64)).Q
            ou = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64)).Q
            eig = torch.logspace(0, math.log10(condition), n, dtype=torch.float64)
            v = (ov * eig @ ov.T).float()
            r = (ou * torch.linspace(0.7, 1.3, n, dtype=torch.float64)).float()
            u = r.double() @ r.double().T
            reference = optim.qme(u, v.double(), eps=None)
            ref_residual = residual(reference, u, v)
            ref_min = float(torch.linalg.eigvalsh(reference).min())
            assert ref_residual < 1e-07 and ref_min > 0
            for root, inverse in SETTINGS:
                approx = optim._gemm_qme(r, v, root, inverse)
                is_finite = bool(torch.isfinite(approx).all())
                rows.append(
                    dict(
                        n=n,
                        condition=condition,
                        seed=seed,
                        root=root,
                        inverse=inverse,
                        device="cpu",
                        dtype="float32",
                        error=relative(approx, reference) if is_finite else float("nan"),
                        residual=residual(approx, u, v) if is_finite else float("nan"),
                        min_eigenvalue=float(torch.linalg.eigvalsh(approx.double()).min())
                        if is_finite
                        else float("nan"),
                        status="finite" if is_finite else "nonfinite",
                        reference_residual=ref_residual,
                        reference_min_eigenvalue=ref_min,
                    )
                )
            print(f"Finished condition={condition}, matrix seed={seed}", flush=True)
    write_csv(OUT / "qme_grid.csv", rows)
    return rows


def cell(rows, condition, root, inverse):
    chosen = [
        r
        for r in rows
        if int(r["condition"]) == condition
        and int(r["root"]) == root
        and (int(r["inverse"]) == inverse)
    ]
    assert len(chosen) == 3 and sorted((int(r["seed"]) for r in chosen)) == [0, 1, 2]
    vals = [float(r["error"]) for r in chosen]
    return statistics.median(vals) if all((math.isfinite(v) for v in vals)) else float("nan")


def scientific(value):
    return f"{value:.1e}".replace("e-0", "e−").replace("e+0", "e").replace("e+", "e")


def plot(rows):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    grids = [
        np.array([[cell(rows, c, r, i) for i in INVERSE_STEPS] for r in ROOT_STEPS])
        for c in CONDITIONS
    ]
    values = np.asarray(grids)
    finite = values[np.isfinite(values)]
    assert len(finite) > 0 and (finite > 0).all()
    lower = 10.0 ** math.floor(math.log10(float(finite.min())))
    upper = 10.0 ** math.ceil(math.log10(float(finite.max())))
    norm = LogNorm(vmin=lower, vmax=upper)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#c4c4c4")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), sharey=True, layout="constrained")
    for ax, condition, grid in zip(axes, CONDITIONS, grids):
        im = ax.imshow(
            grid, cmap=cmap, norm=norm, origin="lower", aspect="auto", interpolation="nearest"
        )
        ax.set_xticks(range(len(INVERSE_STEPS)), [str(i) for i in INVERSE_STEPS])
        ax.set_yticks(range(len(ROOT_STEPS)), [str(r) for r in ROOT_STEPS])
        ax.tick_params(length=0, pad=6)
        ax.set_xlabel("Inverse iterations")
        ax.set_title({10: "κ(V) = 10", 1000: "κ(V) = 10³", 100000: "κ(V) = 10⁵"}[condition])
        for j, r in enumerate(ROOT_STEPS):
            for k, i in enumerate(INVERSE_STEPS):
                value = grid[j, k]
                label = scientific(value) if math.isfinite(value) else "failed"
                color = "white" if math.isfinite(value) and norm(value) < 0.6 else "#222222"
                ax.text(k, j, label, ha="center", va="center", fontsize=8.2, color=color)
        x, y = (INVERSE_STEPS.index(10), ROOT_STEPS.index(18))
        ax.add_patch(
            Rectangle(
                (x - 0.47, y - 0.47), 0.94, 0.94, fill=False, edgecolor="white", linewidth=1.8
            )
        )
        ax.add_patch(
            Rectangle(
                (x - 0.43, y - 0.43), 0.86, 0.86, fill=False, edgecolor="black", linewidth=0.6
            )
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Root iterations")
    colorbar = fig.colorbar(im, ax=list(axes), shrink=0.9, pad=0.025, fraction=0.025)
    colorbar.set_label("Relative solution error")
    colorbar.set_ticks(
        [10.0**p for p in range(int(math.log10(lower)), int(math.log10(upper)) + 1, 2)]
    )
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"ablation_qme_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    plot(compute())
