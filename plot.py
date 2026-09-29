"""Plot optimization paths or learning-rate sweeps from local result.json files."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "softserve_kron": "#D55E00",
    "kron": "#D55E00",
    "softserve_diag": "#CC79A7",
    "sgd_m": "#808080",
    "adam": "#333333",
    "adamw": "#333333",
    "muon": "#0072B2",
    "soap": "#009E73",
    "kfac": "#56B4E9",
    "kbfgs_l": "#8064A2",
    "lbfgs": "#111111",
    "lbfgs_cold": "#111111",
    "lbfgs_warm": "#E69F00",
}
LABELS = {
    "softserve_kron": "SoftServe-Kron",
    "kron": "SoftServe-Kron",
    "softserve_diag": "SoftServe-Diag",
    "sgd_m": "SoftServe-I",
    "adam": "Adam",
    "adamw": "AdamW",
    "muon": "Muon",
    "soap": "SOAP",
    "kfac": "K-FAC",
    "kbfgs_l": "K-BFGS(L)",
    "lbfgs": "L-BFGS",
    "lbfgs_cold": "L-BFGS",
    "lbfgs_warm": "Adam+L-BFGS",
}


def get(record, key):
    for part in key.split("."):
        if not isinstance(record, dict):
            return None
        record = record.get(part)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metric", required=True, help="History field; dotted keys are supported")
    parser.add_argument(
        "--x",
        default="gradient_evaluations",
        choices=("gradient_evaluations", "parameter_steps", "wall_seconds", "unique_tokens"),
    )
    parser.add_argument("--mode", choices=("curves", "lr"), default="curves")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter a configuration, e.g. --where pde=wave --where mode=stochastic",
    )
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/optimization.pdf"))
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.results.rglob("result.json")):
        row = json.loads(path.read_text())
        # Native logs nested under a run directory are not counted twice.
        if not row.get("history") or not row.get("task", "").startswith(args.task):
            continue
        if any(str(row["config"].get(k)) != v for k, v in (s.split("=", 1) for s in args.where)):
            continue
        groups[row.get("method", row["config"]["method"])].append(row)
    if not groups:
        parser.error("No matching results")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for index, (method, rows) in enumerate(groups.items()):
        color, label = COLORS[method], LABELS[method]
        linestyle = ("-", "--", "-.", ":")[index % 4]
        if args.mode == "lr":
            rates = defaultdict(list)
            failures = []
            for r in rows:
                lr = r["config"].get("learning_rate", r["config"].get("lr"))
                value = get(r, args.metric)
                if r["status"] != "complete" or value is None or not math.isfinite(value):
                    failures.append(lr)
                else:
                    rates[lr].append(value)
            x = np.asarray(sorted(rates))
            y = np.asarray([np.mean(rates[t]) for t in x])
            ax.plot(x, y, marker="o", ms=4, color=color, ls=linestyle, label=label)
            if len(x):
                ax.fill_between(
                    x,
                    [min(rates[t]) for t in x],
                    [max(rates[t]) for t in x],
                    color=color,
                    alpha=0.15,
                )
            for lr in failures:
                ax.plot(
                    lr, 0.97, "x", color=color, transform=ax.get_xaxis_transform(), clip_on=False
                )
            continue
        if len({r["seed"] for r in rows}) != len(rows):
            parser.error(
                f"Multiple configurations for the same seed in {method}; use --where to select one"
            )
        traces = []
        for r in rows:
            points = []
            cap = r.get("plot_gradient_cap")
            for h in r["history"]:
                if cap is not None and h.get("gradient_evaluations", 0) > cap:
                    break
                x = h.get(args.x, h.get("step") if args.x == "parameter_steps" else None)
                y = get(h, args.metric)
                if x is not None and y is not None and math.isfinite(y) and (args.linear or y > 0):
                    points.append((x, y))
            if points:
                traces.append((r, np.asarray(points)))
        if not traces:
            continue
        scale = 1e3 if args.x == "gradient_evaluations" else 1e6 if args.x == "unique_tokens" else 1
        if any(r["status"] != "complete" for r, _ in traces):
            for i, (r, p) in enumerate(traces):
                ax.plot(
                    p[:, 0] / scale,
                    p[:, 1],
                    color=color,
                    ls=linestyle,
                    alpha=0.7,
                    label=label if i == 0 else None,
                )
                if r["status"] != "complete":
                    ax.plot(p[-1, 0] / scale, p[-1, 1], "x", color=color)
        else:
            lower, upper = max(p[0, 0] for _, p in traces), min(p[-1, 0] for _, p in traces)
            grid = np.linspace(lower, upper, 250)
            values = np.stack([np.interp(grid, p[:, 0], p[:, 1]) for _, p in traces])
            ax.plot(grid / scale, values.mean(0), color=color, ls=linestyle, label=label)
            ax.fill_between(grid / scale, values.min(0), values.max(0), color=color, alpha=0.15)
    if args.mode == "lr":
        ax.set_xscale("log")
        xlabel = "Learning rate"
    else:
        xlabel = {
            "gradient_evaluations": "Gradient evaluations (× 10³)",
            "parameter_steps": "Parameter updates",
            "wall_seconds": "Wall-clock time (s)",
            "unique_tokens": "Training tokens (millions)",
        }[args.x]
    ax.set(xlabel=xlabel, ylabel=args.ylabel or args.metric.replace("_", " "))
    if not args.linear:
        ax.set_yscale("log")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
