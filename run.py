"""Run one paper configuration; no scheduler or cluster-specific dependencies."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path


RUNNERS = {
    "rnn": ("experiments.rnn.runner", "RNNAddingRunConfig", "run"),
    "rnn_structured": ("experiments.structured.rnn", None, "run"),
    "mnist_stochastic": ("experiments.mnist.runner", "AutoencoderRunConfig", "run"),
    "mnist_deterministic": (
        "experiments.mnist.deterministic_runner",
        "DeterministicRunConfig",
        "run",
    ),
    "mnist_structured": ("experiments.mnist.structured_runner", "StructuredRunConfig", "run"),
    "mnist_lbfgs": (
        "experiments.mnist.lbfgs_budget_runner",
        "DeterministicRunConfig",
        "run_exact_lbfgs",
    ),
    "pinn": ("experiments.pinn.runner", "PINNRunConfig", "run"),
    "pinn_structured": ("experiments.structured.pinn", None, "run"),
    "pidm": ("experiments.pidm.runner", "Config", "run"),
    "llm": ("experiments.llm.runner", "GPT2RunConfig", "run"),
}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--list", action="store_true", help="Show available configurations without running"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Override a config field, e.g. --set gradient_budget=20",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    if args.list:
        for i, item in enumerate(payload["runs"]):
            c = item["config"]
            name = " / ".join(str(c[k]) for k in ("task", "pde", "mode", "method") if k in c)
            lr = c.get("learning_rate", c.get("lr"))
            print(f"{i:2d}  {item['task']}: {name}; LR={lr:g}")
        print("Final seeds:", payload["seeds"])
        return
    item = payload["runs"][args.index]
    if item["task"] == "piratenet":
        parser.error(
            "PirateNet configurations are provided for reference only. Its implementation is withheld pending redistribution permission; see THIRD_PARTY.md."
        )
    config = dict(item["config"])
    if args.seed is not None:
        config["seed"] = args.seed
        for field in ("selected", "final"):
            if field in config:
                config[field] = args.seed != 0
    if args.lr is not None:
        config["learning_rate" if "learning_rate" in config else "lr"] = args.lr
    for setting in args.set:
        key, value = setting.split("=", 1)
        if key not in config:
            parser.error(f"Unknown config field: {key}")
        config[key] = json.loads(value)
    identity = hashlib.sha256(
        json.dumps([item["task"], config], sort_keys=True).encode()
    ).hexdigest()[:12]
    output = (
        args.output.resolve() / f"{item['task']}-{config['method']}-s{config['seed']}-{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["SOFTSERVE_OUTPUT_DIR"] = str(output)
    os.environ["SOFTSERVE_RESULTS_ROOT"] = str(output)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    name, config_class, function = RUNNERS[item["task"]]
    module = importlib.import_module(name)
    cfg = getattr(module, config_class)(**config) if config_class else config
    fn = getattr(module, function)
    print(f"Running {item['task']} / {config['method']} / seed {config['seed']}", flush=True)
    print(f"Output: {output}", flush=True)
    if item["task"] == "piratenet":
        row = fn(cfg, identity)
    elif item["task"] == "pidm":
        fn(cfg)
        native = output / "runs" / cfg.identity()
        row = json.loads((native / "result.json").read_text())
        row["history"] = [
            json.loads(s) for s in (native / "progress.jsonl").read_text().splitlines()
        ]
    elif item["task"] == "mnist_lbfgs":
        row = fn(cfg, "lbfgs", "0")
    elif item["task"] == "llm":
        from experiments.llm.progress import Progress

        row = fn(cfg, observer=Progress(output))
    else:
        row = fn(cfg)
    row.setdefault("config", config)
    row.setdefault("method", config["method"])
    row.setdefault("seed", config["seed"])
    row["task"] = item["task"]
    row["plot_gradient_cap"] = item.get("plot_gradient_cap")
    (output / "result.json").write_text(json.dumps(json_safe(row), allow_nan=False) + "\n")
    print(f"Finished: {row['status']}; {output / 'result.json'}", flush=True)
    if row["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
