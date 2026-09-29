"""Fixed-budget original PDEs using Taylor-jet shared-weight baselines."""

import math
import time
from dataclasses import asdict
import torch
from softserve.benchmarks.pinn import (
    ResidualSampler,
    build_pinn,
    deterministic_sampler,
    loss_components,
    relative_l2,
)
from .jets import JetPINN, gradient_statistics
from .pinn_kfac import PINNKFAC
from .compact_kbfgs import CompactKBFGSL
from .pinn_problem import reference_problem
from .problem_parity import comparison_issue


def lr_multiplier(config, steps):
    """A pilot may stop early without compressing the full-run LR schedule."""
    horizon = config.get("schedule_gradient_budget", config["gradient_budget"])
    if horizon < config["gradient_budget"] or horizon % 2:
        raise ValueError("LR horizon must be even and cover the requested budget")
    progress = steps / max(horizon // 2 - 1, 1)
    return config["min_lr_ratio"] + (1 - config["min_lr_ratio"]) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def run(config):
    torch.set_num_threads(1)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    fisher_dtype = (
        getattr(torch, config["kfac_fisher_dtype"]) if "kfac_fisher_dtype" in config else None
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    issue = comparison_issue({"config": config})
    if issue:
        raise ValueError(issue)
    problem = reference_problem(config)
    original = build_pinn(problem, seed=config["seed"], device=device, dtype=dtype)
    audit_sampler = deterministic_sampler(problem, seed=config["seed"], device=device, dtype=dtype)
    sampler = (
        audit_sampler
        if config["mode"] == "deterministic"
        else ResidualSampler(
            problem,
            seed=config["seed"],
            device=device,
            dtype=dtype,
            continuous=True,
            namespace=config["sampling_namespace"],
        )
    )
    model = JetPINN(original)
    del original
    optimizer = (PINNKFAC if config["method"] == "kfac" else CompactKBFGSL)(
        model.matrices,
        method=config["method"],
        lr=config["learning_rate"],
        damping=config["damping"],
        momentum=config.get("momentum", 0.9),
        factor_decay=config.get("factor_decay", 0.9),
        inverse_frequency=config.get("inverse_frequency", 20),
        history_size=config.get("history_size", 100),
    )
    generator = torch.Generator(device=device).manual_seed(913000 + config["seed"])
    history = []
    calls = steps = auxiliary = 0
    current_loss = None
    status, reason = ("complete", None)

    def record():
        points = audit_sampler.batch(0)
        values = loss_components(model, points, problem)
        metrics = {k + "_loss": float(v.detach()) for k, v in values.items()}
        l2 = relative_l2(model, problem, device=device) if config["selected"] else None
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append(
            {
                "parameter_steps": steps,
                "gradient_evaluations": calls,
                "wall_seconds": time.perf_counter() - started,
                "training_loss": sum(metrics.values()),
                "training_batch_loss": current_loss,
                "relative_l2": l2,
                **metrics,
            }
        )

    record()
    try:
        while calls < config["gradient_budget"]:
            points = sampler.batch(0 if config["mode"] == "deterministic" else steps)
            current_loss, gradients, old = gradient_statistics(
                model,
                points,
                problem,
                fisher=config["method"] == "kfac",
                generator=generator,
                fisher_dtype=fisher_dtype,
            )
            calls += 1
            if config["method"] == "kfac":
                calls += 1
                auxiliary += 1
            if not math.isfinite(current_loss) or current_loss > 1e30:
                raise FloatingPointError("nonfinite/explosive training loss")
            optimizer.lr = config["learning_rate"] * lr_multiplier(config, steps)
            optimizer.apply(gradients, old)
            steps += 1
            if config["method"] == "kbfgs_l":
                _, _, new = gradient_statistics(model, points, problem, fisher=False)
                calls += 1
                auxiliary += 1
                optimizer.update_pairs(old, new, stochastic=config["mode"] == "stochastic")
            if calls % config["evaluation_interval"] == 0 or calls == config["gradient_budget"]:
                record()
            if steps % 1000 == 0:
                print(
                    {
                        "steps": steps,
                        "gradients": calls,
                        "loss": current_loss,
                        "seconds": time.perf_counter() - started,
                    },
                    flush=True,
                )
        if calls != config["gradient_budget"]:
            raise RuntimeError("gradient budget mismatch")
    except (FloatingPointError, RuntimeError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["gradient_evaluations"] != calls:
        record()
    if device.type == "cuda":
        torch.cuda.synchronize()
    score = sum((p["training_loss"] for p in history[-5:])) / len(history[-5:])
    return {
        "config": config,
        "problem": asdict(problem),
        "status": status,
        "reason": reason,
        "method": config["method"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "selection_score": score if status == "complete" and math.isfinite(score) else None,
        "training_loss": history[-1]["training_loss"],
        "relative_l2": history[-1]["relative_l2"],
        "gradient_evaluations": calls,
        "auxiliary_evaluations": auxiliary,
        "parameter_steps": steps,
        "wall_seconds": time.perf_counter() - started,
        "optimizer_state_bytes": optimizer.state_bytes(),
        "rejected_pairs": optimizer.rejected_pairs,
        "resolved_momentum": optimizer.momentum,
        "compact_fallbacks": sum((s.get("compact_fallbacks", 0) for s in optimizer.state)),
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "network_dtype": config["dtype"],
        "kfac_fisher_dtype": config.get("kfac_fisher_dtype", config["dtype"])
        if config["method"] == "kfac"
        else None,
        "variant": "Taylor-jet KFAC-expand, sum-Kronecker direct solve, fixed LR"
        if config["method"] == "kfac"
        else "Taylor-jet K-BFGS(L), shared-weight mean-pair extension",
        "matrix_routing": "all affine weights and homogeneous biases, no Adam fallback",
        "history": history,
    }
