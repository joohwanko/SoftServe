"""Original stochastic Adding RNN with explicitly shared affine statistics."""

import math
import time
import torch
from torch.nn import functional as F
from experiments.rnn.runner import (
    AddingRNN,
    HIDDEN_SIZE,
    RNNAddingRunConfig,
    _mse,
    make_problem,
    training_batch,
)
from .core import MatrixBaseline, Statistics
from .compact_kbfgs import CompactKBFGSL


def observed_forward(model, inputs):
    hidden = inputs.new_zeros((len(inputs), HIDDEN_SIZE))
    ones = inputs.new_ones((len(inputs), 1))
    records = [[], []]
    for index in range(inputs.shape[1]):
        augmented = torch.cat((hidden, inputs[:, index], ones), dim=1)
        preactivation = F.linear(augmented, model.recurrent)
        records[0].append((augmented, preactivation))
        hidden = torch.tanh(preactivation)
    augmented = torch.cat((hidden, ones), dim=1)
    preactivation = F.linear(augmented, model.output)
    records[1].append((augmented, preactivation))
    return (preactivation.squeeze(1), records)


def gradient_statistics(model, batch, *, fisher):
    output, records = observed_forward(model, batch.inputs)
    preactivations = [z for group in records for _, z in group]
    loss = F.mse_loss(output, batch.targets)
    values = torch.autograd.grad(
        loss, (*model.matrix_parameters(), *preactivations), retain_graph=fisher
    )
    parameters, adjoints = (values[:2], values[2:])
    fisher_adjoints = (
        torch.autograd.grad(math.sqrt(2 / len(output)) * output.sum(), preactivations)
        if fisher
        else None
    )
    stats, offset = ([], 0)
    for group in records:
        count = len(group)
        inputs = torch.cat([a.detach() for a, _ in group], dim=0)
        z = torch.cat([z.detach() for _, z in group], dim=0)
        delta = torch.cat([g.detach() for g in adjoints[offset : offset + count]], dim=0)
        probe = (
            torch.cat([g.detach() for g in fisher_adjoints[offset : offset + count]], dim=0)
            if fisher
            else None
        )
        stats.append(
            Statistics(
                inputs.T @ inputs / len(inputs),
                inputs.mean(0),
                z.mean(0),
                delta.sum(0),
                None if probe is None else probe.T @ probe,
            )
        )
        offset += count
    return (float(loss.detach()), tuple((g.detach() for g in parameters)), stats)


def run(config):
    torch.set_num_threads(1)
    started = time.perf_counter()
    device = torch.device("cpu")
    problem = make_problem(config["seed"], device)
    model = AddingRNN(problem.recurrent, problem.output)
    sampling_config = RNNAddingRunConfig(
        method="adam", seed=config["seed"], learning_rate=config["learning_rate"]
    )
    optimizer = (CompactKBFGSL if config["method"] == "kbfgs_l" else MatrixBaseline)(
        model.matrix_parameters(),
        method=config["method"],
        lr=config["learning_rate"],
        damping=config["damping"],
        momentum=config.get("momentum", 0.9),
        factor_decay=config.get("factor_decay", 0.9),
        inverse_frequency=config.get("inverse_frequency", 20),
        history_size=config.get("history_size", 100),
    )
    history = []
    calls = steps = auxiliary = 0
    train_loss = None
    status, reason = ("complete", None)

    def record():
        validation = _mse(model, problem.validation)
        history.append(
            {
                "parameter_steps": steps,
                "gradient_evaluations": calls,
                "wall_seconds": time.perf_counter() - started,
                "training_batch_mse": train_loss,
                "validation_mse": validation,
            }
        )
        if calls % 1000 == 0:
            print(history[-1], flush=True)

    record()
    try:
        while calls < config["gradient_budget"]:
            batch = training_batch(sampling_config, steps, device)
            train_loss, gradients, old = gradient_statistics(
                model, batch, fisher=config["method"] == "kfac"
            )
            calls += 1
            if config["method"] == "kfac":
                calls += 1
                auxiliary += 1
            if not math.isfinite(train_loss) or train_loss > 1e20:
                raise FloatingPointError("nonfinite/explosive training loss")
            optimizer.apply(gradients, old)
            steps += 1
            if config["method"] == "kbfgs_l":
                _, _, new = gradient_statistics(model, batch, fisher=False)
                calls += 1
                auxiliary += 1
                optimizer.update_pairs(old, new, stochastic=True)
            if calls % config["evaluation_interval"] == 0 or calls == config["gradient_budget"]:
                record()
        if calls != config["gradient_budget"]:
            raise RuntimeError("budget mismatch")
    except (FloatingPointError, RuntimeError) as error:
        status, reason = ("failed", repr(error))
    if history[-1]["gradient_evaluations"] != calls:
        record()
    test = _mse(model, problem.test) if config["selected"] and status == "complete" else None
    score = sum((p["validation_mse"] for p in history[-5:])) / len(history[-5:])
    return {
        "config": config,
        "status": status,
        "reason": reason,
        "method": config["method"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "selection_score": score if status == "complete" and math.isfinite(score) else None,
        "final_validation_mse": history[-1]["validation_mse"],
        "test_mse": test,
        "gradient_evaluations": calls,
        "auxiliary_evaluations": auxiliary,
        "parameter_steps": steps,
        "wall_seconds": time.perf_counter() - started,
        "optimizer_state_bytes": optimizer.state_bytes(),
        "rejected_pairs": optimizer.rejected_pairs,
        "resolved_momentum": optimizer.momentum,
        "compact_fallbacks": sum((s.get("compact_fallbacks", 0) for s in optimizer.state)),
        "factor_shapes": [[19, 19], [16, 16], [17, 17], [1, 1]],
        "variant": "KFAC-expand"
        if config["method"] == "kfac"
        else "K-BFGS(L) shared-weight mean-pair extension",
        "history": history,
    }
