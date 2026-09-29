"""Native PIDM mechanics objective with matched SOAP/Kron optimizer adapters."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import yaml
from experiments.pidm.integration import ROOT, Controller, mechanics_paths, rng_snapshot, replay_rng
from softserve.secants import IntervalSecant
from experiments.pidm.upstream.unet_model import Unet3D
from experiments.pidm.upstream.data_utils import Dataset_Paths
from experiments.pidm.upstream.denoising_utils import DenoisingDiffusion, EMA
from experiments.pidm.upstream.residuals_mechanics_K import ResidualsMechanics


@dataclass(frozen=True)
class Config:
    method: str
    lr: float
    seed: int = 0
    updates: int = 30000
    phase: str = "sweep"
    batch: int = 4
    block: int = 256
    eval_every: int = 500
    train_probe: int = 32
    validation_probe: int = 128
    data_root: str = "data/pidm/mechanics"
    campaign: str = "legacy"

    def identity(self):
        value = asdict(self)
        if self.campaign == "legacy":
            value.pop("campaign")
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

    def validate(self):
        if self.method not in ("soap", "kron", "muon", "adam") or self.lr <= 0:
            raise ValueError("Invalid method or learning rate")
        if self.block != 256 or self.batch < 1 or self.updates < 1:
            raise ValueError("Use positive batch/budget and the paper's 256-wide blocks")

    def output_root(self):
        return Path(os.environ.get("SOFTSERVE_OUTPUT_DIR", "outputs/pidm"))


def append(path, value):
    with path.open("a") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RecordingDiffusion(DenoisingDiffusion):
    """Observe the unchanged upstream loss, including its original broadcasting."""

    def gaussian_log_likelihood(self, *args, **kwargs):
        value = super().gaussian_log_likelihood(*args, **kwargs)
        self.likelihood_means.append(value.detach().mean())
        return value

    def objective(self, batch, residual, weights):
        self.likelihood_means = []
        loss, data, residual_mae, volume_signed, compliance = self.model_estimation_loss(
            batch, residual_func=residual, **weights
        )
        parts = {
            "objective": float(loss.detach()),
            "data_weighted": data,
            "equilibrium_weighted": float(-weights["c_residual"] * self.likelihood_means[0]),
            "volume_weighted": float(-weights["c_ineq"] * self.likelihood_means[1]),
            "compliance_weighted": weights["lambda_opt"] * compliance,
            "equilibrium_mae": residual_mae,
            "volume_signed": volume_signed,
            "compliance": compliance,
        }
        total = sum(
            (
                parts[k]
                for k in (
                    "data_weighted",
                    "equilibrium_weighted",
                    "volume_weighted",
                    "compliance_weighted",
                )
            )
        )
        if math.isfinite(total) and (
            not math.isclose(total, parts["objective"], rel_tol=2e-05, abs_tol=1e-05)
        ):
            raise AssertionError(f"Loss component accounting differs: {total}, {parts}")
        return (loss, parts)


def fixed_probe(model, diffusion, residual, dataset, count, weights, seed, batch_size=4):
    mode = model.training
    with replay_rng(rng_snapshot()), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        model.eval()
        indices = np.random.default_rng(seed).choice(
            len(dataset), size=min(count, len(dataset)), replace=False
        )
        rows, sizes = ([], [])
        for begin in range(0, len(indices), batch_size):
            ids = indices[begin : begin + batch_size]
            batch = torch.stack([dataset[int(i)] for i in ids]).cuda()
            _, values = diffusion.objective(batch, residual, weights)
            rows.append(values)
            sizes.append(len(ids))
        output = {
            key: float(np.average([row[key] for row in rows], weights=sizes)) for key in rows[0]
        }
    model.train(mode)
    return output


def generated_design_probe(model, diffusion, residual, dataset):
    """Eight fixed validation designs, native sample.py inference (not LR selection).

    Call the unchanged p_sample step directly: upstream p_sample_loop assumes
    saved intermediate images even when save_output=False. We need no such files.
    """
    previous_ddim = residual.use_ddim_x0
    previous_mode = model.training
    summaries = []
    images = []
    try:
        residual.use_ddim_x0 = False
        model.eval()
        with replay_rng(rng_snapshot()), torch.no_grad():
            torch.manual_seed(16180)
            torch.cuda.manual_seed_all(16180)
            for begin in (0, 4):
                data = torch.stack([dataset[i] for i in range(begin, begin + 4)]).cuda()
                conditioning, target, bcs = torch.tensor_split(data, (3, 6), dim=1)
                current = torch.randn((4, 3, 65, 65), device="cuda")
                for t in reversed(range(diffusion.n_steps)):
                    (current, _), stats = diffusion.p_sample(
                        current,
                        (conditioning, bcs, target),
                        t,
                        save_output=False,
                        surpress_noise=True,
                        use_dynamic_threshold=False,
                        residual_func=residual,
                        eval_residuals=True,
                        return_optimizer=True,
                        return_inequality=True,
                        residual_correction=False,
                        correction_mode="none",
                    )
                summaries.append(
                    {
                        "equilibrium_mae": float(stats["residual"].abs().mean()),
                        "relative_compliance_error": float(stats["rel_CE_error_full_batch"].mean()),
                        "relative_volume_error": float(stats["vf_error_full_batch"].mean()),
                        "floating_material_fraction": float(
                            stats["fm_error_full_batch"].float().mean()
                        ),
                    }
                )
                images.append(current.cpu().numpy())
        result = {k: float(np.mean([r[k] for r in summaries])) for k in summaries[0]}
        if not all((math.isfinite(value) for value in result.values())):
            raise FloatingPointError(f"Nonfinite generated-design metrics: {result}")
        return (result, np.concatenate(images))
    finally:
        residual.use_ddim_x0 = previous_ddim
        model.train(previous_mode)


def run(cfg):
    cfg.validate()
    paths = mechanics_paths(cfg.data_root)
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    gpu = torch.cuda.get_device_name()
    run_id = cfg.identity()
    run_root = cfg.output_root()
    output = run_root / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "progress.jsonl").exists():
        raise RuntimeError(f"Refusing duplicate run: {run_id}")
    atomic_json(output / "config.json", asdict(cfg))
    upstream = yaml.safe_load((ROOT / "released_mechanics.yaml").read_text())
    weights = {k: upstream[k] for k in ("c_data", "c_residual", "c_ineq", "lambda_opt")}
    started = time.perf_counter()
    eval_seconds = 0.0
    optimizer_seconds = 0.0
    checkpoint_seconds = 0.0
    grads = 0
    endpoints = 0
    step = 0
    status = "failed"
    reason = None
    last = None
    qme_max = 0.0
    qme_nonfinite = 0
    torch.cuda.reset_peak_memory_stats()
    model = None
    controller = None
    try:
        train = Dataset_Paths(str(paths["train"]))
        validation = Dataset_Paths(str(paths["validation"]))
        if len(train) < 1000 or len(validation) < cfg.validation_probe:
            raise RuntimeError(
                f"Incomplete official data: train={len(train)}, validation={len(validation)}"
            )
        model = Unet3D(dim=128, channels=10, out_dim=3, sigmoid_last_channel=True).cuda()
        parameter_hash(model)
        residual = ResidualsMechanics(
            model,
            pixels_per_dim=64,
            pixels_at_boundary=True,
            no_BC_folder=str(paths["stiffness"]) + "/",
            device="cuda",
            use_ddim_x0=True,
            ddim_steps=upstream["ddim_steps"],
            topopt_eval=True,
        )
        diffusion = RecordingDiffusion(upstream["diff_steps"], device="cuda")
        ema = EMA(0.99)
        ema.register(model)
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        batches = iter(loader)
        provenance = dict(
            parameters=sum((p.numel() for p in model.parameters())),
            gpu=gpu,
            torch=torch.__version__,
            precision="fp32_no_tf32",
            physics_config=upstream,
        )
        atomic_json(output / "provenance.json", provenance)
        interval = IntervalSecant(10) if cfg.method == "kron" else None
        replay = None
        window = []
        replay_checked = False

        def record():
            nonlocal eval_seconds, last
            torch.cuda.synchronize()
            before = time.perf_counter()
            fixed_train = fixed_probe(
                model, diffusion, residual, train, cfg.train_probe, weights, 31415, cfg.batch
            )
            ema.ema(model)
            try:
                fixed_validation = fixed_probe(
                    model,
                    diffusion,
                    residual,
                    validation,
                    cfg.validation_probe,
                    weights,
                    27182,
                    cfg.batch,
                )
            finally:
                ema.restore(model)
            torch.cuda.synchronize()
            eval_seconds += time.perf_counter() - before
            last = dict(
                event="evaluation",
                step=step,
                gradient_evaluations=grads,
                endpoint_evaluations=endpoints,
                train_raw=fixed_train,
                validation_ema=fixed_validation,
                stochastic_training_mean=float(np.mean(window)) if window else None,
                wall_seconds=time.perf_counter() - started,
                training_seconds=time.perf_counter() - started - eval_seconds,
                optimizer_seconds=optimizer_seconds,
                qme_max_residual=qme_max,
                qme_nonfinite=qme_nonfinite,
                skipped_metric_blocks=controller.qn.skipped if controller and controller.qn else 0,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                samples_seen=step * cfg.batch,
            )
            append(output / "progress.jsonl", last)
            print(json.dumps(last, allow_nan=False), flush=True)
            window.clear()

        record()
        while step < cfg.updates:
            try:
                batch = next(batches).cuda(non_blocking=True)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches).cuda(non_blocking=True)
            model.train()
            model.zero_grad(set_to_none=True)
            random_state = rng_snapshot() if interval and interval.starts_at(step) else None
            loss, parts = diffusion.objective(batch, residual, weights)
            if not math.isfinite(parts["objective"]):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            grads += 1
            if controller is None:
                controller = Controller(model, cfg.method, cfg.lr, block=cfg.block)
                atomic_json(output / "routing.json", controller.route.description)
            controller.route.check_active(model)
            if interval and interval.starts_at(step):
                controller.route.grads()
                interval.start(
                    controller.route.matrices,
                    [p.grad for p in controller.route.matrices],
                    batch_key=step,
                )
                replay = (batch.detach().clone(), random_state)
                if not replay_checked:
                    expected = [p.grad.detach().clone() for p in controller.route.matrices]
                    model.zero_grad(set_to_none=True)
                    with replay_rng(random_state):
                        repeated, repeated_parts = diffusion.objective(batch, residual, weights)
                        repeated.backward()
                    grads += 1
                    controller.route.grads()
                    delta2 = sum(
                        (
                            (p.grad - g).square().sum()
                            for p, g in zip(controller.route.matrices, expected)
                        )
                    )
                    reference2 = sum((g.square().sum() for g in expected))
                    relative = float((delta2 / reference2.clamp_min(1e-30)).sqrt())
                    if relative > 1e-05:
                        raise AssertionError(
                            f"Same-realization gradient replay mismatch {relative}"
                        )
                    atomic_json(
                        output / "replay_test.json",
                        dict(
                            relative_gradient_error=relative,
                            loss_before=parts["objective"],
                            loss_replayed=repeated_parts["objective"],
                            counted_diagnostic_backwards=1,
                        ),
                    )
                    replay_checked = True
                    del expected
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize()
            before = time.perf_counter()
            controller.step()
            torch.cuda.synchronize()
            optimizer_seconds += time.perf_counter() - before
            step += 1
            if interval and interval.ends_at(step - 1) and (step < cfg.updates):
                model.zero_grad(set_to_none=True)
                with replay_rng(replay[1]):
                    endpoint_loss, _ = diffusion.objective(replay[0], residual, weights)
                    if not torch.isfinite(endpoint_loss):
                        raise FloatingPointError("Nonfinite endpoint objective")
                    endpoint_loss.backward()
                grads += 1
                endpoints += 1
                controller.route.grads()
                s, y = interval.pair(
                    controller.route.matrices, [p.grad for p in controller.route.matrices]
                )
                torch.cuda.synchronize()
                before = time.perf_counter()
                controller.qn.update_curvature(s, y)
                torch.cuda.synchronize()
                optimizer_seconds += time.perf_counter() - before
                residuals = controller.qn.last_qme_residual_A + controller.qn.last_qme_residual_G
                values = (
                    torch.stack(residuals).detach() if residuals else torch.zeros(1, device="cuda")
                )
                invalid = int((~torch.isfinite(values)).sum())
                qme_nonfinite += invalid
                finite_values = values[torch.isfinite(values)]
                if finite_values.numel():
                    qme_max = max(qme_max, float(finite_values.max()))
                append(
                    output / "qme.jsonl",
                    dict(
                        step=step,
                        nonfinite=invalid,
                        max_finite_residual=float(finite_values.max())
                        if finite_values.numel()
                        else None,
                    ),
                )
                del s, y
                if invalid:
                    raise FloatingPointError(
                        f"Nonfinite QME refresh diagnostics in {invalid} factor solves"
                    )
            if step - 1 > 1000:
                ema.update(model)
            window.append(parts["objective"])
            if step % 50 == 0:
                row = dict(
                    event="training",
                    step=step,
                    gradient_evaluations=grads,
                    training_seconds=time.perf_counter() - started - eval_seconds,
                    **parts,
                )
                append(output / "training.jsonl", row)
                atomic_json(
                    output / "heartbeat.json",
                    dict(
                        step=step,
                        gradient_evaluations=grads,
                        wall_seconds=time.perf_counter() - started,
                        training_seconds=time.perf_counter()
                        - started
                        - eval_seconds
                        - checkpoint_seconds,
                        peak_memory_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    ),
                )
                print(json.dumps(row, allow_nan=False), flush=True)
            if step % cfg.eval_every == 0 or step == cfg.updates:
                record()
        status = "complete"
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result = dict(
            run_id=run_id,
            status=status,
            reason=reason,
            config=asdict(cfg),
            completed_updates=step,
            gradient_evaluations=grads,
            endpoint_evaluations=endpoints,
            last=last,
            wall_seconds=time.perf_counter() - started,
            training_seconds=time.perf_counter() - started - eval_seconds - checkpoint_seconds,
            evaluation_seconds=eval_seconds,
            checkpoint_seconds=checkpoint_seconds,
            optimizer_seconds=optimizer_seconds,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            gpu=gpu,
            job_id=os.environ.get("SLURM_JOB_ID"),
            qme_nonfinite=qme_nonfinite,
            qme_max_residual=qme_max,
        )
        atomic_json(output / "result.json", result)
        append(run_root / "ledger.jsonl", result)
    if status != "complete":
        raise RuntimeError(reason)
