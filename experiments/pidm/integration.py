"""Task-local optimizer routing and exact RNG replay; no upstream model edits."""

from contextlib import contextmanager
from itertools import product
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
from softserve.batched import BatchedKron, BatchedMuon


def rng_snapshot():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def rng_restore(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@contextmanager
def replay_rng(state):
    current = rng_snapshot()
    rng_restore(state)
    try:
        yield
    finally:
        rng_restore(current)


class Routes:
    """Conv channel matrices per kernel offset, split into <=256-by-256 blocks.

    This is the existing PDE campaign's kernel-offset convention, extended to
    Conv1d/2d/3d and bounded blocks. Proxies share the original tensor storage.
    Routing is by module role, never just ndim: broadcast LayerNorm gains are
    NOT convolution weights. SOAP and Kron get exactly the same partitions.
    """

    def __init__(self, model, block=256, active_only=False):
        self.matrices, self.fallback, self.entries = ([], [], [])
        modules = dict(model.named_modules())
        self.inactive = []
        self.active_names = set()
        conv_types = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        self.description = {"block_size": block, "parameters": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if active_only and parameter.grad is None:
                self.inactive.append(name)
                continue
            self.active_names.add(name)
            module_name, _, role = name.rpartition(".")
            module = modules[module_name]
            is_matrix = role == "weight" and isinstance(module, (nn.Linear, *conv_types))
            if not is_matrix:
                self.fallback.append(parameter)
                self.description["parameters"].append(
                    {"name": name, "shape": list(parameter.shape), "route": "adam"}
                )
                continue
            offsets = product(*(range(k) for k in parameter.shape[2:]))
            count = 0
            for offset in offsets:
                for i in range(0, parameter.shape[0], block):
                    for j in range(0, parameter.shape[1], block):
                        key = (slice(i, i + block), slice(j, j + block), *offset)
                        value = parameter[key]
                        proxy = nn.Parameter(value.detach())
                        assert proxy.ndim == 2
                        assert (
                            proxy.untyped_storage().data_ptr()
                            == parameter.untyped_storage().data_ptr()
                        )
                        self.entries.append((name, parameter, key, proxy))
                        self.matrices.append(proxy)
                        count += 1
            self.description["parameters"].append(
                {"name": name, "shape": list(parameter.shape), "route": "matrix", "blocks": count}
            )
        self.description.update(
            matrix_blocks=len(self.matrices),
            matrix_parameters=sum((p.numel() for p in self.matrices)),
            adam_parameters=sum((p.numel() for p in self.fallback)),
            inactive_names=self.inactive,
        )
        expected = sum((p.numel() for n, p in model.named_parameters() if n in self.active_names))
        assert (
            expected == self.description["matrix_parameters"] + self.description["adam_parameters"]
        )

    def grads(self):
        for name, parameter, key, proxy in self.entries:
            if parameter.grad is None:
                raise RuntimeError(f"Previously active routed parameter has no gradient: {name}")
            proxy.grad = parameter.grad[key].detach()

    def check_active(self, model):
        newly_active = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and n not in self.active_names
        ]
        if newly_active:
            raise RuntimeError(f"Parameter activity changed: {newly_active}")


class Controller:
    def __init__(self, model, method, lr, block=256, active_only=True):
        self.route = Routes(model, block, active_only)
        self.qn = None
        if method == "adam":
            self.main = torch.optim.Adam(
                [p for name, p in model.named_parameters() if name in self.route.active_names],
                lr=lr,
                betas=(0.9, 0.999),
                foreach=True,
            )
        elif method == "muon":
            self.main = BatchedMuon(self.route.matrices, lr=lr)
        elif method == "kron":
            self.qn = BatchedKron(
                self.route.matrices,
                lr=lr,
                lam=9,
                beta1=0.9,
                beta_sy=0.0,
                beta_h=0.0,
                T=1,
                nesterov=False,
                backend="gemm",
                root_steps=18,
                inverse_steps=10,
                normalize=True,
                gauge="balanced_trace",
                bucket_chunk_size=16,
                constrained_update=True,
                pair_diagnostics=False,
                lambda_schedule="fixed",
            )
            self.main = self.qn
        elif method == "soap":
            from distributed_shampoo import (
                DistributedShampoo,
                DefaultSOAPConfig,
                WeightDecayType,
                SingleDeviceDistributedConfig,
            )

            self.main = DistributedShampoo(
                self.route.matrices,
                lr=lr,
                betas=(0.9, 0.999),
                epsilon=1e-12,
                weight_decay=0.0,
                weight_decay_type=WeightDecayType.DECOUPLED,
                max_preconditioner_dim=block,
                precondition_frequency=10,
                start_preconditioning_step=10,
                preconditioner_config=DefaultSOAPConfig,
                distributed_config=SingleDeviceDistributedConfig(target_parameter_dimensionality=2),
            )
        else:
            raise ValueError(method)
        self.fallback = (
            torch.optim.Adam(self.route.fallback, lr=0.0001, betas=(0.9, 0.999), foreach=True)
            if self.route.fallback and method != "adam"
            else None
        )

    def step(self):
        self.route.grads()
        if self.qn is None:
            self.main.step()
        else:
            self.qn.parameter_step()
        if self.fallback is not None:
            self.fallback.step()


def mechanics_paths(data_root):
    root = Path(data_root).resolve()
    paths = {
        "train": root / "train/fields",
        "validation": root / "test/valid/fields",
        "stiffness": root / "solidspy_k_no_BC",
    }
    missing = [str(p) for p in paths.values() if not p.is_dir()]
    missing += [
        str(paths["stiffness"] / name)
        for name in ("nodes.txt", "mater.txt", "eles.txt", "loads.txt")
        if not (paths["stiffness"] / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Official PIDM mechanics data unavailable: " + ", ".join(missing))
    for key in ("train", "validation"):
        if not any(paths[key].rglob("*.npy")):
            raise FileNotFoundError(f"No official field files in {paths[key]}")
    return paths
