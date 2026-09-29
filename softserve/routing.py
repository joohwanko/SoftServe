"""Bounded, model-aware parameter routing for the public optimizer interface."""

from dataclasses import dataclass
from itertools import product

import torch
from torch import nn


@dataclass
class MatrixBlock:
    parameter: nn.Parameter
    index: tuple
    proxy: nn.Parameter

    def view(self, tensor):
        return tensor[self.index]


class ModelRoutes:
    def __init__(self, model, max_dim=256, diagonal=False):
        if not isinstance(max_dim, int) or isinstance(max_dim, bool) or max_dim < 1:
            raise ValueError("max_preconditioner_dim must be a positive integer")
        self.max_dim = max_dim
        self.model = model
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.named:
            raise ValueError("The model has no trainable parameters")
        self.parameters = [p for _, p in self.named]
        self._registered = {n: id(p) for n, p in model.named_parameters()}
        self._specs = {}
        storages = {}
        for name, p in self.named:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise ValueError("Initialize lazy layers before constructing SoftSERVE")
            if p.layout != torch.strided or p.dtype not in (torch.float32, torch.float64):
                raise ValueError(
                    "Model mode requires dense FP32/FP64 parameters; use autocast for mixed precision"
                )
            if p.device.type not in ("cpu", "cuda") or not p.numel():
                raise ValueError("Model mode requires nonempty CPU or CUDA parameters")
            storage = (p.device, p.untyped_storage().data_ptr())
            # cuDNN RNNs pack distinct weights into disjoint parts of one
            # storage. Allow that, but never update overlapping parameters twice.
            start = p.storage_offset() * p.element_size()
            stop = (
                start
                + (1 + sum((n - 1) * s for n, s in zip(p.shape, p.stride()))) * p.element_size()
            )
            occupied = storages.setdefault(storage, [])
            if any(start < b and a < stop for a, b in occupied):
                raise ValueError(
                    "Distinct Parameter objects with overlapping storage are unsupported; tie the same Parameter object instead"
                )
            occupied.append((start, stop))
            self._specs[id(p)] = self._signature(p)
        devices = {p.device for p in self.parameters}
        if len(devices) != 1:
            raise ValueError("Model mode currently supports one CPU or CUDA device")
        self.device = next(iter(devices))
        self.blocks, self.main, self.fallback, self.description = [], [], [], []
        roles = {}
        for module in model.modules():
            if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)) or type(
                module
            ).__module__.startswith("torch.distributed.fsdp"):
                raise ValueError(
                    "DataParallel/DDP/FSDP model mode is not supported yet; use explicit runner integration"
                )
            for role, parameter in module.named_parameters(recurse=False):
                roles.setdefault(id(parameter), []).append((module, role))
        convs = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        norms = (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.modules.batchnorm._BatchNorm,
            nn.modules.instancenorm._InstanceNorm,
            nn.RMSNorm,
        )
        for name, p in self.named:
            owners = roles.get(id(p), [])
            excluded = any(
                isinstance(m, (nn.Embedding, nn.EmbeddingBag, *norms)) or role == "bias"
                for m, role in owners
            )
            conv = any(isinstance(m, convs) and role == "weight" for m, role in owners)
            matrix = not diagonal and not excluded and (p.ndim == 2 or conv)
            if not matrix:
                self.fallback.append(p)
                self.description.append(
                    dict(
                        name=name,
                        shape=tuple(p.shape),
                        route="diag" if diagonal else "fallback",
                        blocks=0,
                    )
                )
                continue
            self.main.append(p)
            start = len(self.blocks)
            # Each spatial kernel offset is a channel matrix, as in PIDM.
            for offset in product(*(range(k) for k in p.shape[2:])):
                for i in range(0, p.shape[0], max_dim):
                    for j in range(0, p.shape[1], max_dim):
                        index = (slice(i, i + max_dim), slice(j, j + max_dim), *offset)
                        proxy = nn.Parameter(p[index].detach())
                        self.blocks.append(MatrixBlock(p, index, proxy))
            self.description.append(
                dict(name=name, shape=tuple(p.shape), route="kron", blocks=len(self.blocks) - start)
            )

    @staticmethod
    def _signature(p):
        return (
            tuple(p.shape),
            p.stride(),
            p.storage_offset(),
            p.dtype,
            p.device,
            p.untyped_storage().data_ptr(),
        )

    def validate(self):
        if {n: id(p) for n, p in self.model.named_parameters()} != self._registered:
            raise RuntimeError("Model parameters changed; reconstruct the optimizer")
        allowed = set(self._specs)
        if any(p.requires_grad and id(p) not in allowed for p in self.model.parameters()):
            raise RuntimeError(
                "A previously excluded parameter was unfrozen; reconstruct the optimizer"
            )
        for _, p in self.named:
            if self._signature(p) != self._specs[id(p)]:
                raise RuntimeError(
                    "Parameter storage, shape, dtype, or device changed; move/initialize the model before constructing the optimizer"
                )

    def assign_proxy_gradients(self):
        for block in self.blocks:
            p = block.parameter
            block.proxy.grad = (
                block.view(p.grad).detach() if p.requires_grad and p.grad is not None else None
            )
