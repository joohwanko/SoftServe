"""Deep fully-connected autoencoder benchmarks used by optimizer experiments."""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

MNIST_WIDTHS = (784, 1000, 500, 250, 30, 250, 500, 1000, 784)
MNIST_TRAIN_EXAMPLES = 60000
MNIST_TEST_EXAMPLES = 10000
MNIST_PARAMETER_COUNT = 2837314


@dataclass(frozen=True)
class AutoencoderData:
    train: torch.Tensor
    test: torch.Tensor | None


class DeepAutoencoder(nn.Module):
    """The MNIST architecture used by Goldfarb, Ren, and Bahamou (2020)."""

    def __init__(self, widths: tuple[int, ...] = MNIST_WIDTHS) -> None:
        super().__init__()
        self.widths = widths
        self.layers = nn.ModuleList(
            (
                nn.Linear(input_width, output_width)
                for input_width, output_width in zip(widths[:-1], widths[1:], strict=True)
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return output logits; BCE-with-logits supplies the sigmoid output."""
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index not in {3, len(self.layers) - 1}:
                values = torch.relu(values)
        return values


def build_model(seed: int, device: torch.device) -> DeepAutoencoder:
    """Build with the seeded default ``torch.nn.Linear`` initialization."""
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = DeepAutoencoder()
    model = model.to(device=device, dtype=torch.float32)
    if sum((parameter.numel() for parameter in model.parameters())) != MNIST_PARAMETER_COUNT:
        raise RuntimeError("unexpected MNIST autoencoder parameter count")
    return model


def matrix_routes(model: DeepAutoencoder) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    matrices = [layer.weight for layer in model.layers]
    selected = {id(parameter) for parameter in matrices}
    fallback = [parameter for parameter in model.parameters() if id(parameter) not in selected]
    if len(matrices) != 8 or len(fallback) != 8:
        raise RuntimeError("unexpected MNIST autoencoder parameter routing")
    return (matrices, fallback)


def load_data(path: str | Path, device: torch.device, *, include_test: bool) -> AutoencoderData:
    source = Path(path)
    metadata = json.loads(source.with_name("metadata.json").read_text())
    if metadata.get("format") != "softserve-deep-autoencoder-data-v1":
        raise ValueError("unexpected deep-autoencoder data format")
    if metadata.get("mnist", {}).get("archive") != source.name:
        raise ValueError("metadata does not identify this MNIST archive")
    with np.load(source, allow_pickle=False) as archive:
        train_array = archive["train_images"]
        if train_array.shape != (MNIST_TRAIN_EXAMPLES, MNIST_WIDTHS[0]):
            raise ValueError(f"unexpected MNIST train shape: {train_array.shape}")
        if train_array.dtype != np.uint8:
            raise ValueError("MNIST images must be stored as uint8")
        train = torch.as_tensor(train_array, device=device).to(torch.float32).div_(255.0)
        if include_test:
            test_array = archive["test_images"]
            if test_array.shape != (MNIST_TEST_EXAMPLES, MNIST_WIDTHS[0]):
                raise ValueError(f"unexpected MNIST test shape: {test_array.shape}")
            test = torch.as_tensor(test_array, device=device).to(torch.float32).div_(255.0)
        else:
            test = None
    if float(train.min()) < 0.0 or float(train.max()) > 1.0:
        raise ValueError("MNIST inputs must lie in [0,1]")
    return AutoencoderData(train=train, test=test)


def stateless_epoch_permutation(seed: int, epoch: int, population: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2006068877, seed, epoch])))
    return rng.permutation(population).astype(np.int64)


def stateless_epoch_indices(seed: int, step: int, batch_size: int, population: int) -> np.ndarray:
    batches_per_epoch = math.ceil(population / batch_size)
    epoch, batch = divmod(step, batches_per_epoch)
    permutation = stateless_epoch_permutation(seed, epoch, population)
    start = batch * batch_size
    return permutation[start : min(start + batch_size, population)]
