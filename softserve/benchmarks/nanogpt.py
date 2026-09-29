"""Compact GPT-2 model and memory-mapped token loader."""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    layers: int = 12
    heads: int = 12
    width: int = 768
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Attention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.heads, self.width, self.dropout = (config.heads, config.width, config.dropout)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.projection = nn.Linear(config.width, config.width, bias=config.bias)

    def forward(self, values: Tensor) -> Tensor:
        batch, length, width = values.shape
        q, k, v = self.qkv(values).split(self.width, dim=-1)
        shape = (batch, length, self.heads, width // self.heads)
        q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.projection(output.transpose(1, 2).contiguous().view(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.width, 4 * config.width, bias=config.bias)
        self.down = nn.Linear(4 * config.width, config.width, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(values), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width, bias=config.bias)
        self.attention = Attention(config)
        self.norm_mlp = nn.LayerNorm(config.width, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class NanoGPT(nn.Module):
    """GPT-2 small (124.5M parameters) with nanoGPT initialization by default."""

    def __init__(self, config: GPTConfig = GPTConfig()) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.block_size, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList((Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width, bias=config.bias)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        for name, parameter in self.named_parameters():
            if name.endswith(("attention.projection.weight", "mlp.down.weight")):
                nn.init.normal_(parameter, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, tokens: Tensor, targets: Tensor | None = None, *, loss_reduction: str = "mean"
    ) -> tuple[Tensor, Tensor | None]:
        _, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError("sequence exceeds block_size")
        positions = torch.arange(length, device=tokens.device)
        values = self.dropout(self.token(tokens) + self.position(positions))
        for block in self.blocks:
            values = block(values)
        logits = self.head(self.norm(values))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction=loss_reduction)
        )
        if loss is not None and loss_reduction == "none":
            loss = loss.view_as(targets)
        return (logits, loss)


def matrix_routes(model: NanoGPT) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Route attention/MLP matrices to SoftSERVE and everything else to AdamW."""
    suffixes = ("qkv.weight", "projection.weight", "up.weight", "down.weight")
    matrices, fallback = ([], [])
    for name, parameter in model.named_parameters():
        (matrices if name.endswith(suffixes) else fallback).append(parameter)
    return (matrices, fallback)


class TokenFile:
    """Stable next-token batches from a flat uint16 token file."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.tokens = np.memmap(Path(path), dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError("token file is too short")

    def batch(
        self,
        starts: Tensor | np.ndarray | list[int],
        *,
        device: torch.device | str,
        length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        starts = np.asarray(starts, dtype=np.int64)
        length = self.block_size if length is None else length
        if starts.ndim != 1 or not len(starts) or length < 1 or (length > self.block_size):
            raise ValueError("invalid token spans")
        if starts.min() < 0 or starts.max() + length >= len(self.tokens):
            raise IndexError("token span is outside the file")
        offsets = np.arange(length + 1)
        windows = np.asarray(self.tokens[starts[:, None] + offsets], dtype=np.int64)
        values = torch.from_numpy(windows).to(device, non_blocking=True)
        return (values[:, :-1], values[:, 1:])
