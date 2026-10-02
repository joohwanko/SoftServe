<div align="center">

# SoftServe 🍦

Official PyTorch implementation of **[SoftServe: A Scalable Quasi-Newton Method for Deep Learning](https://arxiv.org/abs/2610.02182)**.

[![arXiv](https://img.shields.io/badge/arXiv-2610.02182-b31b1b.svg)](https://arxiv.org/abs/2610.02182)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Install](#installation) · [Quick start](#quick-start) · [Examples](#examples) · [Experiments](#experiments) · [Citation](#citation)

</div>

SoftServe learns positive-definite, structured inverse-curvature estimates from secant pairs. This repository provides Kronecker (`SoftServeKron`) and diagonal (`SoftServeDiag`) optimizers, experiment runners, and selected paper configurations.

## Installation

Tested with Python 3.12 and PyTorch 2.13.0. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/joohwanko/SoftServe.git
cd SoftServe
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

<details>
<summary>Use pip instead</summary>

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

</details>

## Quick start

```python
from softserve import SoftServeKron

optimizer = SoftServeKron(
    model, lr=1e-3, lam=99, K=10,
    fallback="adamw", fallback_lr=3e-4,
)
```

Pass the **model**, after moving it to its training device. SoftServe routes matrix weights and convolution kernels to Kron, and biases, embeddings, normalization parameters, and other tensors to the fallback. Matrix factors are bounded to blocks of at most 256 × 256.

Choose `fallback="adam"`, `"adamw"`, or `"softserve-diag"` (default). `fallback_lr` defaults to `lr`. Diag shares Kron's λ, momentum, and refresh interval; Adam/AdamW use independent `fallback_betas=(0.9, 0.999)` and `fallback_weight_decay=0`. Set weight decay explicitly when wanted.

Defaults are `beta1=0.9`, no Nesterov, and constrained normalization. Kron uses Gram QME with 18 root and 10 inverse iterations. `SoftServeDiag(model, lr=..., lam=..., K=10)` applies the diagonal method to every trainable parameter.

## Examples

Both loops use this small regression problem. Run the setup afresh before either loop.

```python
import torch
from torch import nn
from softserve import SoftServeKron

torch.manual_seed(0)
x = torch.randn(256, 8)
y = x[:, :1] - 0.5 * x[:, 1:2]
model = nn.Sequential(
    nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1)
)
loss_fn = nn.MSELoss()
optimizer = SoftServeKron(
    model, lr=1e-2, lam=99, K=10,
    fallback="adam", fallback_lr=1e-3,
)
```

### Deterministic: fixed data

For a fixed, deterministic objective, `step()` reuses gradients from the training loop. Curvature is refreshed after each `K` updates, when the next gradient becomes available. No extra backward pass is needed. Use the closure version below if data, randomness, or forward-pass state changes.

```python
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

### Stochastic: new minibatches

Sample a minibatch **outside** the closure. Bind it in the closure's default arguments so it remains available for replay. `step(closure)` does the usual training backward pass and, every `K` updates, one extra backward pass on the interval's starting minibatch.

```python
for _ in range(100):
    indices = torch.randint(len(x), (64,))
    xb, yb = x[indices], y[indices]

    def closure(xb=xb, yb=yb):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        return loss

    optimizer.step(closure)
```

At `K=10`, 100 updates use 110 backward passes. Parameter/gradient snapshots and replay are handled internally. Replay restores the starting randomness and registered model buffers, including standard Dropout and BatchNorm, without advancing their current state a second time.

<details>
<summary>Checkpointing and supported usage</summary>

- Save stochastic checkpoints when `optimizer.checkpoint_ready` is true, normally after a multiple of `K` updates. `state_dict()` refuses to save an unfinished interval because a Python closure cannot be serialized. Deterministic checkpoints can be saved at any step. Save the model, scheduler, data position, and RNG state as usual for exact training resumption.
- Keep the captured batch and loss settings unchanged until replay finishes. Closures should only zero gradients, evaluate the loss, and call `backward()`; do not advance a data loader or scheduler inside them. Custom RNG generators or mutable state outside registered model buffers need explicit integration.
- Use `max_grad_norm=1.0` if clipping is needed; it clips updates' gradients while retaining raw gradients for secants. LR schedulers act on both parameter groups. `optimizer.routing` shows the parameter assignment.
- Supported: dense FP32/FP64 parameters on one CPU or CUDA device, including tied weights. Autocast is supported; GradScaler, sparse gradients, DataParallel/DDP/FSDP, and changing model parameters/devices after construction are not supported by this interface.
- The original `SoftServeKron(parameters, ...)`, `parameter_step()`, and `update_curvature()` interfaces remain unchanged. Paper runners retain their explicit, task-specific routing and replay; `K` belongs to the new model interface, not the legacy `T` option.

</details>

## Experiments

Install the experiment dependencies, list the selected configurations, and run one:

```bash
uv pip install -e '.[experiments,soap]'
python run.py --config configs/rnn.json --list
python run.py --config configs/rnn.json --index 0 --seed 1
```

Configs cover RNN Adding, MNIST, small PINNs, PIDM, and GPT; see [configs/](configs/). RNN runs on CPU; the other tasks use GPUs. PirateNet provides **configurations only**, pending redistribution permission ([details](THIRD_PARTY.md#piratenet-exclusion)).

Use `--lr` to tune on seed 0, then run the evaluation seeds listed in each config. Use `--set KEY=JSON` for other overrides. Results go to `outputs/`; keep sweeps and final runs in separate directories.

<details>
<summary>Data preparation and reproduction notes</summary>

RNN and small PINNs generate data locally. For the other tasks:

```bash
python prepare.py mnist
uv pip install -r requirements-pidm.txt
python prepare.py pidm       # ~5.3 GB download
python prepare.py fineweb
```

- FineWeb preparation processes 2.01B tokens to preserve the validation split, although training uses 200M. Compatible `train.bin` and `val.bin` files can instead be placed in `data/fineweb/`. The historical dataset revision was not recorded; use `--revision` to pin new downloads. Identical historical data is not guaranteed.
- The method name `sgd_m` denotes SoftServe-I, the normalized identity-metric control.
- PIDM runs 100k updates, but plots stop at 100k gradient evaluations, including replay.

</details>

### Plot results

```bash
python plot.py outputs --task rnn --metric validation_mse \
    --output outputs/rnn.pdf
python plot.py outputs --task rnn --mode lr --metric selection_score \
    --output outputs/rnn_lr.pdf
```

Curves show means and seed min–max ranges; failed cohorts remain individual traces. For the standalone QME ablation, run `python -m experiments.qme --output outputs/qme`. Other options are available through `--help`.

## Citation

```bibtex
@misc{ko2026softserve,
  title         = {{SoftServe}: A Scalable Quasi-Newton Method for Deep Learning},
  author        = {Joohwan Ko and Tetiana Parshakova and Diana Cai and Robert M. Gower},
  year          = {2026},
  eprint        = {2610.02182},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.02182}
}
```

## License and acknowledgments

SoftServe is released under the [MIT License](LICENSE). Third-party code retains its original licenses and copyright notices; see [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources and reproduction limitations.
