# SoftSERVE

PyTorch implementation of **SoftSERVE (Secant Equation Regularized with Variational Entropy)**, with experiment runners and selected paper configurations.

## Paper

**SoftSERVE: A Scalable Quasi-Newton Method for Deep Learning**  
Joohwan Ko, Tetiana Parshakova, Diana Cai, and Robert Gower.

arXiv link and BibTeX citation: coming soon.

## Install

Tested with Python 3.12 and PyTorch 2.13.0.

```bash
pip install -e .
```

For experiments, use `pip install -e '.[experiments,soap]'`. PIDM also requires `pip install -r requirements-pidm.txt`.

## Use the optimizer

```python
from softserve import SoftServeKron

optimizer = SoftServeKron(matrix_parameters, lr=1e-3, lam=99, beta1=0.9)
```

Kron defaults to Gram QME with 18 root and 10 inverse iterations. Constrained normalization is on. `SoftServeDiag` is also available.

For deterministic losses, call `loss.backward()` followed by `optimizer.step()`. Minibatch training needs same-batch secants; the experiment runners below handle the required replay.

## Run an experiment

List the selected configurations, then run one:

```bash
python run.py --config configs/rnn.json --list
python run.py --config configs/rnn.json --index 0 --seed 1
```

Configs cover RNN Adding, MNIST, small PINNs, PIDM, and GPT; see [configs/](configs/). RNN uses CPU; the other tasks use GPUs. PirateNet provides configurations only, pending redistribution permission ([details](THIRD_PARTY.md#piratenet-exclusion)).

Use `--lr` to tune on seed 0, then evaluate the selected configuration on seeds 1–3. Use `--set KEY=JSON` for other overrides. Results go to `outputs/`. The method name `sgd_m` denotes SoftServe-I, the normalized identity-metric control.

RNN and small PINNs generate data locally. Prepare other datasets as needed:

```bash
python prepare.py mnist
python prepare.py pidm       # ~5.3 GB download
python prepare.py fineweb
```

FineWeb preparation processes 2.01B tokens to preserve the validation split, although training uses 200M. Compatible `train.bin` and `val.bin` files can instead be placed in `data/fineweb/`. The historical dataset revision was not recorded; use `--revision` to pin new downloads. Identical historical data is not guaranteed.

## Plot results

```bash
python plot.py outputs --task rnn --metric validation_mse \
    --output outputs/rnn.pdf
python plot.py outputs --task rnn --mode lr --metric selection_score \
    --output outputs/rnn_lr.pdf
```

Curves show means and seed min–max ranges; failed cohorts remain individual traces. Keep sweeps and final runs in separate output directories. PIDM runs 100k updates, but plots stop at 100k gradient evaluations, including replay.

For the standalone QME ablation, run `python -m experiments.qme --output outputs/qme`. Other command options are available through `--help`.

## Third-party code

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream sources, retained licenses, and reproduction limitations.
