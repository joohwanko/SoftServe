"""Download, verify, and parse the raw MNIST IDX files without torchvision."""

from __future__ import annotations
import gzip
import hashlib
import json
import os
import struct
import tempfile
import urllib.request
from pathlib import Path
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
DEFAULT_ROOT = Path("data/mnist")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(root: Path, name: str, expected: str) -> Path:
    destination = root / "raw" / name
    if destination.exists():
        observed = _sha256(destination)
        if observed != expected:
            raise RuntimeError(f"existing {name} has SHA-256 {observed}, expected {expected}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", temporary)
        observed = _sha256(temporary)
        if observed != expected:
            raise RuntimeError(f"downloaded {name} has SHA-256 {observed}, expected {expected}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _images(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        payload = handle.read()
    if (magic, count, rows, columns) != (2051, expected_count, 28, 28):
        raise ValueError(f"unexpected image IDX header in {path.name}")
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != count * rows * columns:
        raise ValueError(f"truncated image IDX payload in {path.name}")
    return values.reshape(count, rows * columns).copy()


def _labels(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        payload = handle.read()
    if (magic, count) != (2049, expected_count):
        raise ValueError(f"unexpected label IDX header in {path.name}")
    values = np.frombuffer(payload, dtype=np.uint8).copy()
    if values.shape != (count,) or int(values.min()) != 0 or int(values.max()) != 9:
        raise ValueError(f"invalid label IDX payload in {path.name}")
    return values


def prepare(root: Path = DEFAULT_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = {name: _download(root, name, digest) for name, digest in FILES.items()}
    train_images = _images(raw["train-images-idx3-ubyte.gz"], 60000)
    train_labels = _labels(raw["train-labels-idx1-ubyte.gz"], 60000)
    test_images = _images(raw["t10k-images-idx3-ubyte.gz"], 10000)
    test_labels = _labels(raw["t10k-labels-idx1-ubyte.gz"], 10000)
    destination = root / "mnist.npz"
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(dir=root, suffix=".npz")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(
            temporary,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        "format": "softserve-deep-autoencoder-data-v1",
        "mnist": {
            "archive": destination.name,
            "archive_sha256": _sha256(destination),
            "train_examples": 60000,
            "test_examples": 10000,
            "shape": [784],
            "storage_dtype": "uint8",
            "training_dtype": "float32",
            "preprocessing": "flatten row-major and divide raw pixels by 255",
            "source_urls": {name: f"{BASE_URL}/{name}" for name in FILES},
            "source_sha256": FILES,
        },
    }
    metadata_path = root / "metadata.json"
    descriptor = os.open(metadata_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 436)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"archive": str(destination), "sha256": metadata["mnist"]["archive_sha256"]}))
    return destination
