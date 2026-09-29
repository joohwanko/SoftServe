"""Prepare one dataset. Data are downloaded only when this command is run."""

import argparse
import json
from pathlib import Path


def fineweb(root, train_tokens, validation_offset, val_tokens, revision):
    import numpy as np
    import tiktoken
    from datasets import load_dataset
    from tqdm import tqdm

    if not 1 < train_tokens <= validation_offset or val_tokens <= 0:
        raise ValueError(
            "Require 1 < training tokens <= validation offset and positive validation tokens"
        )
    root.mkdir(parents=True, exist_ok=True)
    if any(
        (root / name).exists() for name in ("train.bin", "val.bin", "train.partial", "val.partial")
    ):
        raise FileExistsError("Refusing to overwrite existing or partial token data")
    encoder = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision=revision,
    )
    position, total = 0, validation_offset + val_tokens
    with (root / "train.partial").open("wb") as train, (root / "val.partial").open("wb") as val:
        with tqdm(total=total, unit="tokens", unit_scale=True) as progress:
            for row in dataset:
                tokens = encoder.encode_ordinary(row["text"]) + [encoder.eot_token]
                end = position + len(tokens)
                for stream, lower, upper in (
                    (train, 0, train_tokens),
                    (val, validation_offset, total),
                ):
                    lo, hi = max(position, lower), min(end, upper)
                    if lo < hi:
                        np.asarray(tokens[lo - position : hi - position], dtype="<u2").tofile(
                            stream
                        )
                progress.update(min(end, total) - min(position, total))
                position = end
                if position >= total:
                    break
    if position < total:
        raise ValueError("Dataset ended before the requested validation range")
    (root / "train.partial").rename(root / "train.bin")
    (root / "val.partial").rename(root / "val.bin")
    metadata = dict(
        dataset="HuggingFaceFW/fineweb",
        subset="sample-10BT",
        revision=revision,
        tokenizer="gpt2",
        append_eos=True,
        dtype="uint16",
        train_tokens=train_tokens,
        validation_offset=validation_offset,
        val_tokens=val_tokens,
    )
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("mnist", "pidm", "fineweb"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--train-tokens", type=int, default=200_000_001)
    parser.add_argument("--validation-offset", type=int, default=2_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision")
    args = parser.parse_args()
    root = args.root or Path("data") / args.dataset
    if args.dataset == "mnist":
        from experiments.mnist.data import prepare

        prepare(root)
    elif args.dataset == "pidm":
        from experiments.pidm import data

        data.DATA = root
        data.ROOT = root
        data.prepare()
    else:
        fineweb(root, args.train_tokens, args.validation_offset, args.val_tokens, args.revision)


if __name__ == "__main__":
    main()
