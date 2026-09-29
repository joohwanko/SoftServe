"""Compact, crash-safe storage for sweep workers.

One Slurm array task should execute several configurations and commit one shard. This
avoids creating a directory and many tiny files for every run.
"""

from __future__ import annotations
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def results_root() -> Path:
    """Return the configured result root, creating it when needed."""
    root = Path(os.environ.get("SOFTSERVE_RESULTS_ROOT", "results")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


class CampaignStore:
    """Write manifests and one JSONL shard per worker using atomic replacement."""

    def __init__(self, campaign: str, root: Path | None = None) -> None:
        self.path = (root or results_root()) / campaign
        self.shards = self.path / "shards"
        self.checkpoints = self.path / "checkpoints"
        self.shards.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
            for row in rows:
                line = json.dumps(dict(row), default=_jsonable, sort_keys=True)
                handle.write(line + "\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        path = self.path / "manifest.json"
        payload = json.dumps(manifest, default=_jsonable, indent=2, sort_keys=True)
        self._atomic_text(path, payload + "\n")
        return path

    def write_shard(self, worker_id: str | int, rows: Iterable[Mapping[str, Any]]) -> Path:
        """Atomically write all rows produced by one worker."""
        path = self.shards / f"part-{str(worker_id).zfill(5)}.jsonl"
        self._atomic_rows(path, rows)
        return path

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        for shard in sorted(self.shards.glob("part-*.jsonl")):
            with shard.open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def consolidate(self) -> Path:
        """Create one deterministic JSONL table from all completed shards."""
        path = self.path / "runs.jsonl"
        self._atomic_rows(path, self.iter_rows())
        return path
