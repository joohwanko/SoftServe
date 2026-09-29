"""Download checksum-verified official data; extract only the mechanics task."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
DATA = Path("data/pidm")
URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/cebe5a39-f345-491e-bdf0-c99b918ddf41/content"
MD5 = "42bd0ca43c2e43c3ef5d5c6ef6c74b33"
SIZE = 5268056070


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "data.zip"
    if not archive.exists():
        partial = DATA / "data.zip.partial"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "30",
                "--user-agent",
                "Mozilla/5.0",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                URL,
            ],
            check=True,
        )
        assert partial.stat().st_size == SIZE
        partial.rename(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    assert digest == MD5, f"Official archive MD5 mismatch: {digest}"
    manifest = []
    with zipfile.ZipFile(archive) as zipped:
        entries = []
        for entry in zipped.infolist():
            relative = Path(entry.filename)
            if (
                not relative.parts
                or relative.parts[0] == "__MACOSX"
                or "mechanics" not in relative.parts
                or entry.is_dir()
            ):
                continue
            entries.append(entry)

        def extract(entry):
            relative = Path(entry.filename)
            tail = Path(*relative.parts[relative.parts.index("mechanics") :])
            destination = DATA / tail
            assert destination.resolve().is_relative_to(DATA.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size != entry.file_size:
                backup = destination.with_suffix(destination.suffix + ".interrupted")
                if backup.exists():
                    raise RuntimeError(f"Unexpected repeated interrupted extraction: {destination}")
                destination.rename(backup)
            if not destination.exists():
                with zipped.open(entry) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            assert destination.stat().st_size == entry.file_size
            return {"path": str(tail), "bytes": entry.file_size, "crc32": entry.CRC}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, row in enumerate(pool.map(extract, entries), 1):
                manifest.append(row)
                if count % 5000 == 0:
                    print(f"Extracted/verified {count}/{len(entries)} mechanics files", flush=True)
    if not manifest:
        raise RuntimeError("No mechanics data found in official archive")
    result = {
        "url": URL,
        "official_md5": MD5,
        "bytes": SIZE,
        "root": str(DATA / "mechanics"),
        "files": manifest,
    }
    (ROOT / "DATA.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Official archive verified; extracted {len(manifest)} mechanics files to {DATA}",
        flush=True,
    )
