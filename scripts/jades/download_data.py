"""Download the version-pinned JADES DR5 input manifest, resumably.

The manifest is deliberately explicit: the temporary UCSC directory structure
must not silently change which FITS products a scientific build consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def download_file(url: str, destination: Path, sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (sha256 is None or file_hash(destination) == sha256):
        return destination
    temporary = destination.with_name(destination.name + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "MMU-JADES-DR5/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        # Some HTTP servers ignore Range; restart instead of corrupting a file.
        mode = "ab" if offset and response.status == 206 else "wb"
        with temporary.open(mode) as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
    if sha256 and file_hash(temporary) != sha256:
        raise ValueError(f"SHA256 mismatch for {destination}")
    os.replace(temporary, destination)
    return destination


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_manifest(manifest_path: Path, raw_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != "JADES_DR5":
        raise ValueError("Only a pinned JADES_DR5 manifest is supported")
    files = manifest.get("files", [])
    if not files:
        raise ValueError("Manifest has no files; inspect DR5 URLs before downloading")
    for entry in files:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe relative path: {path}")
        target = raw_root / path
        download_file(entry["url"], target, entry.get("sha256"))
        print(f"OK {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    arguments = parser.parse_args()
    fetch_manifest(arguments.manifest, arguments.raw_root)


if __name__ == "__main__":
    main()
