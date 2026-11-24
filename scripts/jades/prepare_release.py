"""Validate the completed November JWST collection and stage it without copying images.

Run as ``python -m scripts.jades.prepare_release --help``. This checks scalar
columns only; it never decodes the image arrays or rebuilds the HATS catalogue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

PIPELINE_COMMIT = "6c6b7aba1d3ec5e741af69fb58da8460dbb247d8"
SELECTED_SHA256 = "1b8a58e05dde752307228b62a352a2366bb63309452c2da1f0bc419375b42f57"
FIELD_COUNTS = {"GOODS-S": 303762, "GOODS-N": 180457}
EXPECTED_PARTITIONS = 527


def validate_collection(root: Path) -> dict:
    import hats
    from hats.catalog.dataset.collection_properties import CollectionProperties
    properties = CollectionProperties.read_from_dir(str(root))
    paths = [properties.hats_primary_table_url, *(properties.all_margins or [])]
    if not properties.all_margins or not properties.default_margin:
        raise ValueError("Expected the completed collection's default margin")
    if (root / properties.hats_primary_table_url).resolve() != root / "jades_jwst":
        raise ValueError("Collection main reference differs from jades_jwst")
    catalogs = {}
    for value in paths:
        relative = Path(value)
        if relative.is_absolute() or ":" in value or not (root / relative).resolve().is_relative_to(root):
            raise ValueError(f"Non-portable collection reference: {value}")
        catalog = root / relative
        if not (catalog / "hats.properties").is_file():
            raise ValueError(f"Missing catalogue metadata: {value}")
        hats.read_hats(str(catalog))
        partitions = sorted((catalog / "dataset").rglob("Npix=*.parquet"))
        if not partitions:
            raise ValueError(f"No partitions in referenced catalogue: {value}")
        catalogs[value] = {"partitions": len(partitions),
                           "rows_including_margin_duplicates": sum(pq.read_metadata(p).num_rows for p in partitions)}
    hats.read_hats(str(root))
    return catalogs


def validate(root: Path, coordinates: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    if not (root / "collection.properties").is_file():
        raise ValueError("Expected COLLECTION root with collection.properties")
    main = root / "jades_jwst"
    if not (main / "hats.properties").is_file():
        raise ValueError("Missing main catalogue jades_jwst/hats.properties")
    catalogs = validate_collection(root)
    parts = sorted((main / "dataset").rglob("Npix=*.parquet"))
    if len(parts) != EXPECTED_PARTITIONS:
        raise ValueError(f"Expected {EXPECTED_PARTITIONS} main partitions, found {len(parts)}")
    rows = sum(pq.read_metadata(p).num_rows for p in parts)
    if rows != sum(FIELD_COUNTS.values()):
        raise ValueError(f"Expected 484219 main rows, found {rows}")
    columns = {c: [] for c in ("object_id", "field", "ra", "dec")}
    for i, part in enumerate(parts, 1):
        table = pq.ParquetFile(part).read(columns=list(columns), use_threads=False)
        for col in columns:
            columns[col].extend(table[col].to_pylist())
        if i % 50 == 0 or i == len(parts):
            print(f"Scalar validation: {i}/{len(parts)} partitions", flush=True)
    ids = np.asarray(columns["object_id"], dtype=str)
    order = np.argsort(ids)
    if len(set(ids)) != rows:
        raise ValueError("Duplicate object_id in main catalogue")
    digest = hashlib.sha256("\n".join(ids[order]).encode()).hexdigest()
    if digest != SELECTED_SHA256:
        raise ValueError(f"Selected ID hash differs: {digest}")
    counts = {f: columns["field"].count(f) for f in FIELD_COUNTS}
    if counts != FIELD_COUNTS or sum(counts.values()) != rows:
        raise ValueError(f"Wrong field counts: {counts}")
    for col in ("ra", "dec"):
        if not np.isfinite(np.asarray(columns[col], dtype=float)).all():
            raise ValueError(f"Non-finite {col}")
    if coordinates is not None:
        with np.load(coordinates, allow_pickle=False) as cache:
            cache_order = np.argsort(cache["object_id"])
            for col in columns:
                if not np.array_equal(np.asarray(columns[col])[order], cache[col][cache_order]):
                    raise ValueError(f"Final {col} differs from exact preflight cache")
    schemas = {str(pq.read_schema(p)) for p in parts}
    if len(schemas) != 1:
        raise ValueError("Inconsistent main partition schemas")
    return {"rows": rows, "main_partitions": len(parts), "field_counts": counts,
            "selected_id_sha256": digest,
            "coordinates_equal_preflight": coordinates is not None,
            "catalogs": catalogs, "schema": next(iter(schemas))}


def prepare(root: Path, stage: Path, manifest: Path, coordinates: Path) -> dict:
    root, stage = root.resolve(strict=True), stage.resolve()
    if stage.exists() or stage.is_symlink():
        raise ValueError("Stage exists; reuse it for upload, or choose a NEW stage path")
    if stage.is_relative_to(root) or root.is_relative_to(stage):
        raise ValueError("Stage and source must be separate directory trees")
    manifest = manifest.resolve(strict=True)
    coordinates = coordinates.resolve(strict=True)
    report = validate(root, coordinates)
    files = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlinks in source collection; inspect before packaging")
    files = [p for p in files if p.is_file()]
    reserved = {"README.md", "release"}
    if any(p.relative_to(root).parts[0] in reserved for p in files):
        raise ValueError("Source already contains publication assets")
    stage.parent.mkdir(parents=True, exist_ok=True)
    if root.stat().st_dev != stage.parent.stat().st_dev:
        raise ValueError("Hardlink staging needs the same filesystem; choose another --stage")
    stage.mkdir()  # Exclusive creation: never merge releases or overwrite a destination.
    inventory = []
    for source in files:
        relative = source.relative_to(root)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Metadata is copied: changing staged metadata cannot alter the original.
        if source.suffix == ".parquet":
            os.link(source, target)
        else:
            shutil.copy2(source, target)
        inventory.append({"path": relative.as_posix(), "bytes": source.stat().st_size})
    validate_collection(stage)
    release = stage / "release"
    release.mkdir()
    shutil.copyfile(manifest, release / "input_manifest.json")
    source_dir = Path(__file__).parent
    shutil.copyfile(source_dir / "DATASET_CARD.md", stage / "README.md")
    shutil.copyfile(source_dir / "PUBLISHING.md", release / "PUBLISHING.md")
    shutil.copyfile(__file__, release / "prepare_release.py")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_commit": PIPELINE_COMMIT,
        "pipeline_url": f"https://github.com/MaxRonce/MultimodalUniverse/tree/{PIPELINE_COMMIT}",
        "packaging_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "license_status": "DR5-specific redistribution terms require confirmation before public release",
        "collection_bytes": sum(f["bytes"] for f in inventory),
        "validation": report,
    }
    for name, value in (("provenance.json", provenance), ("collection_files.json", inventory)):
        (release / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"READY: {stage}. Parquet files are hardlinks: keep both trees immutable.", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True,
                        help="Exact .preflight-coordinates.npz from the validated extraction")
    args = parser.parse_args()
    prepare(args.collection, args.stage, args.manifest, args.coordinates)


if __name__ == "__main__":
    main()
