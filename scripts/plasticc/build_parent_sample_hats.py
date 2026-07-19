"""Build a jagged, quality-annotated MMU HATS catalog for PLAsTiCC."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mmu.hats_configs import MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats_from_parquet_dir


CATALOG_NAME = "plasticc"
QUALITY_POLICY_VERSION = "plasticc_aion_v1"
BANDS = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}


@dataclass(frozen=True)
class QualityPolicy:
    version: str = QUALITY_POLICY_VERSION
    min_quality_epochs: int = 30
    min_quality_bands: int = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--min-quality-epochs", type=int, default=30)
    parser.add_argument("--min-quality-bands", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--only-ingest", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pixel-threshold", type=int, default=100_000)
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--ingest-chunksize", type=int, default=100_000)
    return parser.parse_args(argv)


def discover_inputs(raw_root: Path) -> list[tuple[Path, Path, str]]:
    train_lc = raw_root / "plasticc_train_lightcurves.csv.gz"
    train_meta = raw_root / "plasticc_train_metadata.csv.gz"
    if not train_lc.is_file() or not train_meta.is_file():
        raise FileNotFoundError("PLAsTiCC train lightcurve and metadata CSV files are required.")
    inputs = [(train_lc, train_meta, "plasticc_train")]
    test_meta = raw_root / "plasticc_test_metadata.csv.gz"
    test_parts = sorted(raw_root.glob("plasticc_test_lightcurves_*.csv.gz"))
    if test_meta.is_file():
        inputs.extend((path, test_meta, "plasticc_test") for path in test_parts)
    elif test_parts:
        raise FileNotFoundError("PLAsTiCC test light curves exist but plasticc_test_metadata.csv.gz is missing.")
    return inputs


def iter_complete_object_groups(path: Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield object-complete frames even when a CSV chunk splits an object."""
    carry: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if chunk.empty:
            carry = None
            continue
        last_id = chunk["object_id"].iloc[-1]
        carry = chunk[chunk["object_id"].eq(last_id)].copy()
        complete = chunk[~chunk["object_id"].eq(last_id)]
        for _, group in complete.groupby("object_id", sort=False):
            yield group
    if carry is not None and not carry.empty:
        for _, group in carry.groupby("object_id", sort=False):
            yield group


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    if "object_id" not in metadata:
        raise ValueError(f"{path} is missing object_id")
    metadata = metadata.drop_duplicates("object_id").copy()
    metadata["object_id"] = metadata["object_id"].astype(str)
    return metadata.set_index("object_id", drop=False)


def convert_object(group: pd.DataFrame, metadata: pd.Series, source_split: str, policy: QualityPolicy) -> dict:
    required = {"object_id", "mjd", "passband", "flux", "flux_err"}
    missing = sorted(required - set(group))
    if missing:
        raise ValueError(f"PLAsTiCC lightcurve group is missing columns: {missing}")
    time = pd.to_numeric(group["mjd"], errors="coerce").to_numpy(dtype=np.float64)
    band_id_float = pd.to_numeric(group["passband"], errors="coerce").to_numpy(dtype=np.float64)
    flux = pd.to_numeric(group["flux"], errors="coerce").to_numpy(dtype=np.float64)
    flux_err = pd.to_numeric(group["flux_err"], errors="coerce").to_numpy(dtype=np.float64)
    valid_band = np.isfinite(band_id_float) & np.isin(band_id_float, list(BANDS))
    band_id = np.where(valid_band, band_id_float, -1).astype(np.int16)
    finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & valid_band
    poserr = finite & (flux_err > 0)
    detected_column = "detected_bool" if "detected_bool" in group else "detected" if "detected" in group else None
    if detected_column:
        detected = pd.to_numeric(group[detected_column], errors="coerce").fillna(0).to_numpy(dtype=bool)
    else:
        detected = poserr & (np.abs(flux) / np.maximum(flux_err, 1e-12) >= 5.0)
    quality_bands = np.unique(band_id[poserr])
    n_quality = int(poserr.sum())
    n_quality_bands = int(len(quality_bands))
    candidate = n_quality >= policy.min_quality_epochs and n_quality_bands >= policy.min_quality_bands
    band_labels = [BANDS.get(int(value), "unknown") for value in band_id]
    object_id = str(group["object_id"].iloc[0])
    target = _metadata_value(metadata, "true_target", "target", "obj_type")
    redshift = _metadata_value(metadata, "true_z", "hostgal_specz", "redshift")
    return {
        "object_id": object_id,
        "ra": float(_metadata_value(metadata, "ra", default=np.nan)),
        "dec": float(_metadata_value(metadata, "decl", "dec", default=np.nan)),
        "source_split": source_split,
        "obj_type": None if pd.isna(target) else str(target),
        "redshift": _optional_float(redshift),
        "hostgal_photoz": _optional_float(_metadata_value(metadata, "hostgal_photoz")),
        "hostgal_specz": _optional_float(_metadata_value(metadata, "hostgal_specz")),
        "n_total": int(len(group)),
        "n_finite": int(finite.sum()),
        "n_quality": n_quality,
        "n_detected": int(detected.sum()),
        "n_quality_bands": n_quality_bands,
        "aion_training_candidate_v1": candidate,
        "aion_training_candidate": candidate,
        "quality_policy_version": policy.version,
        "lightcurve": {
            "time": time.tolist(),
            "band": band_labels,
            "band_id": band_id.tolist(),
            "flux": flux.astype(np.float32).tolist(),
            "flux_err": flux_err.astype(np.float32).tolist(),
            "finite_mask": finite.tolist(),
            "quality_mask": poserr.tolist(),
            "detected_mask": detected.tolist(),
        },
    }


def _metadata_value(row: pd.Series, *names: str, default=np.nan):
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def _optional_float(value):
    return None if pd.isna(value) else float(value)


def lightcurve_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("time", pa.list_(pa.float64())),
            pa.field("band", pa.list_(pa.string())),
            pa.field("band_id", pa.list_(pa.int16())),
            pa.field("flux", pa.list_(pa.float32())),
            pa.field("flux_err", pa.list_(pa.float32())),
            pa.field("finite_mask", pa.list_(pa.bool_())),
            pa.field("quality_mask", pa.list_(pa.bool_())),
            pa.field("detected_mask", pa.list_(pa.bool_())),
        ]
    )


def rows_to_table(rows: list[dict]) -> pa.Table:
    if not rows:
        raise ValueError("Cannot build a PLAsTiCC table from no rows.")
    scalar_types = {
        "object_id": pa.string(),
        "ra": pa.float64(),
        "dec": pa.float64(),
        "source_split": pa.string(),
        "obj_type": pa.string(),
        "redshift": pa.float64(),
        "hostgal_photoz": pa.float64(),
        "hostgal_specz": pa.float64(),
        "n_total": pa.int64(),
        "n_finite": pa.int64(),
        "n_quality": pa.int64(),
        "n_detected": pa.int64(),
        "n_quality_bands": pa.int64(),
        "aion_training_candidate_v1": pa.bool_(),
        "aion_training_candidate": pa.bool_(),
        "quality_policy_version": pa.string(),
    }
    columns = {
        name: pa.array([row[name] for row in rows], type=dtype)
        for name, dtype in scalar_types.items()
    }
    columns["lightcurve"] = pa.array([row["lightcurve"] for row in rows], type=lightcurve_type())
    return pa.table(columns)


def convert_input(
    lc_path: Path,
    metadata_path: Path,
    source_split: str,
    scratch: Path,
    *,
    input_index: int,
    chunksize: int,
    max_objects: int | None,
    policy: QualityPolicy,
    resume: bool,
    force: bool,
) -> dict[str, int]:
    metadata = load_metadata(metadata_path)
    rows: list[dict] = []
    objects = epochs = files = 0
    output_batch = 10_000

    def flush() -> None:
        nonlocal rows, files
        if not rows:
            return
        path = scratch / f"input-{input_index:03d}-part-{files:06d}.parquet"
        if force or not (resume and path.is_file() and path.stat().st_size > 0):
            pq.write_table(rows_to_table(rows), path, compression="zstd")
        rows = []
        files += 1

    for group in iter_complete_object_groups(lc_path, chunksize=chunksize):
        object_id = str(group["object_id"].iloc[0])
        if object_id not in metadata.index:
            raise KeyError(f"Object {object_id} from {lc_path.name} is missing from {metadata_path.name}")
        rows.append(convert_object(group, metadata.loc[object_id], source_split, policy))
        objects += 1
        epochs += len(group)
        if len(rows) >= output_batch:
            flush()
        if max_objects is not None and objects >= max_objects:
            break
    flush()
    return {"objects": objects, "epochs": epochs, "files": files}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_ingest and args.only_ingest:
        raise SystemExit("--skip-ingest and --only-ingest are mutually exclusive")
    if not 0 <= args.shard_idx < args.num_shards:
        raise SystemExit("--shard-idx must be in [0, --num-shards)")
    if args.min_quality_epochs < 1 or args.min_quality_bands < 1:
        raise SystemExit("quality thresholds must be positive")
    inputs = discover_inputs(args.raw_root)
    scratch = Path(args.scratch_dir or default_scratch_dir(CATALOG_NAME))
    scratch.mkdir(parents=True, exist_ok=True)
    policy = QualityPolicy(
        min_quality_epochs=args.min_quality_epochs,
        min_quality_bands=args.min_quality_bands,
    )
    totals = {"objects": 0, "epochs": 0, "files": 0}
    if not args.only_ingest:
        for input_index, (lc_path, metadata_path, source_split) in enumerate(inputs):
            if input_index % args.num_shards != args.shard_idx:
                continue
            stats = convert_input(
                lc_path,
                metadata_path,
                source_split,
                scratch,
                input_index=input_index,
                chunksize=args.chunksize,
                max_objects=args.max_objects,
                policy=policy,
                resume=args.resume,
                force=args.force,
            )
            for key, value in stats.items():
                totals[key] += value
        manifest = {
            "input_root": str(args.raw_root),
            "scratch_dir": str(scratch),
            "quality_policy": asdict(policy),
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            **totals,
        }
        (scratch / f"manifest-shard-{args.shard_idx:03d}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    if args.skip_ingest:
        return 0
    catalog_dir = write_hats_from_parquet_dir(
        str(scratch),
        args.output_root,
        CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        chunksize=args.ingest_chunksize,
        n_workers=args.ingest_workers,
        debug=args.ingest_workers == 1,
    )
    policy_path = Path(catalog_dir) / "aion_quality_policy.json"
    policy_payload = {**asdict(policy), "bands": list(BANDS.values())}
    policy_path.write_text(json.dumps(policy_payload, indent=2, sort_keys=True) + "\n")
    if args.scratch_dir is None:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
