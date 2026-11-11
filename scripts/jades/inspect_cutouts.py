"""Inspect JADES Parquet cutouts and optionally render a small RGB contact sheet.

The audit streams small record batches, checks the image schema and records
real-pixel coverage. It does not read the terabyte-scale FITS mosaics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def inspect(scratch: Path, max_samples: int = 24) -> tuple[dict, list[dict]]:
    files = sorted(scratch.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet cutout shards in {scratch}")
    total, band_counts, unique_ids, samples = 0, Counter(), set(), []
    mask_coverage = Counter()
    selected_files = {files[i] for i in np.linspace(
        0, len(files) - 1, num=min(max_samples, len(files)), dtype=int)}
    for file in files:
        parquet = pq.ParquetFile(file)
        total += parquet.metadata.num_rows
        if file not in selected_files:
            continue
        # Stratify the sample across both fields and the full shard sequence.
        batches = parquet.iter_batches(batch_size=1)
        try:
            row = next(batches).to_pylist()[0]
        except StopIteration:
            continue
        sample_id = row["object_id"]
        if sample_id in unique_ids:
            raise ValueError(f"Duplicate sampled object ID: {sample_id}")
        unique_ids.add(sample_id)
        img = row["image"]
        bands = img["band"]
        if len(bands) != len(set(bands)):
            raise ValueError(f"Duplicate bands for {sample_id}")
        if len(bands) != len(row["image_sources"]):
            raise ValueError(f"Misaligned image provenance for {sample_id}")
        if len(bands) != len(row["photometry"]["band"]):
            raise ValueError(f"Misaligned photometry for {sample_id}")
        for i, band in enumerate(bands):
            flux = np.asarray(img["flux"][i], dtype=np.float32)
            ivar = np.asarray(img["ivar"][i], dtype=np.float32)
            mask = np.asarray(img["mask"][i], dtype=bool)
            if flux.shape != ivar.shape or flux.shape != mask.shape:
                raise ValueError(f"Shape mismatch for {sample_id} {band}")
            if flux.ndim != 2 or flux.shape[0] != flux.shape[1]:
                raise ValueError(f"Expected square 2D cutout for {sample_id} {band}")
            if not np.isfinite(flux).all() or not np.isfinite(ivar).all():
                raise ValueError(f"Non-finite stored pixel for {sample_id} {band}")
            if np.any(ivar[~mask] != 0) or np.any(ivar[mask] <= 0):
                raise ValueError(f"Inconsistent ivar/mask for {sample_id} {band}")
            band_counts[band] += 1
            mask_coverage[band] += float(mask.mean())
        samples.append(row)
    report = {"parquet_files": len(files), "total_objects": total,
              "sampled_objects": len(samples),
              "sample_band_occurrences": dict(sorted(band_counts.items())),
              "sample_mean_valid_fraction": {
                  band: round(mask_coverage[band] / count, 4)
                  for band, count in sorted(band_counts.items())}}
    return report, samples


def render(samples: list[dict], directory: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install optional matplotlib: uv run --extra viz ...") from exc
    directory.mkdir(parents=True, exist_ok=True)
    for row in samples:
        img = row["image"]
        bands = img["band"]
        preferred = ["F356W", "F200W", "F115W"]
        selected = [band for band in preferred if band in bands]
        if not selected:
            selected = bands[:3]
        if not selected:
            continue
        layers = []
        for band in selected:
            k = bands.index(band)
            flux = np.asarray(img["flux"][k], dtype=np.float32)
            mask = np.asarray(img["mask"][k], dtype=bool)
            valid = flux[mask]
            if valid.size == 0:
                layers.append(np.zeros_like(flux))
                continue
            lo, hi = np.percentile(valid, [5, 99.5])
            normalized = np.clip((flux - lo) / max(hi - lo, 1e-6), 0, 1)
            layers.append(np.where(mask, np.arcsinh(8 * normalized) / np.arcsinh(8), 0))
        while len(layers) < 3:
            layers.append(layers[-1])
        rgb = np.stack(layers[:3], axis=-1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb, interpolation="nearest", origin="lower")
        ax.set_title(f"{row['object_id']}  RGB: {', '.join(selected)}", fontsize=9)
        ax.axis("off")
        filename = row["object_id"].replace(":", "_") + ".png"
        fig.savefig(directory / filename, dpi=120, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--require-any-band", nargs="+", default=[],
                        help="Fail if none of the sampled cutouts contains any specified band")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    report, samples = inspect(args.scratch_dir, args.max_samples)
    if args.require_any_band and not (set(args.require_any_band) & set(report["sample_band_occurrences"])):
        raise RuntimeError(f"No sampled cutout contains any requested band: {args.require_any_band}; "
                           "increase --max-samples or investigate mosaic coverage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.preview_dir is not None:
        render(samples, args.preview_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
