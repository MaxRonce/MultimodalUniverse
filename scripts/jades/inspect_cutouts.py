"""Inspect JADES Parquet cutouts and optionally render a small RGB contact sheet.

The audit streams small record batches, checks the image schema and records
real-pixel coverage. It does not read the terabyte-scale FITS mosaics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MIRI_BANDS = frozenset((
    "f560w", "f770w", "f1000w", "f1280w", "f1500w", "f1800w", "f2100w", "f2550w",
))


def _sample_rows(files: list[Path], counts: list[int], max_samples: int):
    """Sample uniformly over rows, not merely the first row of each shard.

    Read only the selected row groups: image-rich shards are too large to
    materialize all their rows for a lightweight audit.
    """
    total = sum(counts)
    if not total:
        return
    selected = np.linspace(0, total - 1, num=min(max_samples, total), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    by_file = defaultdict(list)
    for global_row in selected:
        file_index = int(np.searchsorted(offsets[1:], global_row, side="right"))
        by_file[file_index].append(int(global_row - offsets[file_index]))

    for file_index, local_rows in by_file.items():
        parquet = pq.ParquetFile(files[file_index])
        row_group_offsets = np.r_[0, np.cumsum([
            parquet.metadata.row_group(i).num_rows
            for i in range(parquet.metadata.num_row_groups)
        ])]
        cached_group, cached_table = None, None
        for local_row in local_rows:
            group_index = int(np.searchsorted(row_group_offsets[1:], local_row, side="right"))
            if group_index != cached_group:
                cached_table = parquet.read_row_group(group_index)
                cached_group = group_index
            yield cached_table.slice(local_row - row_group_offsets[group_index], 1).to_pylist()[0]


def inspect(scratch: Path, max_samples: int = 24,
            expected_size: int | None = None) -> tuple[dict, list[dict]]:
    if max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if expected_size is not None and expected_size < 1:
        raise ValueError("expected_size must be a positive integer")
    # Distributed Slurm extraction writes disjoint shard subdirectories.
    files = sorted(scratch.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet cutout shards in {scratch}")
    counts = [pq.read_metadata(file).num_rows for file in files]
    total = sum(counts)
    band_counts, phot_counts, phot_method_counts = Counter(), Counter(), Counter()
    unique_ids, samples = set(), []
    mask_coverage = Counter()
    photoz_counts = Counter()
    known_psf = Counter()
    field_counts = Counter()
    miri_objects, joint_nircam_miri_objects, specz_objects = 0, 0, 0
    for row in _sample_rows(files, counts, max_samples):
        sample_id = row["object_id"]
        if sample_id in unique_ids:
            raise ValueError(f"Duplicate sampled object ID: {sample_id}")
        unique_ids.add(sample_id)
        img = row["image"]
        bands = img["band"]
        field_counts[row["field"]] += 1
        if len(bands) != len(set(bands)):
            raise ValueError(f"Duplicate bands for {sample_id}")
        if len(bands) != len(row["image_sources"]):
            raise ValueError(f"Misaligned image provenance for {sample_id}")
        for image_field in ("flux", "ivar", "mask", "psf_fwhm", "scale"):
            if len(bands) != len(img[image_field]):
                raise ValueError(f"Misaligned image.{image_field} for {sample_id}")
        n_miri_bands = sum(b in MIRI_BANDS for b in bands)
        if n_miri_bands:
            miri_objects += 1
            if len(bands) > n_miri_bands:
                joint_nircam_miri_objects += 1
        if row["z_spec"] is not None:
            specz_objects += 1
        for name in ("photoz", "photoz_kron"):
            photoz = row[name]
            if photoz["z_a"] is not None or photoz["z_peak"] is not None:
                photoz_counts[name] += 1
            z_bins, p_bins = photoz["z_bins"], photoz["prob_z_bins"]
            if (z_bins is None) != (p_bins is None) or (
                z_bins is not None and len(z_bins) != len(p_bins)
            ):
                raise ValueError(f"Mismatched photo-z PDF bins for {sample_id}: {name}")
        for name, value in row.items():
            if name.startswith("phot_") and name.endswith("_flux_njy") and value is not None:
                key = name.removeprefix("phot_").removesuffix("_flux_njy")
                phot_method_counts[key] += 1
                if key.endswith("_circ3"):
                    phot_counts[key.removesuffix("_circ3")] += 1
        for i, band in enumerate(bands):
            if band != band.lower():
                raise ValueError(f"Expected lowercase JWST band name for {sample_id}: {band}")
            if not np.isfinite(img["scale"][i]) or img["scale"][i] <= 0:
                raise ValueError(f"Invalid pixel scale for {sample_id} {band}")
            psf = img["psf_fwhm"][i]
            if psf is not None:
                if not np.isfinite(psf) or psf <= 0:
                    raise ValueError(f"Invalid PSF FWHM for {sample_id} {band}")
                known_psf[band] += 1
            flux = np.asarray(img["flux"][i], dtype=np.float32)
            ivar = np.asarray(img["ivar"][i], dtype=np.float32)
            mask = np.asarray(img["mask"][i], dtype=bool)
            if flux.shape != ivar.shape or flux.shape != mask.shape:
                raise ValueError(f"Shape mismatch for {sample_id} {band}")
            if flux.ndim != 2 or flux.shape[0] != flux.shape[1]:
                raise ValueError(f"Expected square 2D cutout for {sample_id} {band}")
            if expected_size is not None and flux.shape != (expected_size, expected_size):
                raise ValueError(f"Incorrect cutout size for {sample_id} {band}: {flux.shape}")
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
              "sample_photometric_band_occurrences": dict(sorted(phot_counts.items())),
              "sample_photometry_flux_occurrences": dict(sorted(phot_method_counts.items())),
              "sample_field_occurrences": dict(sorted(field_counts.items())),
              "sample_miri_objects": miri_objects,
              "sample_joint_nircam_miri_objects": joint_nircam_miri_objects,
              "sample_z_spec_objects": specz_objects,
              "sample_photoz_occurrences": dict(sorted(photoz_counts.items())),
              "sample_known_psf_occurrences": dict(sorted(known_psf.items())),
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
        preferred = ["f356w", "f200w", "f115w"]
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
    parser.add_argument("--expected-size", type=int, default=160,
                        help="Require exactly this square pixel size (default: 160)")
    parser.add_argument("--require-any-band", nargs="+", default=[],
                        help="Fail if none of the sampled cutouts contains any specified band")
    parser.add_argument("--require-photoz", action="store_true",
                        help="Fail if either EAZY photo-z estimate is absent from all sampled objects")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    report, samples = inspect(args.scratch_dir, args.max_samples, args.expected_size)
    if args.require_any_band and not (
        {b.lower() for b in args.require_any_band} & set(report["sample_band_occurrences"])
    ):
        raise RuntimeError(f"No sampled cutout contains any requested band: {args.require_any_band}; "
                           "increase --max-samples or investigate mosaic coverage")
    if args.require_photoz and any(
        report["sample_photoz_occurrences"].get(field, 0) == 0
        for field in ("photoz", "photoz_kron")
    ):
        raise RuntimeError("Missing PHOTOZ or PHOTOZ_KRON estimates in sampled objects; "
                           "check FITS HDU column names and selected sources")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.preview_dir is not None:
        render(samples, args.preview_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
