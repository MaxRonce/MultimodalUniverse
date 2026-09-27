"""Check a completed offline dataset and export a small, reproducible QA sample.

Only Parquet footers and sampled row groups are read. Nothing in the production
run is rewritten. W&B publication is a separate, explicit command.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.lsst_dp2.common import BANDS, sha256_file
from scripts.lsst_dp2.validate_parent_sample import (
    _hats_row_count,
    _validate_image,
    image_from_arrow,
)


def check_completed_run(root: Path) -> tuple[dict, list[Path], list[int]]:
    report = json.loads((root / "validation_report.json").read_text())
    plan_path = root / "inputs/plan.json"
    plan = json.loads(plan_path.read_text())
    if report["status"] != "PASS" or report["plan_sha256"] != sha256_file(plan_path):
        raise ValueError("missing PASS or final report does not match the frozen plan")
    expected = plan["objects"]
    for key in ("catalog_rows", "parquet_rows", "hats_rows", "unique_object_ids"):
        if report[key] != expected:
            raise ValueError(f"report/plan mismatch: {key}={report[key]} != {expected}")
    if report["center_checks"] != sum(report["band_present_counts"].values()):
        raise ValueError("centering checks disagree with available band counts")
    hats_rows, properties = _hats_row_count(str(root / "hats"))
    paths = sorted((Path(properties).parent / "dataset").rglob("*.parquet"))
    counts = []
    for index, path in enumerate(paths, 1):
        counts.append(pq.read_metadata(path).num_rows)
        if index % 250 == 0 or index == len(paths):
            print(f"[HATS footers] {index}/{len(paths)} files", flush=True)
    if not paths or hats_rows != expected or sum(counts) != expected:
        raise ValueError(
            f"HATS metadata/data rows differ from {expected}: {hats_rows}/{sum(counts)}"
        )
    print(
        f"PASS: report, snapshot and actual HATS files agree on {expected:,} rows",
        flush=True,
    )
    return report, paths, counts


def sample_locations(counts: list[int], count: int, seed: int) -> dict[int, list[int]]:
    """Uniform sampling of rows, not files, so small partitions are not favored."""
    if count < 1 or sum(counts) < 1:
        raise ValueError("positive sample size and nonempty dataset required")
    offsets = np.cumsum([0, *counts], dtype=np.int64)
    selected = np.random.default_rng(seed).choice(
        int(offsets[-1]), size=min(count, int(offsets[-1])), replace=False
    )
    result = defaultdict(list)
    for index in sorted(selected):
        partition = int(np.searchsorted(offsets[1:], index, side="right"))
        result[partition].append(int(index - offsets[partition]))
    return dict(result)


def read_selected(path: Path, indices: list[int]):
    """Decode selected row groups in small batches; retain no full-image cache."""
    parquet = pq.ParquetFile(path)
    offsets = np.cumsum(
        [0]
        + [
            parquet.metadata.row_group(i).num_rows
            for i in range(parquet.num_row_groups)
        ]
    )
    groups = defaultdict(set)
    for index in indices:
        group = int(np.searchsorted(offsets[1:], index, side="right"))
        groups[group].add(int(index - offsets[group]))
    columns = [
        "object_id",
        "image",
        "i_cModelMag",
        "sersic_reff_major",
        "ra",
        "dec",
        "tract",
        "patch",
    ]
    for group, wanted in sorted(groups.items()):
        start = 0
        for batch in parquet.iter_batches(
            row_groups=[group], columns=columns, batch_size=8
        ):
            for index in sorted(wanted.intersection(range(start, start + len(batch)))):
                row = index - start
                record = {
                    name: batch[name][row].as_py()
                    for name in columns
                    if name != "image"
                }
                record["image"] = image_from_arrow(batch["image"][row])
                yield record
                wanted.remove(index)
            start += len(batch)
            if not wanted:
                break
        if wanted:
            raise ValueError(f"missing selected rows in {path}: {wanted}")


def render(record: dict, path: Path, stretch: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch

    image = record["image"]
    flux, ivar, mask = image["flux"], image["ivar"], image["mask"]
    finite = flux[mask]
    lo, hi = np.percentile(finite, [0.5, 99.5]) if finite.size else (-1, 1)
    if hi <= lo:
        hi = lo + 1
    norm = ImageNormalize(
        vmin=lo,
        vmax=hi,
        stretch=AsinhStretch(a=0.1) if stretch == "asinh" else LinearStretch(),
    )
    positive_ivar = ivar[mask]
    ivar_max = (
        max(float(np.percentile(positive_ivar, 99.5)), 1e-20)
        if positive_ivar.size
        else 1
    )
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5), squeeze=False)
    for index, band in enumerate(BANDS):
        axes[0, index].imshow(
            flux[index], cmap="gray", norm=norm, origin="lower", interpolation="nearest"
        )
        axes[1, index].imshow(
            ivar[index],
            cmap="viridis",
            vmin=0,
            vmax=ivar_max,
            origin="lower",
            interpolation="nearest",
        )
        axes[2, index].imshow(
            ~mask[index],
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        axes[0, index].set_title(
            band + (" (absent)" if not image["band_present"][index] else "")
        )
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, label in zip(
        axes[:, 0], ("Flux [nJy]", "IVAR [nJy^-2]", "Bad pixels = white")
    ):
        axis.set_ylabel(label)
    mag = "n/a" if record["i_cModelMag"] is None else f"{record['i_cModelMag']:.2f}"
    radius = (
        "n/a"
        if record["sersic_reff_major"] is None
        else f"{record['sersic_reff_major']:.2f}"
    )
    fig.suptitle(
        f"{record['object_id']} | i={mag} | Re={radius} arcsec\n"
        f"160 x 160 px | {stretch} flux display, shared band scale | no crop or smoothing",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def export_review(
    root: Path, output: Path, count: int, seed: int, stretch: str
) -> dict:
    if output.exists():
        raise ValueError(
            "review directory already exists; choose a new output directory"
        )
    source_report, paths, counts = check_completed_run(root)
    locations = sample_locations(counts, count, seed)
    output.mkdir(parents=True)
    items = []
    ids = set()
    for partition, indices in locations.items():
        for record in read_selected(paths[partition], indices):
            oid, image = record["object_id"], record["image"]
            if oid in ids:
                raise ValueError(f"duplicate sampled object: {oid}")
            ids.add(oid)
            clean, _, _, _ = _validate_image(oid, image)
            stem = f"object_{len(items):03d}"
            # Small NPZ files retain every raw array and string provenance field.
            np.savez_compressed(
                output / f"{stem}.npz",
                object_id=np.array(oid),
                **{key: np.asarray(value) for key, value in image.items()},
            )
            render(record, output / f"{stem}.png", stretch)
            item = {key: value for key, value in record.items() if key != "image"}
            item.update(
                png=f"{stem}.png",
                npz=f"{stem}.npz",
                clean_fraction=clean,
                band_present=image["band_present"].tolist(),
                psf_valid=image["psf_image_valid"].tolist(),
            )
            items.append(item)
            print(
                f"[review] {len(items)}/{min(count, sum(counts))} object={oid}",
                flush=True,
            )
    review = {
        "status": "PASS",
        "dataset_rows": sum(counts),
        "hats_partitions": len(paths),
        "sample_rows": len(items),
        "seed": seed,
        "sampling": "uniform rows without replacement; no SNR/quality filter",
        "scope": "source pipeline report + all HATS footers + sampled image validation, not a new full pixel audit",
        "rendering": {
            "stretch": stretch,
            "flux_limits_percentiles": [0.5, 99.5],
            "ivar_limits": "0 to valid-pixel percentile 99.5",
            "scale": "shared per object across bands",
            "bad_pixel_white_means": "image.mask == False",
            "raw_arrays_modified": False,
        },
        "source_validation": source_report,
        "items": items,
    }
    (output / "review.json").write_text(
        json.dumps(review, indent=2, allow_nan=False) + "\n"
    )
    cards = "\n".join(
        f'<figure><a href="{item["npz"]}">Raw NPZ: {html.escape(str(item["object_id"]))}</a><img loading="lazy" src="{item["png"]}" alt="Six-band flux, inverse variance and bad-pixel mask"></figure>'
        for item in items
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>DP2 dataset review</title><style>body{font-family:system-ui;margin:24px auto;max-width:1600px;padding:0 12px}"
        "h1{font-size:24px}figure{margin:24px 0}img{display:block;width:100%;height:auto}a{color:#086759}</style>"
        f'<h1>DP2 dataset review: {len(items)} objects</h1><a href="review.json">Validation and sample manifest</a>{cards}</html>'
    )
    print(f"PASS: {output / 'review.json'}", flush=True)
    return review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--run-root", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--count", type=int, default=24)
    export.add_argument("--seed", type=int, default=20260927)
    export.add_argument("--stretch", choices=("linear", "asinh"), default="linear")
    args = parser.parse_args(argv)
    if args.command == "export" and args.count < 1:
        parser.error("--count must be positive")
    try:
        export_review(
            args.run_root.resolve(),
            args.output_dir.resolve(),
            args.count,
            args.seed,
            args.stretch,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
