"""Render full-size DP2 MMU cutouts as separate grayscale band panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch

from mmu.data import HATSDataset
from scripts.lsst_dp2.plot_high_snr_gallery import _scalar
from scripts.lsst_dp2.stratify_catalog import parse_edges

BANDS = ("u", "g", "r", "i", "z", "y")


def _score(object_id: str, seed: int) -> int:
    payload = f"{seed}:{object_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _slug(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    return re.sub(r"[^0-9a-z]+", "p", f"{value:g}".lower()).strip("p")


def _bounds_label(lower: float, upper: float, unit: str = "") -> str:
    upper_text = "inf" if np.isposinf(upper) else f"{upper:g}"
    return f"[{lower:g}, {upper_text}){unit}"


def _display_limits(values: np.ndarray, percentile: float) -> tuple[float, float]:
    """Return robust display limits without modifying stored pixel values."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return -1.0, 1.0
    tail = (100.0 - percentile) / 2.0
    lower, upper = np.percentile(finite, [tail, 100.0 - tail])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        center = float(np.median(finite))
        return center - 1.0, center + 1.0
    return float(lower), float(upper)


def _normalization(
    flux: np.ndarray,
    mask: np.ndarray,
    percentile: float,
    stretch: str,
    shared_object_scale: bool,
) -> list[ImageNormalize]:
    stretch_function = AsinhStretch(a=0.1) if stretch == "asinh" else LinearStretch()
    if shared_object_scale:
        limits = _display_limits(flux[mask], percentile)
        return [
            ImageNormalize(vmin=limits[0], vmax=limits[1], stretch=stretch_function)
            for _ in BANDS
        ]
    return [
        ImageNormalize(
            vmin=(limits := _display_limits(flux[index][mask[index]], percentile))[0],
            vmax=limits[1],
            stretch=stretch_function,
        )
        for index in range(len(BANDS))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hats-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mag-edges", default="18,20,21,22,23,24")
    parser.add_argument("--size-edges", default="0.4,0.6,1.0,1.5,inf")
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--stretch", choices=("linear", "asinh"), default="asinh")
    parser.add_argument(
        "--shared-object-scale",
        action="store_true",
        help="use one display range for all six bands of each object",
    )
    args = parser.parse_args()
    if args.per_cell <= 0:
        parser.error("--per-cell must be positive")
    if not 0 < args.percentile <= 100:
        parser.error("--percentile must be in (0, 100]")

    mag_edges = parse_edges(args.mag_edges)
    size_edges = parse_edges(args.size_edges, allow_infinite_last=True)
    metadata = HATSDataset(
        args.hats_path,
        columns=["object_id", "i_cModelMag", "sersic_reff_major"],
    )
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index in range(len(metadata)):
        row = metadata[index]
        mag = _scalar(row["i_cModelMag"])
        size = _scalar(row["sersic_reff_major"])
        if not np.isfinite(mag) or not np.isfinite(size):
            continue
        mi = int(np.searchsorted(mag_edges, mag, side="right") - 1)
        si = int(np.searchsorted(size_edges, size, side="right") - 1)
        if 0 <= mi < len(mag_edges) - 1 and 0 <= si < len(size_edges) - 1:
            object_id = str(row["object_id"])
            groups.setdefault((mi, si), []).append((_score(object_id, args.seed), index))

    dataset = HATSDataset(args.hats_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")
    manifest_rows: list[dict] = []

    for mi in range(len(mag_edges) - 1):
        for si in range(len(size_edges) - 1):
            selected = [
                index
                for _, index in sorted(groups.get((mi, si), []))[: args.per_cell]
            ]
            if not selected:
                continue
            fig, axes = plt.subplots(
                len(selected),
                len(BANDS),
                figsize=(12, 2 * len(selected)),
                squeeze=False,
            )
            for row_index, dataset_index in enumerate(selected):
                sample = dataset[dataset_index]
                flux = sample["image"]["flux"].numpy().astype(np.float32)
                mask = sample["image"]["mask"].numpy().astype(bool)
                if flux.shape != (6, 160, 160) or mask.shape != flux.shape:
                    raise ValueError(
                        f"unexpected image shape for {sample['object_id']}: {flux.shape}"
                    )
                norms = _normalization(
                    flux,
                    mask,
                    args.percentile,
                    args.stretch,
                    args.shared_object_scale,
                )
                object_id = str(sample["object_id"])
                mag = _scalar(sample["i_cModelMag"])
                size = _scalar(sample["sersic_reff_major"])
                for band_index, band in enumerate(BANDS):
                    axis = axes[row_index, band_index]
                    pixels = np.ma.array(flux[band_index], mask=~mask[band_index])
                    axis.imshow(
                        pixels,
                        origin="lower",
                        cmap=cmap,
                        norm=norms[band_index],
                        interpolation="nearest",
                    )
                    axis.set_title(band if row_index == 0 else "", fontsize=10)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if band_index == 0:
                        axis.set_ylabel(
                            f"{object_id}\ni={mag:.2f}, Re={size:.2f}\"",
                            fontsize=7,
                        )
                manifest_rows.append(
                    {
                        "object_id": object_id,
                        "i_cModelMag": mag,
                        "sersic_reff_major_arcsec": size,
                        "mag_bin": _bounds_label(mag_edges[mi], mag_edges[mi + 1]),
                        "size_bin": _bounds_label(
                            size_edges[si], size_edges[si + 1], " arcsec"
                        ),
                    }
                )
            scale_label = "shared object" if args.shared_object_scale else "per band"
            fig.suptitle(
                "LSST DP2 MMU cutouts, full 160x160 pixels: "
                f"i={_bounds_label(mag_edges[mi], mag_edges[mi + 1])}, "
                f"Re={_bounds_label(size_edges[si], size_edges[si + 1], ' arcsec')}\n"
                f"raw flux panels; {args.stretch} display, {scale_label} scale",
                fontsize=12,
            )
            fig.tight_layout()
            output = output_dir / (
                f"mag_{_slug(mag_edges[mi])}_{_slug(mag_edges[mi + 1])}__"
                f"reff_{_slug(size_edges[si])}_{_slug(size_edges[si + 1])}__bands.png"
            )
            fig.savefig(output, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {len(selected)} objects to {output}", flush=True)

    if not manifest_rows:
        raise RuntimeError("no HATS rows fall inside the requested bins")
    manifest_path = output_dir / "gallery_manifest.csv"
    with manifest_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    rendering = {
        "hats_path": str(Path(args.hats_path).resolve()),
        "stored_shape": [6, 160, 160],
        "bands": list(BANDS),
        "crop": None,
        "smoothing": None,
        "background_subtraction": None,
        "stretch": args.stretch,
        "percentile": args.percentile,
        "normalization": "shared_object" if args.shared_object_scale else "per_band",
        "invalid_pixels": "displayed black using image.mask",
        "note": "Display normalization only; stored MMU arrays are unchanged.",
    }
    (output_dir / "rendering.json").write_text(
        json.dumps(rendering, indent=2) + "\n",
        encoding="ascii",
    )
    print(f"Wrote {len(manifest_rows)} manifest rows to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
