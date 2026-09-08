"""Render representative LSST DP2 galleries in magnitude and size cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmu.data import HATSDataset
from scripts.lsst_dp2.plot_high_snr_gallery import _rgb, _scalar
from scripts.lsst_dp2.stratify_catalog import parse_edges


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hats-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mag-edges", default="20,21,22,23,24")
    parser.add_argument("--size-edges", default="0.4,0.6,1.0,1.5,inf")
    parser.add_argument("--per-cell", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--crop-size", type=int, default=80)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--display-sigma", type=float, default=1.5)
    parser.add_argument(
        "--stretch-njy",
        type=float,
        help="fixed Lupton stretch; omit for a separate adaptive stretch per object",
    )
    args = parser.parse_args()
    if args.per_cell <= 0:
        parser.error("--per-cell must be positive")

    mag_edges = parse_edges(args.mag_edges)
    size_edges = parse_edges(args.size_edges, allow_infinite_last=True)
    metadata_columns = ["object_id", "i_cModelMag", "sersic_reff_major"]
    metadata = HATSDataset(args.hats_path, columns=metadata_columns)

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
    manifest_rows = []
    for mi in range(len(mag_edges) - 1):
        for si in range(len(size_edges) - 1):
            candidates = sorted(groups.get((mi, si), []))
            selected = [index for _, index in candidates[: args.per_cell]]
            if not selected:
                print(f"EMPTY cell magnitude={mi} size={si}", flush=True)
                continue

            columns = min(4, len(selected))
            rows = math.ceil(len(selected) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
            axes = np.atleast_1d(axes).reshape(-1)
            for axis, index in zip(axes, selected):
                sample = dataset[index]
                mag = _scalar(sample["i_cModelMag"])
                size = _scalar(sample["sersic_reff_major"])
                object_id = str(sample["object_id"])
                axis.imshow(
                    _rgb(
                        sample["image"],
                        crop_size=args.crop_size,
                        smooth_sigma=args.smooth_sigma,
                        display_sigma=args.display_sigma,
                        stretch_njy=args.stretch_njy,
                    ),
                    origin="lower",
                )
                axis.set_title(f"{object_id}\ni={mag:.2f}, Re={size:.2f} arcsec", fontsize=9)
                axis.set_axis_off()
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
            for axis in axes[len(selected) :]:
                axis.set_visible(False)
            mode = "fixed" if args.stretch_njy is not None else "adaptive"
            fig.suptitle(
                "LSST DP2 galaxy candidates: "
                f"i={_bounds_label(mag_edges[mi], mag_edges[mi + 1])}, "
                f"Re={_bounds_label(size_edges[si], size_edges[si + 1], ' arcsec')} "
                f"({mode} stretch)",
                fontsize=13,
            )
            fig.tight_layout()
            output = output_dir / (
                f"mag_{_slug(mag_edges[mi])}_{_slug(mag_edges[mi + 1])}__"
                f"reff_{_slug(size_edges[si])}_{_slug(size_edges[si + 1])}__{mode}.png"
            )
            fig.savefig(output, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {len(selected)} objects to {output}", flush=True)

    if not manifest_rows:
        raise RuntimeError("no HATS rows fall inside the requested bins")
    manifest = output_dir / "gallery_manifest.csv"
    with manifest.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote gallery manifest to {manifest}", flush=True)

    rendering = {
        "hats_path": str(Path(args.hats_path).resolve()),
        "mag_edges": mag_edges.tolist(),
        "size_edges_arcsec": [
            float(value) if np.isfinite(value) else None for value in size_edges
        ],
        "per_cell": args.per_cell,
        "seed": args.seed,
        "crop_size": args.crop_size,
        "smooth_sigma": args.smooth_sigma,
        "display_sigma": args.display_sigma,
        "stretch_mode": "fixed" if args.stretch_njy is not None else "adaptive",
        "stretch_njy": args.stretch_njy,
        "rgb_bands": ["i", "r", "g"],
    }
    rendering_path = output_dir / "rendering.json"
    rendering_path.write_text(json.dumps(rendering, indent=2) + "\n", encoding="ascii")
    print(f"Wrote rendering parameters to {rendering_path}", flush=True)


if __name__ == "__main__":
    main()
