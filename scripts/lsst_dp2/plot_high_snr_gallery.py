"""Render an asinh RGB gallery of high-S/N extended LSST DP2 objects."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb

from mmu.data import HATSDataset


def _scalar(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _rgb(image: dict) -> np.ndarray:
    flux = image["flux"].numpy().astype(np.float32)
    mask = image["mask"].numpy().astype(bool)
    # Lupton order is red, green, blue; use LSST i, r, g respectively.
    planes = []
    common_mask = mask[3] & mask[2] & mask[1]
    for index in (3, 2, 1):
        plane = flux[index].copy()
        valid = common_mask & np.isfinite(plane)
        background = np.median(plane[valid]) if valid.any() else 0.0
        plane -= background
        plane[~valid] = 0.0
        planes.append(plane)
    positive = np.concatenate([plane[plane > 0] for plane in planes])
    scale = np.percentile(positive, 90) if positive.size else 1.0
    return make_lupton_rgb(*planes, stretch=max(0.1 * scale, 1e-6), Q=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hats-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--min-snr", type=float, default=30.0)
    parser.add_argument("--max-snr", type=float, default=1000.0)
    args = parser.parse_args()

    columns = [
        "object_id", "ref_extendedness", "detect_is_isolated",
        "i_cModelFlux", "i_cModelFluxErr",
    ]
    metadata = HATSDataset(args.hats_path, columns=columns)
    ranked = []
    for index in range(len(metadata)):
        row = metadata[index]
        flux = _scalar(row["i_cModelFlux"])
        error = _scalar(row["i_cModelFluxErr"])
        snr = flux / error if np.isfinite(flux) and np.isfinite(error) and error > 0 else np.nan
        if (
            bool(row["detect_is_isolated"])
            and _scalar(row["ref_extendedness"]) >= 0.5
            and np.isfinite(snr)
            and args.min_snr <= snr <= args.max_snr
        ):
            ranked.append((snr, index))
    ranked.sort(reverse=True)
    selected = ranked[: args.count]
    if not selected:
        raise RuntimeError("no objects satisfy the requested selection")

    dataset = HATSDataset(args.hats_path)
    columns_count = min(4, len(selected))
    rows_count = math.ceil(len(selected) / columns_count)
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(4 * columns_count, 4 * rows_count))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, (snr, index) in zip(axes, selected):
        sample = dataset[index]
        axis.imshow(_rgb(sample["image"]), origin="lower")
        axis.set_title(f"{sample['object_id']}  S/N$_i$={snr:.0f}", fontsize=10)
        axis.set_axis_off()
    for axis in axes[len(selected):]:
        axis.set_visible(False)
    fig.suptitle("LSST DP2 galaxies - asinh RGB (i/r/g)", fontsize=14)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Wrote {len(selected)} galaxies to {output}")


if __name__ == "__main__":
    main()
