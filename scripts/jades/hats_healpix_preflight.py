"""Cheap read-only JADES HATS sizing audit: only RA/Dec from the Parquet inputs.

Read the 484k coordinates, compute the *actual* densest pixel at candidate
HEALPix orders, and estimate the dense histogram footprint used by hats-import.
Run this before committing to terabytes of splitting/reducing work.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from hats.pixel_math import healpix_shim as hp


def pixel_occupancies(ra: np.ndarray, dec: np.ndarray, order: int) -> np.ndarray:
    """Count sources per occupied NESTED HEALPix pixel without a sky-wide array."""
    pixels = hp.radec2pix(order, ra, dec)
    _, counts = np.unique(pixels, return_counts=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--orders", nargs="+", type=int, default=[10, 11, 12, 13])
    parser.add_argument("--highest-order", type=int, default=13)
    parser.add_argument("--pixel-threshold", type=int, default=1024)
    parser.add_argument("--expected-rows", type=int, default=484219)
    args = parser.parse_args()

    paths = sorted(str(path) for path in args.scratch_dir.rglob("*.parquet"))
    if not paths:
        parser.error("no Parquet files found in --scratch-dir")
    print(f"Reading ra/dec from {len(paths)} Parquet checkpoint files", flush=True)
    coordinates = ds.dataset(paths, format="parquet").to_table(columns=["ra", "dec"])
    ra = coordinates["ra"].combine_chunks().to_numpy(zero_copy_only=False)
    dec = coordinates["dec"].combine_chunks().to_numpy(zero_copy_only=False)
    n_rows = len(ra)
    print(f"Rows: {n_rows:,} (expected: {args.expected_rows:,})", flush=True)
    if n_rows != args.expected_rows:
        raise ValueError("Parquet row count does not match verified extraction")
    if (not np.isfinite(ra).all() or not np.isfinite(dec).all()
            or ((ra < 0) | (ra >= 360)).any()
            or ((dec < -90) | (dec > 90)).any()):
        raise ValueError("Invalid or non-finite RA/Dec coordinates")

    print("Order | Histogram GiB | Occupied pixels | Median rows | P99 rows | Max rows")
    for order in sorted(set(args.orders + [args.highest_order])):
        if not 0 <= order <= 29:
            parser.error(f"invalid HEALPix order: {order}")
        counts = pixel_occupancies(ra, dec, order)
        histogram_gib = (12 * (4 ** order) * 8) / (1024 ** 3)
        print(f"{order:>5} | {histogram_gib:>13.2f} | {len(counts):>15} | "
              f"{np.median(counts):>11.0f} | {np.percentile(counts, 99):>8.0f} | "
              f"{counts.max():>8}", flush=True)
        if order == args.highest_order and counts.max() > args.pixel_threshold:
            raise ValueError(
                f"Order {order} has a pixel with {counts.max()} galaxies, exceeding "
                f"--pixel-threshold {args.pixel_threshold}. Raise the threshold, or "
                "use a finer order with sufficient worker RAM."
            )

    print("PASS: row count, coordinates, and maximum pixel occupancy", flush=True)


if __name__ == "__main__":
    main()
