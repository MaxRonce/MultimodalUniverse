"""Report JADES DR5 photometric SNR cut yields without imposing a build cut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


def read_extension(path: Path, extension: str) -> Table:
    with fits.open(path, memmap=True) as hdus:
        names = {hdu.name.upper(): i for i, hdu in enumerate(hdus)}
        if extension.upper() not in names:
            raise ValueError(f"{path}: missing HDU {extension}; available: {list(names)}")
        return Table(hdus[names[extension.upper()]].data)


def summarize_snr(catalogs: list[Path], bands: list[str], aperture: str = "CIRC3") -> dict:
    summaries = {}
    for path in catalogs:
        table = read_extension(path, "CIRC")
        n = len(table)
        snrs = np.full((n, len(bands)), np.nan, dtype=np.float32)
        for i, band in enumerate(bands):
            value, error = f"{band}_{aperture}", f"{band}_{aperture}_e"
            if value not in table.colnames or error not in table.colnames:
                continue
            flux, sigma = np.asarray(table[value], dtype=float), np.asarray(table[error], dtype=float)
            np.divide(flux, sigma, out=snrs[:, i], where=np.isfinite(sigma) & (sigma > 0))
            snrs[~(np.isfinite(sigma) & (sigma > 0)), i] = np.nan
        valid = np.isfinite(snrs)
        summary = {"objects": n, "aperture": aperture, "available_bands": {}}
        for i, band in enumerate(bands):
            summary["available_bands"][band] = int(valid[:, i].sum())
        for threshold in (2, 3, 5, 10):
            count = ((snrs >= threshold) & valid).sum(axis=1)
            summary[f"snr_ge_{threshold}"] = {
                "at_least_1_band": int((count >= 1).sum()),
                "at_least_2_bands": int((count >= 2).sum()),
                "at_least_4_bands": int((count >= 4).sum()),
            }
        summaries[path.stem] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", nargs="+", type=Path)
    parser.add_argument("--bands", nargs="+", required=True)
    parser.add_argument("--aperture", default="CIRC3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_snr(args.catalogs, [b.upper() for b in args.bands], args.aperture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
