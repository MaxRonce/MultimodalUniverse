"""Pin locally downloaded JADES DR5 NIRCam/MIRI mosaics and photometric catalogs.

This command reads only FITS headers (not the multi-GB images). It intentionally
rejects unexpected naming, incomplete downloads and missing uncertainty HDUs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from astropy.io import fits


NIRCAM_NAME = re.compile(
    r"hlsp_jades_jwst_nircam_(goods-[sn])_(f\d+[wm])_v5\.0_drz\.fits$",
    re.IGNORECASE,
)
MIRI_NAME = re.compile(
    r"hlsp_jades_jwst_miri_(goodss|goodsn)_(deep|medium(?:-\d+)?)_(f\d+[wm])_v1\.0_drz\.fits$",
    re.IGNORECASE,
)
EXPECTED_COUNTS = {"GOODS-S": 18, "GOODS-N": 16, "MIRI": 18}
UCSC_BASE = "http://slate.ucsc.edu/~brant/jades-dr5"
MIRI_BASE = "https://github.com/staceyalberts/JWST-JADES-MIRI-PARALLELS/releases/download/v1.0"


def _validate_mosaic(path: Path) -> None:
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        # A FITS file with a valid first header can still be truncated in a
        # later image extension. Check advertised padded data spans on disk.
        size = path.stat().st_size
        for i in range(len(hdus)):
            info = hdus.fileinfo(i)
            if info is not None and size < info["datLoc"] + info["datSpan"]:
                raise ValueError(f"{path}: FITS payload truncated at extension {i}")
        for extension in ("SCI", "ERR"):
            if extension not in hdus:
                raise ValueError(f"{path}: missing {extension} HDU")
        sci = hdus["SCI"].header
        err = hdus["ERR"].header
        shape = (sci.get("NAXIS2"), sci.get("NAXIS1"))
        if shape != (err.get("NAXIS2"), err.get("NAXIS1")):
            raise ValueError(f"{path}: SCI/ERR dimensions differ")
        if any(not isinstance(axis, int) or axis <= 0 for axis in shape):
            raise ValueError(f"{path}: invalid SCI shape {shape}")
        if str(sci.get("BUNIT", "")).strip().lower() != "mjy/sr":
            raise ValueError(f"{path}: unexpected SCI BUNIT={sci.get('BUNIT')}")
        if str(err.get("BUNIT", "")).strip().lower() != "mjy/sr":
            raise ValueError(f"{path}: unexpected ERR BUNIT={err.get('BUNIT')}")
        if not np.isfinite(sci.get("PIXAR_SR", np.nan)) or sci["PIXAR_SR"] <= 0:
            raise ValueError(f"{path}: SCI must specify positive PIXAR_SR")
        if not sci.get("CTYPE1", "").startswith("RA") or not sci.get("CTYPE2", "").startswith("DEC"):
            raise ValueError(f"{path}: SCI celestial WCS is missing")


def make_manifest(raw_root: Path, *, allow_partial: bool = False) -> dict:
    files, catalogs, mosaics = [], [], []
    counts = {}
    for field in ("GOODS-S", "GOODS-N"):
        catalog = raw_root / "catalogs" / f"{field}.fits"
        if not catalog.is_file() or catalog.stat().st_size == 0:
            raise FileNotFoundError(f"Missing photometric catalog: {catalog}")
        relative = catalog.relative_to(raw_root).as_posix()
        catalog_name = f"hlsp_jades_jwst_nircam_{field.lower()}_photometry_v5.0_catalog.fits"
        files.append({"path": relative, "url": f"{UCSC_BASE}/{field}/hlsp/catalogs/{catalog_name}"})
        catalogs.append({"field": field, "path": relative})

        folder = raw_root / "mosaics" / field
        paths = sorted(folder.glob("*.fits"))
        counts[field] = len(paths)
        for path in paths:
            match = NIRCAM_NAME.fullmatch(path.name)
            if not match or match.group(1).upper() != field:
                raise ValueError(f"Unexpected NIRCam file for {field}: {path.name}")
            _validate_mosaic(path)
            relative = path.relative_to(raw_root).as_posix()
            files.append({"path": relative, "url": f"{UCSC_BASE}/{field}/hlsp/images/mosaics/{path.name}"})
            mosaics.append({"field": field, "band": match.group(2).upper(), "path": relative,
                            "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                            "uncertainty_kind": "sigma", "unit": "MJy/sr"})

    miri_paths = sorted((raw_root / "mosaics" / "MIRI").glob("*.fits"))
    counts["MIRI"] = len(miri_paths)
    for path in miri_paths:
        match = MIRI_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected MIRI file: {path.name}")
        _validate_mosaic(path)
        field = {"goodss": "GOODS-S", "goodsn": "GOODS-N"}[match.group(1).lower()]
        relative = path.relative_to(raw_root).as_posix()
        files.append({"path": relative, "url": f"{MIRI_BASE}/{path.name}"})
        mosaics.append({"field": field, "band": match.group(3).upper(), "path": relative,
                        "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                        "uncertainty_kind": "sigma", "unit": "MJy/sr"})

    if not allow_partial and counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected input counts {counts}; expected {EXPECTED_COUNTS}")
    return {"release": "JADES_DR5", "catalogs": catalogs, "mosaics": mosaics,
            "files": files, "provenance": {
                "nircam": "JADES DR5 NIRCam global mosaics, v5.0",
                "miri": "JADES DR5 MIRI coordinated parallels, v1.0",
                "uncertainty": "ERR is resampled per-pixel sigma; correlated noise requires separate treatment",
            }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Generate a development manifest with fewer than the expected 52 mosaics")
    args = parser.parse_args()
    manifest = make_manifest(args.raw_root, allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(manifest['catalogs'])} catalogs, "
          f"{len(manifest['mosaics'])} mosaics")


if __name__ == "__main__":
    main()
