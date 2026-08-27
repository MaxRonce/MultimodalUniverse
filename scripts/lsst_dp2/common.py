"""Shared contracts and filesystem helpers for the LSST DP2 pipeline."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.table import Table

CATALOG_NAME = "lsst_dp2"
BANDS = ("u", "g", "r", "i", "z", "y")
EFFECTIVE_WAVELENGTH_M = {
    "u": 367.1e-9,
    "g": 482.7e-9,
    "r": 622.1e-9,
    "i": 754.5e-9,
    "z": 869.1e-9,
    "y": 971.0e-9,
}
IMAGE_SIZE = 160
PIXEL_SCALE_ARCSEC = 0.2
SKYMAP = "lsst_cells_v2"
DP2_COLLECTION = "dp2"
DEFAULT_REJECT_MASK_PLANES = (
    "BAD",
    "SAT",
    "NO_DATA",
    "SUSPECT",
    "UNMASKEDNAN",
)

_COLUMN_ALIASES = {
    "object_id": ("object_id", "objectId"),
    "ra": ("ra", "coord_ra"),
    "dec": ("dec", "coord_dec"),
    "tract": ("tract",),
    "patch": ("patch",),
    "ref_band": ("ref_band", "refBand"),
    "ref_extendedness": ("ref_extendedness", "refExtendedness"),
    "detect_is_isolated": ("detect_is_isolated", "detect_isIsolated"),
}


def read_catalog(path: str | os.PathLike[str]) -> Table:
    """Read a parent catalog from Parquet or any Astropy-supported format."""
    path = str(path)
    if path.lower().endswith((".parquet", ".pq")):
        return Table.from_pandas(pq.read_table(path).to_pandas())
    return Table.read(path)


def resolve_column(table: Table, canonical_name: str, *, required: bool = True) -> str | None:
    """Resolve one canonical parent-catalog field against supported aliases."""
    for candidate in _COLUMN_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in table.colnames:
            return candidate
    if required:
        aliases = ", ".join(_COLUMN_ALIASES.get(canonical_name, (canonical_name,)))
        raise ValueError(f"catalog is missing {canonical_name!r}; expected one of: {aliases}")
    return None


def validate_catalog(table: Table) -> dict[str, str | None]:
    """Return the resolved core columns after validating row identifiers."""
    columns = {
        name: resolve_column(table, name, required=name not in {
            "ref_band", "ref_extendedness", "detect_is_isolated"
        })
        for name in _COLUMN_ALIASES
    }
    if len(table) == 0:
        raise ValueError("parent catalog is empty")
    return columns


def coadd_path(root: str | os.PathLike[str], tract: int, patch: int, band: str) -> Path:
    """Canonical mirrored coadd path for one DP2 dataset."""
    return (
        Path(root)
        / f"tract={int(tract)}"
        / f"patch={int(patch)}"
        / f"band={band}"
        / "deep_coadd.fits"
    )


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def mask_plane_mapping(*headers) -> dict[str, int]:
    """Decode both classic ``MP_*`` and DP2 ``MSKN/MSKM`` mask metadata."""
    mapping: dict[str, int] = {}
    for header in headers:
        for key, value in header.items():
            name = str(key).upper()
            if name.startswith("MP_"):
                try:
                    mapping[name[3:]] = int(value)
                except (TypeError, ValueError):
                    continue
            elif name.startswith("MSKN"):
                suffix = name[4:]
                mask_value = header.get(f"MSKM{suffix}")
                try:
                    mask_value = int(mask_value)
                except (TypeError, ValueError):
                    continue
                if mask_value > 0 and mask_value & (mask_value - 1) == 0:
                    mapping[str(value).strip().upper()] = mask_value.bit_length() - 1
    return mapping


def native_array(array: np.ndarray) -> np.ndarray:
    """Return an array with native byte order for PyArrow."""
    array = np.asarray(array)
    if array.dtype.byteorder not in ("=", "|"):
        return array.byteswap().view(array.dtype.newbyteorder("="))
    return array
