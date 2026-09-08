"""PyArrow representation of the LSST DP2 MMU image contract."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from scripts.lsst_dp2.coadd import PSF_SIZE
from scripts.lsst_dp2.common import (
    BANDS,
    IMAGE_SIZE,
    SELECTION_FLAG_COLUMNS,
    SELECTION_FLOAT_COLUMNS,
    SKYMAP,
)

PHOTOMETRY_SUFFIXES = ("psfFlux", "psfFluxErr", "cModelFlux", "cModelFluxErr")
OUTPUT_SCHEMA_VERSION = 5


def _nested_column(
    arrays: list[np.ndarray], shape: tuple[int, ...], value_type: pa.DataType
) -> pa.Array:
    """Build nested Arrow lists without expanding NumPy values into Python scalars."""
    if not arrays:
        raise ValueError("cannot build an image column without records")
    stacked = np.ascontiguousarray(np.stack(arrays))
    if stacked.shape[1:] != shape:
        raise ValueError(f"expected nested shape {shape}, found {stacked.shape[1:]}")
    values: pa.Array = pa.array(stacked.reshape(-1), type=value_type)
    for size in reversed(shape):
        offsets = np.arange(0, len(values) + 1, size, dtype=np.int32)
        values = pa.ListArray.from_arrays(offsets, values)
    return values


def build_image_struct(records: list[dict]) -> pa.StructArray:
    n_bands = len(BANDS)
    image_shape = (n_bands, IMAGE_SIZE, IMAGE_SIZE)
    return pa.StructArray.from_arrays(
        [
            pa.array([list(BANDS)] * len(records), type=pa.list_(pa.string())),
            _nested_column([r["flux"] for r in records], image_shape, pa.float32()),
            _nested_column([r["ivar"] for r in records], image_shape, pa.float32()),
            _nested_column([r["mask"] for r in records], image_shape, pa.bool_()),
            _nested_column([r["mask_bits"] for r in records], image_shape, pa.int32()),
            _nested_column(
                [r["psf_image"] for r in records],
                (n_bands, PSF_SIZE, PSF_SIZE),
                pa.float32(),
            ),
            _nested_column(
                [r["psf_image_valid"] for r in records], (n_bands,), pa.bool_()
            ),
            _nested_column([r["psf_fwhm"] for r in records], (n_bands,), pa.float32()),
            _nested_column([r["scale"] for r in records], (n_bands,), pa.float32()),
            _nested_column(
                [r["band_present"] for r in records], (n_bands,), pa.bool_()
            ),
            pa.array([r["psf_source"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["dataset_id"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["sha256"] for r in records], type=pa.list_(pa.string())),
            pa.array(
                [r["mask_plane_map"] for r in records],
                type=pa.list_(pa.string()),
            ),
            pa.array([["nJy"] * n_bands] * len(records), type=pa.list_(pa.string())),
            pa.array(
                [["nJy^-2"] * n_bands] * len(records),
                type=pa.list_(pa.string()),
            ),
        ],
        names=[
            "band",
            "flux",
            "ivar",
            "mask",
            "mask_bits",
            "psf_image",
            "psf_image_valid",
            "psf_fwhm",
            "scale",
            "band_present",
            "psf_source",
            "dataset_id",
            "sha256",
            "mask_plane_map",
            "flux_unit",
            "ivar_unit",
        ],
    )


def build_table(records: list[dict]) -> pa.Table:
    """Build one MMU-compatible Arrow table from homogeneous source records."""
    columns: dict[str, pa.Array] = {
        "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "tract": pa.array([r["tract"] for r in records], type=pa.int32()),
        "patch": pa.array([r["patch"] for r in records], type=pa.int32()),
        "ref_band": pa.array([r["ref_band"] for r in records], type=pa.string()),
        "ref_extendedness": pa.array(
            [r["ref_extendedness"] for r in records], type=pa.float32()
        ),
        "detect_is_isolated": pa.array(
            [r["detect_is_isolated"] for r in records], type=pa.bool_()
        ),
        "image": build_image_struct(records),
        "dp2_collection": pa.array(["dp2"] * len(records), type=pa.string()),
        "skymap": pa.array([SKYMAP] * len(records), type=pa.string()),
    }
    for band in BANDS:
        for suffix in PHOTOMETRY_SUFFIXES:
            name = f"{band}_{suffix}"
            columns[name] = pa.array(
                [r["photometry"][name] for r in records], type=pa.float32()
            )
    for name in SELECTION_FLOAT_COLUMNS:
        columns[name] = pa.array(
            [r["selection_metadata"].get(name) for r in records], type=pa.float32()
        )
    for name in SELECTION_FLAG_COLUMNS:
        columns[name] = pa.array(
            [r["selection_metadata"].get(name) for r in records], type=pa.bool_()
        )
    return pa.table(columns)
