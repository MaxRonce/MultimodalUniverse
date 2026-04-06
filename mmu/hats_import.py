"""Shared infrastructure for building HATS catalogs from MMU data.

Each dataset defines a schema (feature lists) and this module handles
the Arrow conversion and hats-import pipeline.
"""

import numpy as np
import pyarrow as pa
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader
from hats_import.pipeline import pipeline_with_client


def np_to_pyarrow_list(array: np.ndarray) -> pa.Array:
    """Convert a 1D or 2D numpy array to a PyArrow array or ListArray."""
    if array.dtype.byteorder == ">":
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    values = pa.array(array.reshape(-1))
    if array.ndim == 1:
        return values
    n_lists, length = array.shape
    offsets = np.arange(0, (n_lists + 1) * length, length, dtype=np.int32)
    return pa.ListArray.from_arrays(values=values, offsets=offsets)


def build_spectrum_column(
    data: dict,
    flux_key: str = "spectrum_flux",
    ivar_key: str = "spectrum_ivar",
    lsf_sigma_key: str = "spectrum_lsf_sigma",
    lambda_key: str = "spectrum_lambda",
    mask_key: str = "spectrum_mask",
) -> pa.StructArray:
    """Build a spectrum struct column from arrays."""
    return pa.StructArray.from_arrays(
        [
            np_to_pyarrow_list(np.asarray(data[flux_key], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(data[ivar_key], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(data[lsf_sigma_key], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(data[lambda_key], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(data[mask_key]).astype(bool)),
        ],
        names=["flux", "ivar", "lsf_sigma", "lambda", "mask"],
    )


def build_arrow_table(
    data: dict,
    float_features: list[str],
    bool_features: list[str],
    has_spectrum: bool = True,
    flux_features: list[str] | None = None,
    flux_filters: list[str] | None = None,
    invert_bool: list[str] | None = None,
) -> pa.Table:
    """Convert a dict of arrays (from astropy Table or HDF5) to a PyArrow table.

    Args:
        data: Dict-like mapping column names to arrays.
        float_features: Scalar float32 columns.
        bool_features: Boolean columns.
        has_spectrum: Whether to include spectrum struct.
        flux_features: Multi-band flux columns (shape [N, n_filters]).
        flux_filters: Filter names for flux columns (e.g., ["U","G","R","I","Z"]).
        invert_bool: Bool features to invert (e.g., ZWARN where 0=good).
    """
    columns = {}

    if has_spectrum:
        columns["spectrum"] = build_spectrum_column(data)

    for f in float_features:
        columns[f] = pa.array(np.asarray(data[f]).astype(np.float32))

    columns["ra"] = pa.array(np.asarray(data["ra"]).astype(np.float64))
    columns["dec"] = pa.array(np.asarray(data["dec"]).astype(np.float64))

    for f in bool_features:
        arr = np.asarray(data[f]).astype(bool)
        if invert_bool and f in invert_bool:
            arr = ~arr
        columns[f] = pa.array(arr)

    if flux_features and flux_filters:
        for f in flux_features:
            flux_data = np.asarray(data[f])
            for n, b in enumerate(flux_filters):
                columns[f"{f}_{b}"] = pa.array(flux_data[:, n].astype(np.float32))

    obj_ids = data["object_id"]
    columns["object_id"] = pa.array([str(oid) for oid in np.asarray(obj_ids)])

    return pa.table(columns)


class ArrowTableReader(InputReader):
    """InputReader that yields pre-built PyArrow tables by index."""

    def __init__(self, tables: list[pa.Table]):
        self.tables = tables

    def read(self, input_file, read_columns=None):
        idx = int(input_file)
        table = self.tables[idx]
        if read_columns:
            table = table.select(read_columns)
        yield table


def write_hats(
    tables: list[pa.Table],
    output_path: str,
    catalog_name: str,
    pixel_threshold: int = 8192,
    n_workers: int = 1,
    debug: bool = True,
):
    """Write a list of PyArrow tables as a HATS catalog.

    Args:
        tables: List of PyArrow tables to write.
        output_path: Root directory for HATS output.
        catalog_name: Name of the output catalog.
        pixel_threshold: Max rows per HATS partition.
        n_workers: Number of Dask workers.
        debug: If True, run single-threaded for easier debugging.
    """
    import os
    from dask.distributed import Client

    tmp_dir = os.path.join(output_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    reader = ArrowTableReader(tables)
    import_args = (
        CollectionArguments(
            output_artifact_name=catalog_name,
            output_path=output_path,
            tmp_dir=tmp_dir,
        )
        .catalog(
            input_file_list=[str(i) for i in range(len(tables))],
            file_reader=reader,
            ra_column="ra",
            dec_column="dec",
            pixel_threshold=pixel_threshold,
            lowest_healpix_order=4,
        )
        .add_margin(margin_threshold=10.0, is_default=True)
    )

    if debug:
        client_kwargs = {"n_workers": 1, "threads_per_worker": 1, "processes": False}
    else:
        client_kwargs = {"n_workers": n_workers, "threads_per_worker": 1}

    with Client(**client_kwargs) as client:
        pipeline_with_client(import_args, client)
