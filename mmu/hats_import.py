"""Shared infrastructure for building HATS catalogs from MMU data.

Two API styles are provided:
- ``build_arrow_table``: explicit schema (used by SDSS-style scripts)
- ``auto_arrow_table_from_hdf5``: introspect an MMU HDF5 file and build a
  PyArrow table automatically, handling spectra/images/lightcurves/tabular
  with the same code path.
"""

import h5py
import numpy as np
import pyarrow as pa
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader
from hats_import.pipeline import pipeline_with_client

# Keys we recognize as parts of a spectrum struct.
SPECTRUM_KEYS = {
    "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
    "spectrum_lsf_sigma", "spectrum_lsf", "spectrum_mask",
    # GALAH variants
    "spectrum_norm_flux", "spectrum_norm_ivar", "spectrum_norm_lambda",
    # APOGEE variants
    "spectrum_pseudo_continuum",
    # LAMOST uses spectrum_wavelength instead of spectrum_lambda
    "spectrum_wavelength",
}

# Possible RA/Dec column name aliases.
RA_ALIASES = ("ra", "RA", "Ra")
DEC_ALIASES = ("dec", "DEC", "Dec", "decl", "DECL")

# Possible object_id aliases.
OBJECT_ID_ALIASES = ("object_id", "OBJECT_ID", "objid", "OBJID", "object_id_")


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


def _decode_strings(arr: np.ndarray) -> list[str]:
    """Convert a numpy array of bytes/strings/numbers to a list of Python strings."""
    if arr.dtype.kind == "S":
        return [x.decode("utf-8", errors="replace") for x in arr]
    if arr.dtype.kind == "U":
        return [str(x) for x in arr]
    return [str(x) for x in arr]


def _flatten_array_to_list_with_shape(
    arr: np.ndarray,
) -> tuple[pa.Array, list[int]]:
    """Flatten an N-D array (per-row) to a ListArray and return the per-row shape.

    For an array of shape (n_rows, d1, d2, ..., dk), returns:
        - a ListArray with one list per row, length = d1 * d2 * ... * dk
        - the shape [d1, d2, ..., dk] (same for every row)
    """
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    n_rows = arr.shape[0]
    per_row_shape = list(arr.shape[1:])
    per_row_size = int(np.prod(per_row_shape)) if per_row_shape else 1
    flat = arr.reshape(n_rows, per_row_size)
    values = pa.array(flat.reshape(-1))
    offsets = np.arange(0, (n_rows + 1) * per_row_size, per_row_size, dtype=np.int32)
    return pa.ListArray.from_arrays(values=values, offsets=offsets), per_row_shape


def _column_to_pyarrow(arr: np.ndarray) -> pa.Array | None:
    """Best-effort conversion of an HDF5 column to a PyArrow array.

    Returns None if the column should be dropped (e.g., 0-D scalar).
    """
    if arr.ndim == 0:
        return None  # scalar metadata, skip

    # String/bytes columns
    if arr.dtype.kind in ("S", "U", "O"):
        return pa.array(_decode_strings(arr))

    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))

    if arr.ndim == 1:
        return pa.array(arr)

    if arr.ndim == 2:
        return np_to_pyarrow_list(arr)

    # 3D+ array — flatten per-row, caller is expected to also store shape
    flat, _ = _flatten_array_to_list_with_shape(arr)
    return flat


def _resolve_alias(h5_keys, aliases):
    """Return the first key from h5_keys matching one of `aliases`, or None."""
    keyset = set(h5_keys)
    for a in aliases:
        if a in keyset:
            return a
    return None


def auto_arrow_table_from_hdf5(
    h5_file: h5py.File,
    *,
    n_rows: int | None = None,
    skip_columns: set[str] | None = None,
    spectrum_struct: bool = True,
    image_columns: tuple[str, ...] = ("image_array",),
) -> pa.Table:
    """Build a PyArrow table from an MMU HDF5 file by introspection.

    Handles:
    - 1D scalar columns (float/int/bool/string)
    - 2D array columns (e.g., flux × filter, spectrum_flux × wavelength) → list
    - 3D+ array columns (e.g., image_array (N, bands, H, W)) → flat list + shape column
    - Spectrum struct (groups spectrum_* columns into a struct named ``spectrum``)
    - RA/Dec aliases (RA, dec, decl, etc.) → normalized to ``ra``/``dec``
    - object_id → cast to string

    Args:
        h5_file: Open h5py.File handle.
        n_rows: If set, read only the first ``n_rows`` rows from each column.
        skip_columns: Set of column names to skip.
        spectrum_struct: If True, group spectrum_* columns into a single struct column.
        image_columns: Tuple of column names that should be flattened with shape metadata.
    """
    skip_columns = set(skip_columns or set())

    keys = list(h5_file.keys())

    # Resolve coordinate columns
    ra_key = _resolve_alias(keys, RA_ALIASES)
    dec_key = _resolve_alias(keys, DEC_ALIASES)
    if ra_key is None or dec_key is None:
        raise ValueError(
            f"Could not find RA/Dec columns. Available keys: {keys[:20]}..."
        )

    obj_id_key = _resolve_alias(keys, OBJECT_ID_ALIASES)

    # Determine slice
    sample = h5_file[ra_key]
    if n_rows is not None and sample.shape[0] > n_rows:
        sl = slice(0, n_rows)
    else:
        sl = slice(None)

    columns: dict[str, pa.Array] = {}

    # Coordinates
    columns["ra"] = pa.array(np.asarray(h5_file[ra_key][sl]).astype(np.float64))
    columns["dec"] = pa.array(np.asarray(h5_file[dec_key][sl]).astype(np.float64))

    # object_id (always string)
    if obj_id_key is not None:
        columns["object_id"] = pa.array(_decode_strings(h5_file[obj_id_key][sl]))

    # Spectrum struct
    spectrum_keys_present = [k for k in keys if k in SPECTRUM_KEYS]
    if spectrum_struct and spectrum_keys_present:
        struct_arrays = []
        struct_names = []
        for k in spectrum_keys_present:
            arr = np.asarray(h5_file[k][sl])
            if arr.ndim == 2:
                struct_arrays.append(np_to_pyarrow_list(arr))
            else:
                struct_arrays.append(pa.array(arr))
            # Strip "spectrum_" prefix for the struct field name
            struct_names.append(k.replace("spectrum_", ""))
        columns["spectrum"] = pa.StructArray.from_arrays(struct_arrays, names=struct_names)
        skip_columns.update(spectrum_keys_present)

    # Image columns: flatten + add shape metadata
    for img_key in image_columns:
        if img_key in keys:
            arr = np.asarray(h5_file[img_key][sl])
            if arr.ndim >= 3:
                flat, shape = _flatten_array_to_list_with_shape(arr)
                columns[img_key] = flat
                columns[f"{img_key}_shape"] = pa.array(
                    [shape] * len(flat), type=pa.list_(pa.int32())
                )
                skip_columns.add(img_key)

    # All remaining columns
    handled = {ra_key, dec_key} | (
        {obj_id_key} if obj_id_key else set()
    ) | skip_columns

    for k in keys:
        if k in handled:
            continue
        try:
            arr = np.asarray(h5_file[k][sl])
        except (TypeError, ValueError):
            continue  # skip unreadable
        col = _column_to_pyarrow(arr)
        if col is None or len(col) != len(columns["ra"]):
            continue
        columns[k] = col

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
