"""Shared infrastructure for building HATS catalogs from MMU survey data.

The public surface is small:

- ``np_to_pyarrow_list`` — convert a 2D numpy array (one row per object) to a
  PyArrow ListArray suitable for use as a column in a HATS catalog.
- ``ArrowTableReader`` — adapter that lets ``hats-import`` consume in-memory
  PyArrow tables instead of reading from disk.
- ``write_hats`` — feed a list of PyArrow tables through the ``hats-import``
  pipeline to produce a HATS catalog (Parquet + healpix partitioning + margin
  cache + collection wrapper).

Per-survey ``build_parent_sample_hats.py`` scripts (under ``scripts/{survey}/``)
build their own PyArrow tables from the survey-native raw inputs and call
``write_hats``. This module deliberately knows nothing about HDF5.
"""

import logging
import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.catalog.file_readers import InputReader
from hats_import.pipeline import pipeline_with_client

LOGGER = logging.getLogger(__name__)


def to_native_endian(array: np.ndarray) -> np.ndarray:
    """Return a copy of ``array`` in the machine's native byte order.

    Astropy reads FITS files in the file's byte order (usually big-endian) but
    PyArrow refuses to ingest byte-swapped arrays — convert before handing to
    ``pa.array``.
    """
    if array.dtype.byteorder in ("=", "|"):
        return array
    return array.byteswap().view(array.dtype.newbyteorder("="))


def np_to_pyarrow_list(array: np.ndarray) -> pa.Array:
    """Convert a 1D numpy array to a flat PyArrow array, or a 2D numpy array
    (shape ``(n_rows, length)``) to a PyArrow ListArray with one ``length``-long
    list per row.
    """
    if array.dtype.byteorder == ">":
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    values = pa.array(array.reshape(-1))
    if array.ndim == 1:
        return values
    if array.ndim != 2:
        raise ValueError(
            f"np_to_pyarrow_list expects 1D or 2D, got ndim={array.ndim}"
        )
    n_lists, length = array.shape
    offsets = np.arange(0, (n_lists + 1) * length, length, dtype=np.int32)
    return pa.ListArray.from_arrays(values=values, offsets=offsets)


class ArrowTableReader(InputReader):
    """``hats-import`` ``InputReader`` that yields pre-built PyArrow tables.

    Each "input file" is just an integer index into the in-memory table list.
    """

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
    *,
    pixel_threshold: int = 8192,
    lowest_healpix_order: int = 4,
    margin_threshold_arcsec: float = 10.0,
    n_workers: int = 1,
    debug: bool = True,
) -> str:
    """Write a list of PyArrow tables as a HATS catalog under ``output_path``.

    Uses a private temp directory (under the system tmp dir) for hats-import
    intermediate state and cleans it up on exit.

    Args:
        tables: PyArrow tables to write. Must contain ``ra`` (float64) and
            ``dec`` (float64) columns.
        output_path: Root directory under which the catalog collection is written.
        catalog_name: Name of the output catalog (becomes both the collection
            and the inner catalog directory name).
        pixel_threshold: Max rows per HATS partition.
        lowest_healpix_order: Coarsest HEALPix order used for partitioning.
        margin_threshold_arcsec: Width of the margin cache in arcseconds.
        n_workers: Dask workers to use in production mode.
        debug: If True, run single-process / single-thread (easier debugging).

    Returns:
        The path to the inner catalog directory ``output_path/{catalog_name}/{catalog_name}``.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"hats_import_{catalog_name}_")
    try:
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
                lowest_healpix_order=lowest_healpix_order,
            )
            .add_margin(margin_threshold=margin_threshold_arcsec, is_default=True)
        )

        if debug:
            client_kwargs = {"n_workers": 1, "threads_per_worker": 1, "processes": False}
        else:
            client_kwargs = {"n_workers": n_workers, "threads_per_worker": 1}

        with Client(**client_kwargs) as client:
            pipeline_with_client(import_args, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return os.path.join(output_path, catalog_name, catalog_name)
