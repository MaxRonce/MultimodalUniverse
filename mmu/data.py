"""HATS-native data loading for Multimodal Universe v2.

Provides PyTorch Dataset and Lightning DataModule backed by HATS catalogs
with lazy column loading, spatial filtering, and cross-matching.
"""

import typing as T
from dataclasses import dataclass
from functools import cached_property

import hats
import lsdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class _ConeFilter:
    ra: float
    dec: float
    radius_arcsec: float

    def apply_hats(self, cat: "hats.catalog.Catalog") -> "hats.catalog.Catalog":
        return cat.filter_by_cone(self.ra, self.dec, self.radius_arcsec)

    def apply_lsdb(self, cat: "lsdb.Catalog") -> "lsdb.Catalog":
        return cat.cone_search(self.ra, self.dec, self.radius_arcsec)


@dataclass(frozen=True)
class _BoxFilter:
    ra_range: tuple[float, float]
    dec_range: tuple[float, float]

    def apply_hats(self, cat: "hats.catalog.Catalog") -> "hats.catalog.Catalog":
        return cat.filter_by_box(self.ra_range, self.dec_range)

    def apply_lsdb(self, cat: "lsdb.Catalog") -> "lsdb.Catalog":
        return cat.box_search(ra=self.ra_range, dec=self.dec_range)


class HATSDataset(Dataset):
    """PyTorch Dataset backed by a HATS catalog.

    Reads only the requested columns from Parquet files, caching loaded
    pixels in memory. Supports spatial filtering via cone/box and cross-matching
    via LSDB. All filter operations return a new HATSDataset that defers the
    same filter to both the underlying ``hats`` and ``lsdb`` views, so e.g.
    ``ds.filter_by_cone(...).crossmatch(other)`` matches against the cone, not
    the unfiltered catalog.

    Args:
        catalog_path: Path to a HATS catalog directory.
        columns: Columns to load. None means all columns.
        catalog: Pre-loaded hats.Catalog (alternative to catalog_path).
        _filters: Internal — list of filter operations to apply lazily.
    """

    def __init__(
        self,
        catalog_path: str | None = None,
        columns: list[str] | None = None,
        catalog: hats.catalog.Catalog | None = None,
        _filters: tuple = (),
    ):
        if catalog is not None:
            self.catalog = catalog
            self._catalog_base_dir = str(catalog.catalog_base_dir)
        elif catalog_path is not None:
            self.catalog = hats.read_hats(catalog_path)
            self._catalog_base_dir = catalog_path
        else:
            raise ValueError("Provide either catalog_path or catalog")

        self.columns = columns
        self._filters = tuple(_filters)
        self._pixel_cache: dict[int, pa.Table] = {}

    @cached_property
    def _pixel_paths(self) -> list:
        return list(self.catalog.get_pixel_paths())

    @cached_property
    def _pixel_row_counts(self) -> list[int]:
        return [pq.read_metadata(p).num_rows for p in self._pixel_paths]

    @cached_property
    def _cumulative_counts(self) -> np.ndarray:
        return np.cumsum([0] + self._pixel_row_counts)

    def __len__(self) -> int:
        return int(self._cumulative_counts[-1])

    def _load_pixel(self, pixel_idx: int) -> pa.Table:
        if pixel_idx not in self._pixel_cache:
            self._pixel_cache[pixel_idx] = pq.read_table(
                self._pixel_paths[pixel_idx], columns=self.columns
            )
        return self._pixel_cache[pixel_idx]

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        """Map global index to (pixel_idx, row_within_pixel)."""
        pixel_idx = int(np.searchsorted(self._cumulative_counts[1:], idx, side="right"))
        row_idx = idx - int(self._cumulative_counts[pixel_idx])
        return pixel_idx, row_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pixel_idx, row_idx = self._resolve_index(idx)
        table = self._load_pixel(pixel_idx)
        row = table.slice(row_idx, 1)
        return _arrow_row_to_torch(row)

    def filter_by_cone(self, ra: float, dec: float, radius_arcsec: float) -> "HATSDataset":
        f = _ConeFilter(ra, dec, radius_arcsec)
        new = HATSDataset(
            columns=self.columns,
            catalog=f.apply_hats(self.catalog),
            _filters=self._filters + (f,),
        )
        new._catalog_base_dir = self._catalog_base_dir
        return new

    def filter_by_box(
        self, ra: tuple[float, float], dec: tuple[float, float]
    ) -> "HATSDataset":
        f = _BoxFilter(ra, dec)
        new = HATSDataset(
            columns=self.columns,
            catalog=f.apply_hats(self.catalog),
            _filters=self._filters + (f,),
        )
        new._catalog_base_dir = self._catalog_base_dir
        return new

    def clear_cache(self):
        self._pixel_cache.clear()

    @property
    def schema(self) -> pa.Schema:
        return self.catalog.schema

    @cached_property
    def lsdb_catalog(self) -> lsdb.Catalog:
        """Return an LSDB catalog with this dataset's column projection AND any
        spatial filters applied. Reads the same on-disk catalog the hats view
        was built from."""
        cat = lsdb.read_hats(self._catalog_base_dir, columns=self.columns)
        for f in self._filters:
            cat = f.apply_lsdb(cat)
        return cat

    def crossmatch(
        self,
        other: "HATSDataset",
        radius_arcsec: float = 1.0,
        n_neighbors: int = 1,
        suffixes: tuple[str, str] = ("_left", "_right"),
    ) -> "CrossMatchedHATSDataset":
        """Cross-match this catalog against another using LSDB.

        Honors ``self.columns`` / ``other.columns`` (passed through to
        ``lsdb.read_hats(columns=...)``) and any spatial filters previously
        applied via ``filter_by_cone`` / ``filter_by_box``.

        Args:
            other: The other HATSDataset to match against.
            radius_arcsec: Maximum match radius in arcseconds.
            n_neighbors: Number of nearest neighbors to find.
            suffixes: Suffixes appended to overlapping column names.
        """
        result = self.lsdb_catalog.crossmatch(
            other.lsdb_catalog,
            n_neighbors=n_neighbors,
            radius_arcsec=radius_arcsec,
            suffixes=suffixes,
        )
        df = result.compute()
        return CrossMatchedHATSDataset(df, suffixes=suffixes)


class CrossMatchedHATSDataset(Dataset):
    """PyTorch Dataset from a cross-matched result.

    Holds the materialized cross-match result as a DataFrame and provides
    indexed access to matched pairs.
    """

    def __init__(
        self,
        df,
        suffixes: tuple[str, str] = ("_left", "_right"),
    ):
        self.df = df.reset_index(drop=True)
        self.suffixes = suffixes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, T.Any]:
        row = self.df.iloc[idx]
        result = {}
        for col in self.df.columns:
            val = row[col]
            if hasattr(val, 'to_dict'):
                # Nested pandas DataFrame (e.g., spectrum struct) → dict of tensors
                d = val.to_dict(orient='list')
                result[col] = {k: _python_to_torch(v) for k, v in d.items()}
            elif hasattr(val, 'item'):
                result[col] = _python_to_torch(val.item())
            else:
                result[col] = _python_to_torch(val)
        return result

    @property
    def matched_count(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)


def _arrow_row_to_torch(row: pa.Table) -> dict[str, torch.Tensor]:
    """Convert a single-row Arrow table to a dict of tensors."""
    result = {}
    for field in row.schema:
        col = row.column(field.name)
        value = col[0]
        result[field.name] = _arrow_value_to_torch(value, field.type)
    return result


def _arrow_value_to_torch(value: pa.Scalar, arrow_type: pa.DataType) -> T.Any:
    """Convert a PyArrow scalar to a torch tensor or nested dict of tensors."""
    if pa.types.is_struct(arrow_type):
        struct = value.as_py()
        return {
            k: _python_to_torch(v) for k, v in struct.items()
        }
    if pa.types.is_list(arrow_type):
        arr = value.as_py()
        return _python_to_torch(arr)
    return _python_to_torch(value.as_py())


def _python_to_torch(value: T.Any) -> T.Any:
    """Convert a Python value to a torch tensor."""
    if isinstance(value, dict):
        return {k: _python_to_torch(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == 0:
            return torch.tensor([])
        first = value[0]
        if isinstance(first, bool):
            return torch.tensor(value, dtype=torch.bool)
        if isinstance(first, (int, float)):
            return torch.tensor(value)
        if isinstance(first, list):
            return torch.tensor(value)
        return value
    if isinstance(value, bool):
        return torch.tensor(value)
    if isinstance(value, int):
        return torch.tensor(value)
    if isinstance(value, float):
        return torch.tensor(value)
    if isinstance(value, str):
        return value
    if value is None:
        return value
    return value
