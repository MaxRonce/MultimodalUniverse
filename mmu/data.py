"""HATS-native data loading for Multimodal Universe v2.

Provides PyTorch Dataset and Lightning DataModule backed by HATS catalogs
with lazy column loading and spatial filtering.
"""

import typing as T
from functools import cached_property

import hats
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


class HATSDataset(Dataset):
    """PyTorch Dataset backed by a HATS catalog.

    Reads only the requested columns from Parquet files, caching loaded
    pixels in memory. Supports spatial filtering via cone/box/MOC.

    Args:
        catalog_path: Path to a HATS catalog directory.
        columns: Columns to load. None means all columns.
        catalog: Pre-loaded hats.Catalog (alternative to catalog_path).
    """

    def __init__(
        self,
        catalog_path: str | None = None,
        columns: list[str] | None = None,
        catalog: hats.catalog.Catalog | None = None,
    ):
        if catalog is not None:
            self.catalog = catalog
        elif catalog_path is not None:
            self.catalog = hats.read_hats(catalog_path)
        else:
            raise ValueError("Provide either catalog_path or catalog")

        self.columns = columns
        self._pixel_cache: dict[tuple[int, int], pa.Table] = {}

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
        filtered = self.catalog.filter_by_cone(ra, dec, radius_arcsec)
        return HATSDataset(columns=self.columns, catalog=filtered)

    def filter_by_box(
        self, ra: tuple[float, float], dec: tuple[float, float]
    ) -> "HATSDataset":
        filtered = self.catalog.filter_by_box(ra, dec)
        return HATSDataset(columns=self.columns, catalog=filtered)

    def clear_cache(self):
        self._pixel_cache.clear()

    @property
    def schema(self) -> pa.Schema:
        return self.catalog.schema


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
