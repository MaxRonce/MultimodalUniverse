"""Tests for HATS-native dataset loader."""

import os

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mmu.data import HATSDataset

TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
HATS_CATALOG = os.path.join(TEST_DATA, "hats_from_script", "sdss_test", "sdss_test")


@pytest.fixture
def catalog_path():
    if not os.path.exists(HATS_CATALOG):
        pytest.skip("HATS test catalog not built (run test_hats_conversion.py first)")
    return HATS_CATALOG


@pytest.fixture
def ds(catalog_path):
    return HATSDataset(catalog_path)


@pytest.fixture
def ds_light(catalog_path):
    return HATSDataset(catalog_path, columns=["ra", "dec", "Z", "object_id"])


class TestBasicLoading:
    def test_len(self, ds):
        assert len(ds) == 364

    def test_schema(self, ds):
        names = [f.name for f in ds.schema]
        assert "ra" in names
        assert "spectrum" in names

    def test_getitem_returns_dict(self, ds_light):
        item = ds_light[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {"ra", "dec", "Z", "object_id"}

    def test_getitem_types(self, ds_light):
        item = ds_light[0]
        assert isinstance(item["ra"], torch.Tensor)
        assert isinstance(item["Z"], torch.Tensor)
        assert isinstance(item["object_id"], str)

    def test_first_and_last(self, ds_light):
        first = ds_light[0]
        last = ds_light[len(ds_light) - 1]
        assert first["ra"].item() != last["ra"].item()

    def test_index_out_of_bounds(self, ds_light):
        with pytest.raises((IndexError, Exception)):
            ds_light[len(ds_light)]


class TestLazyColumns:
    def test_column_subset(self, catalog_path):
        ds_full = HATSDataset(catalog_path)
        ds_partial = HATSDataset(catalog_path, columns=["ra", "dec"])
        assert len(ds_full) == len(ds_partial)
        # Partial should load less data into cache
        _ = ds_partial[0]
        _ = ds_full[0]
        partial_bytes = sum(t.nbytes for t in ds_partial._pixel_cache.values())
        full_bytes = sum(t.nbytes for t in ds_full._pixel_cache.values())
        assert partial_bytes < full_bytes

    def test_spectrum_column(self, catalog_path):
        ds = HATSDataset(catalog_path, columns=["spectrum", "object_id"])
        item = ds[0]
        assert "spectrum" in item
        assert isinstance(item["spectrum"], dict)
        assert "flux" in item["spectrum"]
        assert item["spectrum"]["flux"].shape[0] > 0


class TestSpectrumStruct:
    def test_spectrum_fields(self, catalog_path):
        ds = HATSDataset(catalog_path, columns=["spectrum"])
        item = ds[0]
        spec = item["spectrum"]
        assert set(spec.keys()) == {"flux", "ivar", "lsf_sigma", "lambda", "mask"}

    def test_spectrum_shapes_consistent(self, catalog_path):
        ds = HATSDataset(catalog_path, columns=["spectrum"])
        item = ds[0]
        spec = item["spectrum"]
        length = spec["flux"].shape[0]
        assert spec["ivar"].shape[0] == length
        assert spec["lsf_sigma"].shape[0] == length
        assert spec["lambda"].shape[0] == length
        assert spec["mask"].shape[0] == length

    def test_spectrum_dtypes(self, catalog_path):
        ds = HATSDataset(catalog_path, columns=["spectrum"])
        item = ds[0]
        spec = item["spectrum"]
        assert spec["flux"].dtype == torch.float32
        assert spec["mask"].dtype == torch.bool


class TestSpatialFiltering:
    def test_cone_filter(self, ds_light):
        filtered = ds_light.filter_by_cone(253.0, 30.0, 3600.0)
        assert len(filtered) <= len(ds_light)
        assert len(filtered) > 0

    def test_box_filter(self, ds_light):
        filtered = ds_light.filter_by_box((252.0, 254.0), (29.0, 31.0))
        assert len(filtered) <= len(ds_light)
        assert len(filtered) > 0


class TestDataLoader:
    def test_batch_loading(self, ds_light):
        loader = torch.utils.data.DataLoader(ds_light, batch_size=8)
        batch = next(iter(loader))
        assert batch["ra"].shape == (8,)
        assert batch["Z"].shape == (8,)

    def test_shuffle(self, ds_light):
        loader = torch.utils.data.DataLoader(ds_light, batch_size=16, shuffle=True)
        b1 = next(iter(loader))["ra"]
        b2 = next(iter(loader))["ra"]
        # Shuffled batches should differ (probabilistically)
        assert not torch.equal(b1, b2) or len(ds_light) <= 16

    def test_full_epoch(self, ds_light):
        loader = torch.utils.data.DataLoader(ds_light, batch_size=32)
        total = sum(batch["ra"].shape[0] for batch in loader)
        assert total == len(ds_light)


class TestClearCache:
    def test_cache_clears(self, ds_light):
        _ = ds_light[0]
        assert len(ds_light._pixel_cache) > 0
        ds_light.clear_cache()
        assert len(ds_light._pixel_cache) == 0
