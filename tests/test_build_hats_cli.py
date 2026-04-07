"""Tests for the mmu.cli.build_hats CLI.

These tests monkeypatch the cluster paths to point at local test data
so the CLI can be exercised without cluster access.
"""

import os
import sys

import pyarrow.parquet as pq
import pytest

from mmu.cli import build_hats as cli
from mmu import hats_configs


TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
SDSS_HDF5 = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")


@pytest.fixture
def patch_cluster_root(monkeypatch):
    """Point MMU_V1_ROOT at the local test_data directory."""
    monkeypatch.setattr(hats_configs, "MMU_V1_ROOT", TEST_DATA)


@pytest.fixture
def have_test_data():
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test HDF5 not downloaded")


def test_convert_tiles_one_pixel(have_test_data, patch_cluster_root, tmp_path):
    out = cli.convert_tiles(
        dataset="sdss",
        healpix_list=[583],
        output_root=str(tmp_path),
        config="sdss",
    )
    assert os.path.exists(out)
    assert os.path.exists(os.path.join(out, "hats.properties")) or \
        os.path.exists(os.path.join(out, "properties"))


def test_convert_tiles_writes_parquet(have_test_data, patch_cluster_root, tmp_path):
    out = cli.convert_tiles(
        dataset="sdss",
        healpix_list=[583],
        output_root=str(tmp_path),
        config="sdss",
    )
    parquet_files = list((tmp_path / "sdss_sdss" / "sdss_sdss" / "dataset").rglob("*.parquet"))
    assert len(parquet_files) > 0
    # Read first parquet to verify schema
    table = pq.read_table(parquet_files[0])
    names = table.schema.names
    assert "ra" in names
    assert "dec" in names
    assert "object_id" in names
    assert "spectrum" in names


def test_convert_tiles_with_n_rows(have_test_data, patch_cluster_root, tmp_path):
    out = cli.convert_tiles(
        dataset="sdss",
        healpix_list=[583],
        output_root=str(tmp_path),
        config="sdss",
        n_rows=20,
    )
    parquet_files = list((tmp_path / "sdss_sdss" / "sdss_sdss" / "dataset").rglob("*.parquet"))
    total = sum(pq.read_metadata(p).num_rows for p in parquet_files)
    assert total == 20


def test_convert_tiles_unknown_dataset(patch_cluster_root, tmp_path):
    with pytest.raises(KeyError):
        cli.convert_tiles(
            dataset="not_a_real_dataset",
            healpix_list=[0],
            output_root=str(tmp_path),
        )


def test_convert_tiles_missing_tile(patch_cluster_root, tmp_path):
    with pytest.raises(RuntimeError, match="No tiles found"):
        cli.convert_tiles(
            dataset="sdss",
            healpix_list=[99999999],
            output_root=str(tmp_path),
            config="sdss",
        )


def test_main_smoke(have_test_data, patch_cluster_root, tmp_path):
    rc = cli.main([
        "--dataset", "sdss",
        "--healpix", "583",
        "--output", str(tmp_path),
        "--config", "sdss",
    ])
    assert rc == 0
