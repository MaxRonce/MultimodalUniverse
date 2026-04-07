"""Tests for the mmu.cli.build_hats CLI.

Slow ``write_hats`` calls are module-scoped so each catalog is built once,
keeping the suite fast.
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mmu import hats_configs
from mmu.cli import build_hats as cli


TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
SDSS_HDF5 = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")


@pytest.fixture(scope="module")
def have_test_data():
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test HDF5 not downloaded")


@pytest.fixture(scope="module")
def cli_full_catalog(have_test_data, tmp_path_factory):
    """Build the full SDSS HATS catalog once via the CLI helper."""
    out = tmp_path_factory.mktemp("cli_full")
    original = hats_configs.MMU_V1_ROOT
    hats_configs.MMU_V1_ROOT = TEST_DATA
    try:
        catalog_dir = cli.convert_tiles(
            dataset="sdss",
            healpix_list=[583],
            output_root=str(out),
            config="sdss",
        )
        yield catalog_dir, Path(out)
    finally:
        hats_configs.MMU_V1_ROOT = original


@pytest.fixture(scope="module")
def cli_n20_catalog(have_test_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("cli_n20")
    original = hats_configs.MMU_V1_ROOT
    hats_configs.MMU_V1_ROOT = TEST_DATA
    try:
        catalog_dir = cli.convert_tiles(
            dataset="sdss",
            healpix_list=[583],
            output_root=str(out),
            config="sdss",
            n_rows=20,
        )
        yield catalog_dir, Path(out)
    finally:
        hats_configs.MMU_V1_ROOT = original


@pytest.mark.slow
def test_convert_tiles_one_pixel(cli_full_catalog):
    out, _ = cli_full_catalog
    assert os.path.exists(out)
    assert os.path.exists(os.path.join(out, "hats.properties")) or \
        os.path.exists(os.path.join(out, "properties"))


@pytest.mark.slow
def test_convert_tiles_writes_parquet(cli_full_catalog):
    _, out_root = cli_full_catalog
    parquet_files = list((out_root / "sdss_sdss" / "sdss_sdss" / "dataset").rglob("*.parquet"))
    assert len(parquet_files) > 0
    table = pq.read_table(parquet_files[0])
    names = table.schema.names
    assert "ra" in names
    assert "dec" in names
    assert "object_id" in names
    assert "spectrum" in names


@pytest.mark.slow
def test_convert_tiles_with_n_rows(cli_n20_catalog):
    _, out_root = cli_n20_catalog
    parquet_files = list((out_root / "sdss_sdss" / "sdss_sdss" / "dataset").rglob("*.parquet"))
    total = sum(pq.read_metadata(p).num_rows for p in parquet_files)
    assert total == 20


def test_convert_tiles_unknown_dataset(tmp_path):
    with pytest.raises(KeyError):
        cli.convert_tiles(
            dataset="not_a_real_dataset",
            healpix_list=[0],
            output_root=str(tmp_path),
        )


def test_convert_tiles_missing_tile(monkeypatch, tmp_path):
    monkeypatch.setattr(hats_configs, "MMU_V1_ROOT", TEST_DATA)
    with pytest.raises(RuntimeError, match="No tiles found"):
        cli.convert_tiles(
            dataset="sdss",
            healpix_list=[99999999],
            output_root=str(tmp_path),
            config="sdss",
        )


@pytest.mark.slow
def test_main_smoke(cli_full_catalog):
    """The full catalog fixture runs main() under the hood; just check it exists."""
    out, _ = cli_full_catalog
    assert os.path.isdir(out)
