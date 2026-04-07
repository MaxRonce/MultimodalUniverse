"""Tests for the HATS catalog verification utilities.

These tests build a fake catalog directory by hand (just a parquet file in
dataset/Norder=N/Dir=0/) instead of running the full hats-import pipeline.
That keeps each test under ~0.1s.
"""

import os

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mmu.hats_import import auto_arrow_table_from_grouped_hdf5, auto_arrow_table_from_hdf5
from mmu.verify import verify_catalog_against_hdf5


TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
SDSS_HDF5 = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")


def _make_fake_hats_catalog(catalog_dir: str, table: pa.Table, npix: int = 583, norder: int = 4):
    """Write a single parquet file in HATS-style layout: dataset/Norder=N/Dir=0/Npix=M.parquet"""
    dataset_dir = os.path.join(catalog_dir, "dataset", f"Norder={norder}", "Dir=0")
    os.makedirs(dataset_dir, exist_ok=True)
    pq.write_table(table, os.path.join(dataset_dir, f"Npix={npix}.parquet"))


@pytest.fixture(scope="module")
def have_sdss_test_data():
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test HDF5 not downloaded")


@pytest.fixture(scope="module")
def fake_sdss_catalog(have_sdss_test_data, tmp_path_factory):
    """Build a fake HATS catalog from the SDSS test HDF5 (no hats-import call)."""
    with h5py.File(SDSS_HDF5, "r") as f:
        table = auto_arrow_table_from_hdf5(f)
    out = tmp_path_factory.mktemp("fake_sdss") / "sdss_catalog"
    _make_fake_hats_catalog(str(out), table)
    return str(out)


@pytest.fixture(scope="module")
def fake_sdss_catalog_n10(have_sdss_test_data, tmp_path_factory):
    """Slice fixture: only first 10 rows."""
    with h5py.File(SDSS_HDF5, "r") as f:
        table = auto_arrow_table_from_hdf5(f, n_rows=10)
    out = tmp_path_factory.mktemp("fake_sdss_n10") / "sdss_catalog"
    _make_fake_hats_catalog(str(out), table)
    return str(out)


@pytest.fixture(scope="module")
def fake_grouped_catalog(tmp_path_factory):
    """Build a fake HATS catalog from a synthetic grouped HDF5 file."""
    h5_path = tmp_path_factory.mktemp("grouped_src") / "manga_like.hdf5"
    with h5py.File(h5_path, "w") as f:
        for i, name in enumerate(["a-1", "a-2", "a-3"]):
            g = f.create_group(name)
            g.create_dataset("object_id", data=name.encode())
            g.create_dataset("ra", data=float(180 + i))
            g.create_dataset("dec", data=float(20 + i))
            g.create_dataset("z", data=0.01 * (i + 1))

    with h5py.File(h5_path, "r") as f:
        table = auto_arrow_table_from_grouped_hdf5(f)

    out = tmp_path_factory.mktemp("fake_grouped") / "grouped_catalog"
    _make_fake_hats_catalog(str(out), table, npix=0, norder=4)
    return str(out), str(h5_path)


class TestVerification:
    def test_passes_for_correct_catalog(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        assert report.ok, "\n" + report.summary()

    def test_fails_for_missing_catalog(self):
        report = verify_catalog_against_hdf5("/nonexistent", SDSS_HDF5)
        assert not report.ok

    def test_fails_for_missing_source(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, "/nonexistent.hdf5")
        assert not report.ok

    def test_row_count_check(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        row_check = next(c for c in report.checks if c[0] == "row count matches")
        assert row_check[1] is True

    def test_ra_dec_value_check(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        ra_check = next(c for c in report.checks if "ra values match" in c[0])
        dec_check = next(c for c in report.checks if "dec values match" in c[0])
        assert ra_check[1] is True
        assert dec_check[1] is True

    def test_spectrum_struct_check(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        struct_check = next(c for c in report.checks if c[0] == "spectrum struct present")
        assert struct_check[1] is True

    def test_lazy_column_check(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        lazy_check = next(c for c in report.checks if "lazy column" in c[0])
        assert lazy_check[1] is True

    def test_n_rows_used(self, fake_sdss_catalog_n10):
        report = verify_catalog_against_hdf5(fake_sdss_catalog_n10, SDSS_HDF5, n_rows_used=10)
        row_check = next(c for c in report.checks if c[0] == "row count matches")
        assert row_check[1] is True

    def test_summary_format(self, fake_sdss_catalog):
        report = verify_catalog_against_hdf5(fake_sdss_catalog, SDSS_HDF5)
        summary = report.summary()
        assert "Verification of" in summary
        assert "Result:" in summary


class TestGroupedHDF5Verify:
    def test_grouped_verify_passes(self, fake_grouped_catalog):
        catalog_dir, h5_path = fake_grouped_catalog
        report = verify_catalog_against_hdf5(catalog_dir, h5_path)
        assert report.ok, "\n" + report.summary()

    def test_grouped_row_count(self, fake_grouped_catalog):
        catalog_dir, h5_path = fake_grouped_catalog
        report = verify_catalog_against_hdf5(catalog_dir, h5_path)
        check = next(c for c in report.checks if "row count matches" in c[0])
        assert check[1] is True
