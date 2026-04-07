"""Tests for the HATS catalog verification utilities."""

import os
import pytest

from mmu.cli.build_hats import convert_tiles
from mmu.verify import verify_catalog_against_hdf5
from mmu import hats_configs


TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
SDSS_HDF5 = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")


@pytest.fixture
def patch_cluster_root(monkeypatch):
    monkeypatch.setattr(hats_configs, "MMU_V1_ROOT", TEST_DATA)


@pytest.fixture
def have_test_data():
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test HDF5 not downloaded")


@pytest.fixture
def built_catalog(have_test_data, patch_cluster_root, tmp_path):
    catalog_dir = convert_tiles(
        dataset="sdss",
        healpix_list=[583],
        output_root=str(tmp_path),
        config="sdss",
    )
    return catalog_dir


class TestVerification:
    def test_passes_for_correct_catalog(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        assert report.ok, "\n" + report.summary()

    def test_fails_for_missing_catalog(self):
        report = verify_catalog_against_hdf5("/nonexistent", SDSS_HDF5)
        assert not report.ok

    def test_fails_for_missing_source(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, "/nonexistent.hdf5")
        assert not report.ok

    def test_row_count_check(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        row_check = next(c for c in report.checks if c[0] == "row count matches")
        assert row_check[1] is True

    def test_ra_dec_value_check(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        ra_check = next(c for c in report.checks if "ra values match" in c[0])
        dec_check = next(c for c in report.checks if "dec values match" in c[0])
        assert ra_check[1] is True
        assert dec_check[1] is True

    def test_spectrum_struct_check(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        struct_check = next(c for c in report.checks if c[0] == "spectrum struct present")
        assert struct_check[1] is True

    def test_lazy_column_check(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        lazy_check = next(c for c in report.checks if "lazy column" in c[0])
        assert lazy_check[1] is True

    def test_n_rows_used(self, have_test_data, patch_cluster_root, tmp_path):
        # Build a catalog with only 10 rows
        catalog_dir = convert_tiles(
            dataset="sdss",
            healpix_list=[583],
            output_root=str(tmp_path),
            config="sdss",
            n_rows=10,
        )
        # Verifying with n_rows_used=10 should succeed
        report = verify_catalog_against_hdf5(catalog_dir, SDSS_HDF5, n_rows_used=10)
        row_check = next(c for c in report.checks if c[0] == "row count matches")
        assert row_check[1] is True

    def test_summary_format(self, built_catalog):
        report = verify_catalog_against_hdf5(built_catalog, SDSS_HDF5)
        summary = report.summary()
        assert "Verification of" in summary
        assert "Result:" in summary


class TestGroupedHDF5Verify:
    @pytest.fixture
    def grouped_h5_file(self, tmp_path):
        import h5py
        import numpy as np
        path = tmp_path / "manga_like.hdf5"
        with h5py.File(path, "w") as f:
            for i, name in enumerate(["a-1", "a-2", "a-3"]):
                g = f.create_group(name)
                g.create_dataset("object_id", data=name.encode())
                g.create_dataset("ra", data=float(180 + i))
                g.create_dataset("dec", data=float(20 + i))
                g.create_dataset("z", data=0.01 * (i + 1))
        return str(path)

    @pytest.fixture
    def grouped_catalog(self, grouped_h5_file, tmp_path):
        import h5py
        from mmu.hats_import import auto_arrow_table_from_grouped_hdf5, write_hats
        with h5py.File(grouped_h5_file, "r") as f:
            table = auto_arrow_table_from_grouped_hdf5(f)
        out = tmp_path / "out"
        write_hats([table], str(out), "grouped_test")
        return str(out / "grouped_test" / "grouped_test")

    def test_grouped_verify_passes(self, grouped_catalog, grouped_h5_file):
        report = verify_catalog_against_hdf5(grouped_catalog, grouped_h5_file)
        assert report.ok, "\n" + report.summary()

    def test_grouped_row_count(self, grouped_catalog, grouped_h5_file):
        report = verify_catalog_against_hdf5(grouped_catalog, grouped_h5_file)
        check = next(c for c in report.checks if "row count matches" in c[0])
        assert check[1] is True
