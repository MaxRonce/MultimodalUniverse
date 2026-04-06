"""Tests for SDSS HDF5 -> HATS conversion pipeline."""

import os
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.table import Table as AstropyTable
from dask.distributed import Client
from hats_import import CollectionArguments
from hats_import.pipeline import pipeline_with_client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "sdss"))
from build_parent_sample_hats import (
    BOOL_FEATURES,
    FLOAT_FEATURES,
    FLUX_FEATURES,
    FLUX_FILTERS,
    ArrowTableReader,
    catalog_to_arrow,
)

TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")
HDF5_PATH = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")


@pytest.fixture
def sdss_hdf5():
    if not os.path.exists(HDF5_PATH):
        pytest.skip("SDSS test data not downloaded")
    return HDF5_PATH


@pytest.fixture
def sdss_astropy_table(sdss_hdf5):
    with h5py.File(sdss_hdf5, "r") as f:
        cols = [
            "ra", "dec", "object_id",
            "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
            "spectrum_lsf_sigma", "spectrum_mask",
            *FLOAT_FEATURES, *BOOL_FEATURES, *FLUX_FEATURES,
        ]
        return AstropyTable({k: f[k][:] for k in cols})


@pytest.fixture
def sdss_arrow_table(sdss_astropy_table):
    return catalog_to_arrow(sdss_astropy_table)


@pytest.fixture
def hats_output_dir(sdss_arrow_table, tmp_path):
    output_dir = tmp_path / "hats_out"
    reader = ArrowTableReader([sdss_arrow_table])
    import_args = (
        CollectionArguments(
            output_artifact_name="sdss_test",
            output_path=str(output_dir),
            tmp_dir=str(tmp_path / "tmp"),
        )
        .catalog(
            input_file_list=["0"],
            file_reader=reader,
            ra_column="ra",
            dec_column="dec",
            pixel_threshold=8192,
            lowest_healpix_order=4,
        )
        .add_margin(margin_threshold=10.0, is_default=True)
    )
    with Client(n_workers=1, threads_per_worker=1, processes=False) as client:
        pipeline_with_client(import_args, client)
    return output_dir


class TestCatalogToArrow:
    def test_row_count(self, sdss_arrow_table, sdss_astropy_table):
        assert sdss_arrow_table.num_rows == len(sdss_astropy_table)

    def test_has_required_columns(self, sdss_arrow_table):
        names = sdss_arrow_table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "spectrum" in names

    def test_spectrum_struct(self, sdss_arrow_table):
        spec_type = sdss_arrow_table.schema.field("spectrum").type
        field_names = [spec_type.field(i).name for i in range(spec_type.num_fields)]
        assert set(field_names) == {"flux", "ivar", "lsf_sigma", "lambda", "mask"}

    def test_float_features(self, sdss_arrow_table):
        for f in FLOAT_FEATURES:
            assert f in sdss_arrow_table.schema.names

    def test_flux_features_split(self, sdss_arrow_table):
        for f in FLUX_FEATURES:
            for b in FLUX_FILTERS:
                assert f"{f}_{b}" in sdss_arrow_table.schema.names

    def test_object_id_is_string(self, sdss_arrow_table):
        import pyarrow as pa
        assert sdss_arrow_table.schema.field("object_id").type == pa.string()

    def test_ra_dec_values(self, sdss_arrow_table, sdss_hdf5):
        with h5py.File(sdss_hdf5, "r") as f:
            expected_ra = f["ra"][:]
            expected_dec = f["dec"][:]
        actual_ra = sdss_arrow_table.column("ra").to_numpy()
        actual_dec = sdss_arrow_table.column("dec").to_numpy()
        np.testing.assert_allclose(actual_ra, expected_ra)
        np.testing.assert_allclose(actual_dec, expected_dec)


class TestHATSPipeline:
    def test_output_exists(self, hats_output_dir):
        catalog_dir = hats_output_dir / "sdss_test" / "sdss_test"
        assert (catalog_dir / "hats.properties").exists() or (catalog_dir / "properties").exists()

    def test_parquet_files_exist(self, hats_output_dir):
        dataset_dir = hats_output_dir / "sdss_test" / "sdss_test" / "dataset"
        parquet_files = list(dataset_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

    def test_row_count_preserved(self, hats_output_dir, sdss_arrow_table):
        dataset_dir = hats_output_dir / "sdss_test" / "sdss_test" / "dataset"
        total = sum(
            pq.read_metadata(p).num_rows
            for p in dataset_dir.rglob("*.parquet")
        )
        assert total == sdss_arrow_table.num_rows

    def test_lazy_column_read(self, hats_output_dir):
        dataset_dir = hats_output_dir / "sdss_test" / "sdss_test" / "dataset"
        pf = next(dataset_dir.rglob("*.parquet"))
        full = pq.read_table(pf)
        partial = pq.read_table(pf, columns=["ra", "dec", "Z"])
        assert partial.num_rows == full.num_rows
        assert partial.num_columns == 3
        assert partial.nbytes < full.nbytes

    def test_values_match_source(self, hats_output_dir, sdss_arrow_table):
        dataset_dir = hats_output_dir / "sdss_test" / "sdss_test" / "dataset"
        pf = next(dataset_dir.rglob("*.parquet"))
        hats_table = pq.read_table(pf, columns=["ra", "dec", "Z", "VDISP"])
        src = sdss_arrow_table.select(["ra", "dec", "Z", "VDISP"]).to_pandas().sort_values("ra").reset_index(drop=True)
        out = hats_table.to_pandas().sort_values("ra").reset_index(drop=True)
        np.testing.assert_allclose(out["ra"].values, src["ra"].values)
        np.testing.assert_allclose(out["Z"].values, src["Z"].values)

    def test_margin_catalog_exists(self, hats_output_dir):
        margin_dir = hats_output_dir / "sdss_test" / "sdss_test_10arcs"
        assert margin_dir.exists()
