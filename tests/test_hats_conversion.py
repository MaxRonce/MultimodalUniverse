"""Tests for HDF5 -> HATS conversion pipeline (SDSS and DESI)."""

import os

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.table import Table as AstropyTable

from mmu.hats_import import build_arrow_table, write_hats

TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")

# --- SDSS ---

SDSS_HDF5 = os.path.join(TEST_DATA, "sdss", "sdss", "healpix=583", "001-of-001.hdf5")
SDSS_FLOAT = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
SDSS_BOOL = ["ZWARNING"]
SDSS_FLUX = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
SDSS_FILTERS = ["U", "G", "R", "I", "Z"]

# --- DESI ---

DESI_HDF5 = os.path.join(TEST_DATA, "desi", "edr_sv3", "healpix=626", "001-of-001.hdf5")
DESI_FLOAT = [
    "Z", "ZERR", "EBV",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z",
    "FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z",
]
DESI_BOOL = ["ZWARN"]


def _load_hdf5_table(path, float_features, bool_features, flux_features=None):
    """Load an MMU HDF5 file into an astropy Table with required columns."""
    with h5py.File(path, "r") as f:
        cols = [
            "ra", "dec", "object_id",
            "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
            "spectrum_lsf_sigma", "spectrum_mask",
        ]
        cols += float_features + bool_features
        if flux_features:
            cols += flux_features
        return AstropyTable({k: f[k][:] for k in cols})


# --- Fixtures ---

@pytest.fixture
def sdss_hdf5():
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test data not downloaded")
    return SDSS_HDF5


@pytest.fixture
def sdss_astropy_table(sdss_hdf5):
    return _load_hdf5_table(sdss_hdf5, SDSS_FLOAT, SDSS_BOOL, SDSS_FLUX)


@pytest.fixture
def sdss_arrow_table(sdss_astropy_table):
    return build_arrow_table(
        sdss_astropy_table,
        float_features=SDSS_FLOAT,
        bool_features=SDSS_BOOL,
        flux_features=SDSS_FLUX,
        flux_filters=SDSS_FILTERS,
    )


@pytest.fixture(scope="module")
def hats_output_dir(tmp_path_factory):
    """Build the SDSS HATS catalog ONCE for all tests in this module."""
    if not os.path.exists(SDSS_HDF5):
        pytest.skip("SDSS test HDF5 not downloaded")
    table = build_arrow_table(
        _load_hdf5_table(SDSS_HDF5, SDSS_FLOAT, SDSS_BOOL, SDSS_FLUX),
        float_features=SDSS_FLOAT,
        bool_features=SDSS_BOOL,
        flux_features=SDSS_FLUX,
        flux_filters=SDSS_FILTERS,
    )
    output_dir = tmp_path_factory.mktemp("sdss_hats_out")
    write_hats([table], str(output_dir), "sdss_test")
    return output_dir


@pytest.fixture
def desi_hdf5():
    if not os.path.exists(DESI_HDF5):
        pytest.skip("DESI test data not downloaded")
    return DESI_HDF5


@pytest.fixture
def desi_astropy_table(desi_hdf5):
    return _load_hdf5_table(desi_hdf5, DESI_FLOAT, DESI_BOOL)


@pytest.fixture
def desi_arrow_table(desi_astropy_table):
    return build_arrow_table(
        desi_astropy_table,
        float_features=DESI_FLOAT,
        bool_features=DESI_BOOL,
        invert_bool=["ZWARN"],
    )


@pytest.fixture(scope="module")
def desi_hats_dir(tmp_path_factory):
    """Build the DESI HATS catalog ONCE for all tests in this module."""
    if not os.path.exists(DESI_HDF5):
        pytest.skip("DESI test HDF5 not downloaded")
    table = build_arrow_table(
        _load_hdf5_table(DESI_HDF5, DESI_FLOAT, DESI_BOOL),
        float_features=DESI_FLOAT,
        bool_features=DESI_BOOL,
        invert_bool=["ZWARN"],
    )
    output_dir = tmp_path_factory.mktemp("desi_hats_out")
    write_hats([table], str(output_dir), "desi_test")
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
        for f in SDSS_FLOAT:
            assert f in sdss_arrow_table.schema.names

    def test_flux_features_split(self, sdss_arrow_table):
        for f in SDSS_FLUX:
            for b in SDSS_FILTERS:
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


@pytest.mark.slow
class TestHATSPipeline:
    """Integration tests that actually run the hats-import pipeline (slow)."""

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


class TestDESIArrow:
    def test_row_count(self, desi_arrow_table, desi_astropy_table):
        assert desi_arrow_table.num_rows == len(desi_astropy_table)

    def test_has_required_columns(self, desi_arrow_table):
        names = desi_arrow_table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "spectrum" in names

    def test_float_features(self, desi_arrow_table):
        for f in DESI_FLOAT:
            assert f in desi_arrow_table.schema.names

    def test_bool_inverted(self, desi_arrow_table, desi_hdf5):
        """ZWARN=0 means good, so it should be inverted to True."""
        with h5py.File(desi_hdf5, "r") as f:
            raw = f["ZWARN"][0]
        arrow_val = desi_arrow_table.column("ZWARN")[0].as_py()
        assert arrow_val == (not bool(raw))

    def test_ra_dec_values(self, desi_arrow_table, desi_hdf5):
        with h5py.File(desi_hdf5, "r") as f:
            expected_ra = f["ra"][:]
        actual_ra = desi_arrow_table.column("ra").to_numpy()
        np.testing.assert_allclose(actual_ra, expected_ra)


@pytest.mark.slow
class TestDESIHATS:
    """Integration tests that actually run the hats-import pipeline (slow)."""

    def test_output_exists(self, desi_hats_dir):
        catalog_dir = desi_hats_dir / "desi_test" / "desi_test"
        assert (catalog_dir / "hats.properties").exists() or (catalog_dir / "properties").exists()

    def test_row_count_preserved(self, desi_hats_dir, desi_arrow_table):
        dataset_dir = desi_hats_dir / "desi_test" / "desi_test" / "dataset"
        total = sum(
            pq.read_metadata(p).num_rows
            for p in dataset_dir.rglob("*.parquet")
        )
        assert total == desi_arrow_table.num_rows

    def test_lazy_column_read(self, desi_hats_dir):
        dataset_dir = desi_hats_dir / "desi_test" / "desi_test" / "dataset"
        pf = next(dataset_dir.rglob("*.parquet"))
        partial = pq.read_table(pf, columns=["ra", "dec", "Z"])
        full = pq.read_table(pf)
        assert partial.nbytes < full.nbytes

    def test_loader_works(self, desi_hats_dir):
        """Verify the HATS dataset loader works with DESI data."""
        from mmu.data import HATSDataset

        catalog_dir = str(desi_hats_dir / "desi_test" / "desi_test")
        ds = HATSDataset(catalog_dir, columns=["ra", "dec", "Z", "object_id"])
        assert len(ds) > 0
        item = ds[0]
        assert "ra" in item
        assert "Z" in item
