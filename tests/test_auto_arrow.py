"""Tests for the auto HDF5 → PyArrow converter (mmu.hats_import.auto_arrow_table_from_hdf5).

These tests use synthetic in-memory HDF5 files to verify the converter handles
all the modality types and quirks listed in scripts/hats_configs/SCHEMA_SURVEY.md.
"""

import io
import h5py
import numpy as np
import pyarrow as pa
import pytest

from mmu.hats_import import auto_arrow_table_from_hdf5


def make_hdf5(columns: dict) -> h5py.File:
    """Create an in-memory HDF5 file from a dict of (name -> ndarray)."""
    bio = io.BytesIO()
    f = h5py.File(bio, "w")
    for name, arr in columns.items():
        f.create_dataset(name, data=arr)
    return f


@pytest.fixture
def n():
    return 50


@pytest.fixture
def coords(n):
    rng = np.random.default_rng(42)
    return {
        "ra": rng.uniform(0, 360, n).astype(np.float64),
        "dec": rng.uniform(-90, 90, n).astype(np.float64),
        "object_id": np.array([f"obj_{i:04d}" for i in range(n)], dtype="S10"),
    }


class TestSpectrumDataset:
    """Mimics SDSS/DESI: ra, dec, object_id, spectrum_*, scalar features."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(0)
        spec_len = 100
        cols = {
            **coords,
            "spectrum_flux": rng.normal(0, 1, (n, spec_len)).astype(np.float32),
            "spectrum_ivar": rng.uniform(0.1, 1, (n, spec_len)).astype(np.float32),
            "spectrum_lambda": np.tile(np.linspace(3800, 9200, spec_len), (n, 1)).astype(np.float32),
            "spectrum_lsf_sigma": np.ones((n, spec_len), dtype=np.float32),
            "spectrum_mask": np.zeros((n, spec_len), dtype=bool),
            "Z": rng.uniform(0, 0.5, n).astype(np.float32),
            "ZWARNING": np.zeros(n, dtype=bool),
        }
        return make_hdf5(cols)

    def test_basic_columns(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "object_id" in table.schema.names
        assert "spectrum" in table.schema.names
        assert "Z" in table.schema.names
        assert "ZWARNING" in table.schema.names

    def test_spectrum_struct_fields(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        spec_type = table.schema.field("spectrum").type
        names = [spec_type.field(i).name for i in range(spec_type.num_fields)]
        assert "flux" in names
        assert "ivar" in names
        assert "lambda" in names
        assert "lsf_sigma" in names
        assert "mask" in names

    def test_spectrum_keys_not_duplicated(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        # Raw spectrum_* keys should NOT appear at top level
        assert "spectrum_flux" not in table.schema.names
        assert "spectrum_mask" not in table.schema.names

    def test_object_id_is_string(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert table.schema.field("object_id").type == pa.string()
        # Bytes should be decoded
        assert table.column("object_id")[0].as_py() == "obj_0000"

    def test_row_count(self, h5, n):
        table = auto_arrow_table_from_hdf5(h5)
        assert table.num_rows == n

    def test_row_slice(self, h5):
        table = auto_arrow_table_from_hdf5(h5, n_rows=10)
        assert table.num_rows == 10


class TestImageDataset:
    """Mimics LegacySurvey: image_array (N, bands, H, W)."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(1)
        cols = {
            **coords,
            "image_array": rng.normal(0, 1, (n, 4, 32, 32)).astype(np.float32),
            "FLUX_R": rng.uniform(0.1, 100, n).astype(np.float32),
        }
        return make_hdf5(cols)

    def test_image_flattened(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "image_array" in table.schema.names
        assert "image_array_shape" in table.schema.names

    def test_image_reconstructable(self, h5, n):
        table = auto_arrow_table_from_hdf5(h5)
        flat = np.array(table.column("image_array")[0].as_py())
        shape = table.column("image_array_shape")[0].as_py()
        img = flat.reshape(shape)
        assert img.shape == (4, 32, 32)

    def test_other_features_preserved(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "FLUX_R" in table.schema.names


class TestTimeSeriesDataset:
    """Mimics PLAsTiCC/Kepler: 2D arrays for time/flux/flux_err."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(2)
        seq_len = 50
        cols = {
            **coords,
            "time": rng.uniform(0, 1000, (n, seq_len)).astype(np.float64),
            "flux": rng.normal(0, 1, (n, seq_len)).astype(np.float32),
            "flux_err": rng.uniform(0.01, 0.1, (n, seq_len)).astype(np.float32),
            "redshift": rng.uniform(0, 2, n).astype(np.float32),
        }
        return make_hdf5(cols)

    def test_time_series_columns(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "time" in table.schema.names
        assert "flux" in table.schema.names
        assert "flux_err" in table.schema.names

    def test_time_series_is_list(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert pa.types.is_list(table.schema.field("flux").type)

    def test_first_lightcurve(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        flux = np.array(table.column("flux")[0].as_py())
        assert flux.shape == (50,)


class TestTabularDataset:
    """Mimics AllWise/2MASS: only scalar columns, no spectrum/image."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(3)
        cols = {
            **coords,
            "w1mpro": rng.uniform(8, 18, n).astype(np.float32),
            "w2mpro": rng.uniform(8, 18, n).astype(np.float32),
            "w1snr": rng.uniform(0, 100, n).astype(np.float32),
            "n_obs": rng.integers(1, 100, n).astype(np.int32),
        }
        return make_hdf5(cols)

    def test_scalar_features_pass_through(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "w1mpro" in table.schema.names
        assert "n_obs" in table.schema.names
        assert "spectrum" not in table.schema.names

    def test_dtypes_preserved_or_widened(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        # w1mpro is float32, should remain numeric
        assert pa.types.is_floating(table.schema.field("w1mpro").type)


class TestRADecAliases:
    """Verify alias resolution for RA/Dec column names."""

    def test_uppercase_ra_dec(self, n):
        h5 = make_hdf5({
            "RA": np.random.rand(n).astype(np.float64) * 360,
            "DEC": np.random.rand(n).astype(np.float64) * 180 - 90,
            "object_id": np.array([str(i) for i in range(n)], dtype="S10"),
        })
        table = auto_arrow_table_from_hdf5(h5)
        # Output should always be lowercase ra/dec
        assert "ra" in table.schema.names
        assert "dec" in table.schema.names
        assert "RA" not in table.schema.names
        assert "DEC" not in table.schema.names

    def test_decl_alias(self, n):
        h5 = make_hdf5({
            "ra": np.random.rand(n).astype(np.float64) * 360,
            "decl": np.random.rand(n).astype(np.float64) * 180 - 90,
            "object_id": np.array([str(i) for i in range(n)], dtype="S10"),
        })
        table = auto_arrow_table_from_hdf5(h5)
        assert "dec" in table.schema.names
        assert "decl" not in table.schema.names

    def test_missing_radec_raises(self, n):
        h5 = make_hdf5({
            "foo": np.random.rand(n).astype(np.float64),
            "bar": np.random.rand(n).astype(np.float64),
        })
        with pytest.raises(ValueError, match="RA/Dec"):
            auto_arrow_table_from_hdf5(h5)


class TestSpectrum3DSkip:
    """DESI has spectrum_lsf with shape (N, 11, 7781) — should be skipped, not crash."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(7)
        spec_len = 30
        n_lsf = 11
        cols = {
            **coords,
            "spectrum_flux": rng.normal(0, 1, (n, spec_len)).astype(np.float32),
            "spectrum_ivar": rng.uniform(0.1, 1, (n, spec_len)).astype(np.float32),
            "spectrum_lambda": np.tile(np.linspace(3800, 9200, spec_len), (n, 1)).astype(np.float32),
            "spectrum_lsf_sigma": np.ones((n, spec_len), dtype=np.float32),
            "spectrum_mask": np.zeros((n, spec_len), dtype=bool),
            # 3D resolution matrix - should be silently skipped
            "spectrum_lsf": rng.normal(0, 1, (n, n_lsf, spec_len)).astype(np.float32),
        }
        return make_hdf5(cols)

    def test_does_not_crash(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "spectrum" in table.schema.names

    def test_3d_field_not_in_struct(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        spec_type = table.schema.field("spectrum").type
        names = [spec_type.field(i).name for i in range(spec_type.num_fields)]
        assert "lsf" not in names  # the 3D one
        assert "lsf_sigma" in names  # the 2D one

    def test_3d_not_at_top_level_either(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "spectrum_lsf" not in table.schema.names


class TestStringColumns:
    def test_byte_strings_decoded(self, n, coords):
        h5 = make_hdf5({
            **coords,
            "obj_class": np.array(["GALAXY", "STAR", "QSO"] * (n // 3 + 1), dtype="S6")[:n],
        })
        table = auto_arrow_table_from_hdf5(h5)
        assert table.column("obj_class")[0].as_py() == "GALAXY"
        assert table.schema.field("obj_class").type == pa.string()


class TestColumnDispatchErrors:
    """Verify _column_to_pyarrow raises with column name on bad input."""

    def test_object_dtype_with_ndim2_raises(self):
        from mmu.hats_import import _column_to_pyarrow
        arr = np.empty((5, 2), dtype=object)
        with pytest.raises(ValueError, match="myfield.*object dtype.*ndim=2"):
            _column_to_pyarrow(arr, name="myfield")

    def test_object_dtype_with_array_elements_raises(self):
        from mmu.hats_import import _column_to_pyarrow
        arr = np.empty(3, dtype=object)
        for i in range(3):
            arr[i] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="object dtype with element type ndarray"):
            _column_to_pyarrow(arr, name="oddcol")


class TestImageColumn4D:
    """Auto-converter should handle 4D image cubes (e.g., LegacySurvey image_array)."""

    @pytest.fixture
    def h5(self, coords, n):
        rng = np.random.default_rng(8)
        cols = {
            **coords,
            # 4D image cube: (N, bands, H, W)
            "image_array": rng.normal(0, 1, (n, 4, 16, 16)).astype(np.float32),
            # 3D mask
            "image_mask": rng.integers(0, 2, (n, 16, 16)).astype(np.uint8),
        }
        return make_hdf5(cols)

    def test_4d_image_column_present(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "image_array" in table.schema.names
        assert "image_array_shape" in table.schema.names

    def test_4d_image_reconstructable(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        flat = np.array(table.column("image_array")[0].as_py(), dtype=np.float32)
        shape = list(table.column("image_array_shape")[0].as_py())
        assert shape == [4, 16, 16]
        img = flat.reshape(shape)
        assert img.shape == (4, 16, 16)

    def test_3d_mask_also_handled(self, h5):
        table = auto_arrow_table_from_hdf5(h5)
        assert "image_mask" in table.schema.names
        assert "image_mask_shape" in table.schema.names
