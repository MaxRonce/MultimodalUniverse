"""Unit tests for scripts/jwst/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "jwst", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_jwst_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build_module()

N_ROWS = 6
N_FILTERS = 7
_FILTERS = [b"f090w", b"f115w", b"f150w", b"f200w", b"f277w", b"f356w", b"f444w"]


def _write_fake_jwst_hdf5(path: str, n_rows: int = N_ROWS, n_filters: int = N_FILTERS) -> None:
    """Write a minimal JWST HDF5 file matching the real on-disk layout."""
    rng = np.random.default_rng(99)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("ra", data=rng.uniform(0, 360, n_rows).astype(np.float64))
        f.create_dataset("dec", data=rng.uniform(-30, 30, n_rows).astype(np.float64))
        f.create_dataset("object_id", data=rng.integers(1_000, 9_999, n_rows).astype(np.int64))
        f.create_dataset("healpix", data=np.zeros(n_rows, dtype=np.int64))
        # image_band: (N, N_filters) bytes
        bands = np.array([_FILTERS[:n_filters]] * n_rows, dtype="S5")
        f.create_dataset("image_band", data=bands)
        f.create_dataset(
            "image_flux",
            data=rng.normal(0, 1, (n_rows, n_filters, build.IMAGE_SIZE, build.IMAGE_SIZE)).astype(np.float32),
        )
        f.create_dataset(
            "image_ivar",
            data=rng.uniform(0.1, 10, (n_rows, n_filters, build.IMAGE_SIZE, build.IMAGE_SIZE)).astype(np.float32),
        )
        f.create_dataset(
            "image_mask",
            data=rng.integers(0, 2, (n_rows, n_filters, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.uint8).astype(bool),
        )
        f.create_dataset(
            "image_psf_fwhm",
            data=rng.uniform(0.03, 0.15, (n_rows, n_filters)).astype(np.float32),
        )
        f.create_dataset(
            "image_scale",
            data=np.full((n_rows, n_filters), 0.04, dtype=np.float32),
        )
        for feat in build.FLOAT_FEATURES:
            f.create_dataset(feat, data=rng.uniform(0, 1, n_rows).astype(np.float32))


@pytest.fixture
def fake_raw_root(tmp_path):
    root = str(tmp_path / "jwst")
    hdf5_path = os.path.join(root, "ceers", "healpix=0", "001-of-001.hdf5")
    _write_fake_jwst_hdf5(hdf5_path, n_rows=N_ROWS, n_filters=N_FILTERS)
    return root


class TestConstants:
    def test_image_size(self):
        assert build.IMAGE_SIZE == 96

    def test_float_features(self):
        assert "mag_auto" in build.FLOAT_FEATURES
        assert "flux_radius" in build.FLOAT_FEATURES
        assert len(build.FLOAT_FEATURES) == 7


class TestFindRawFiles:
    def test_finds_hdf5(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 1
        assert files[0].endswith(".hdf5")

    def test_max_files(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root, max_files=0)
        assert len(files) == 0


class TestReadHdf5:
    def test_row_count(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        assert table.num_rows == N_ROWS

    def test_required_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "image"} <= names

    def test_float_features_present(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        names = set(table.schema.names)
        for f in build.FLOAT_FEATURES:
            assert f in names, f"Missing column: {f}"

    def test_object_id_is_string(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        assert table.schema.field("object_id").type == pa.string()

    def test_image_struct_fields(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        img_type = table.schema.field("image").type
        assert pa.types.is_struct(img_type)
        field_names = {f.name for f in img_type}
        assert field_names == {"band", "flux", "ivar", "mask", "psf_fwhm", "scale"}

    def test_image_flux_type(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["flux"] == pa.list_(pa.list_(pa.list_(pa.float32())))

    def test_image_ivar_type(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["ivar"] == pa.list_(pa.list_(pa.list_(pa.float32())))

    def test_image_mask_type(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        img_type = table.schema.field("image").type
        fields = {f.name: f.type for f in img_type}
        assert fields["mask"] == pa.list_(pa.list_(pa.list_(pa.bool_())))

    def test_image_roundtrip_shape(self, fake_raw_root):
        """image[0].flux must decode to (N_FILTERS, IMAGE_SIZE, IMAGE_SIZE)."""
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        first = table.column("image")[0].as_py()
        assert len(first["band"]) == N_FILTERS
        # Bands should be decoded strings
        assert first["band"][0] == "f090w"
        flux = np.asarray(first["flux"], dtype=np.float32)
        assert flux.shape == (N_FILTERS, build.IMAGE_SIZE, build.IMAGE_SIZE)
        mask = np.asarray(first["mask"])
        assert mask.dtype == bool
        assert mask.shape == (N_FILTERS, build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_image_psf_scale_length(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_hdf5(files[0])
        first = table.column("image")[0].as_py()
        assert len(first["psf_fwhm"]) == N_FILTERS
        assert len(first["scale"]) == N_FILTERS

    def test_cone_all_inside(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table_all = build.read_hdf5(files[0])
        table_cone = build.read_hdf5(files[0], ra_center=0.0, dec_center=0.0, radius=180.0)
        assert table_cone.num_rows == table_all.num_rows

    def test_cone_all_outside(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        result = build.read_hdf5(files[0], ra_center=42.0, dec_center=85.0, radius=0.0001)
        assert result is None or result.num_rows < N_ROWS

    def test_ngdeep_six_filters(self, tmp_path):
        """Six-filter files (ngdeep uses f115w-f444w, no f090w) should parse correctly."""
        hdf5_path = str(tmp_path / "ngdeep" / "healpix=0" / "001-of-001.hdf5")
        _write_fake_jwst_hdf5(hdf5_path, n_rows=4, n_filters=6)
        table = build.read_hdf5(hdf5_path)
        first = table.column("image")[0].as_py()
        assert len(first["band"]) == 6
        flux = np.asarray(first["flux"], dtype=np.float32)
        assert flux.shape == (6, build.IMAGE_SIZE, build.IMAGE_SIZE)
