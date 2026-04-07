"""Fast unit tests for scripts/ssl_legacysurvey/build_parent_sample_hats.py."""

import importlib.util
import os

import h5py
import numpy as np
import pyarrow as pa
import pytest


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "ssl_legacysurvey", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_ssl_legacysurvey_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_fake_chunk(path: str, n_rows: int = 8) -> None:
    """Write a tiny images_npix152_*.h5 chunk matching the real Stein-et-al layout."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("inds", data=np.arange(n_rows, dtype=np.int64))
        f.create_dataset("ra", data=rng.uniform(0, 360, n_rows).astype(np.float64))
        f.create_dataset("dec", data=rng.uniform(-90, 90, n_rows).astype(np.float64))
        f.create_dataset("ebv", data=rng.uniform(0, 0.1, n_rows).astype(np.float32))
        f.create_dataset("z_spec", data=rng.uniform(0, 1, n_rows).astype(np.float32))
        f.create_dataset("flux", data=rng.uniform(0, 100, (n_rows, 3)).astype(np.float32))
        f.create_dataset("fiberflux", data=rng.uniform(0, 100, (n_rows, 3)).astype(np.float32))
        f.create_dataset("psfdepth", data=rng.uniform(20, 25, (n_rows, 3)).astype(np.float32))
        f.create_dataset("psfsize", data=rng.uniform(1, 2, (n_rows, 3)).astype(np.float32))
        # (n_rows, 3, 152, 152) is the real shape but we use 16x16 for speed
        f.create_dataset(
            "images",
            data=rng.normal(0, 1, (n_rows, 3, build.IMAGE_SIZE, build.IMAGE_SIZE)).astype(np.float32),
        )


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "ssl"
    root.mkdir()
    _write_fake_chunk(str(root / "images_npix152_000000000_001000000.h5"), n_rows=5)
    _write_fake_chunk(str(root / "images_npix152_001000000_002000000.h5"), n_rows=3)
    return str(root)


class TestFindRawFiles:
    def test_finds_all(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root)) == 2

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadChunk:
    def test_basic_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        names = table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "image" in names
        assert "ebv" in names
        assert "z_spec" in names
        # per-band flux unrolled
        assert "flux_g" in names
        assert "flux_r" in names
        assert "flux_z" in names
        assert "fiberflux_g" in names
        assert "psfdepth_z" in names

    def test_row_count(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        assert table.num_rows == 5

    def test_object_id_is_string(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        assert table.schema.field("object_id").type == pa.string()

    def test_image_struct_fields(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        spec = table.schema.field("image").type
        names = [spec.field(i).name for i in range(spec.num_fields)]
        assert set(names) == {"band", "flux", "psf_fwhm", "scale"}

    def test_image_band_names(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        first = table.column("image")[0].as_py()
        assert first["band"] == ["DES-G", "DES-R", "DES-Z"]

    def test_image_shape_from_schema(self, fake_raw_root):
        """The Array2DExtensionType storage should preserve (152, 152) shape info."""
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0])
        image_type = table.schema.field("image").type
        flux_field = image_type.field("flux").type
        # flux is list<Array2DExtensionType>; check the value type.
        assert pa.types.is_list(flux_field)
        inner = flux_field.value_type
        # Extension types expose .shape on the Array2DExtensionType instance.
        assert hasattr(inner, "shape"), (
            "flux value type should be an Array2DExtensionType with .shape"
        )
        assert tuple(inner.shape) == (build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_max_rows_per_file(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_chunk(files[0], max_rows=2)
        assert table.num_rows == 2
