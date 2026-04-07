"""Fast unit tests for scripts/desi/build_parent_sample_hats.py."""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "desi", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_desi_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _make_camera_hdus(prefix: str, n_pixels: int, n_fibers: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(3000.0, 10000.0, n_pixels).astype(np.float32)
    flux = rng.normal(0, 1, (n_fibers, n_pixels)).astype(np.float32)
    ivar = rng.uniform(0.1, 1, (n_fibers, n_pixels)).astype(np.float32)
    mask = np.zeros((n_fibers, n_pixels), dtype=np.uint32)
    return [
        fits.ImageHDU(wavelength, name=f"{prefix}_WAVELENGTH"),
        fits.ImageHDU(flux, name=f"{prefix}_FLUX"),
        fits.ImageHDU(ivar, name=f"{prefix}_IVAR"),
        fits.ImageHDU(mask, name=f"{prefix}_MASK"),
    ]


def _write_fake_coadd(path: str, n_fibers: int = 5) -> None:
    """Write a minimal DESI coadd FITS file: FIBERMAP + B/R/Z camera HDUs."""
    rng = np.random.default_rng(7)
    fibermap_cols = [
        fits.Column(name="TARGETID", format="K", array=np.arange(1000, 1000 + n_fibers, dtype=np.int64)),
        fits.Column(name="COADD_FIBERSTATUS", format="J",
                    array=np.array([0, 0, 16777216, 0, 0][:n_fibers], dtype=np.int32)),
        fits.Column(name="TARGET_RA", format="D", array=rng.uniform(0, 360, n_fibers)),
        fits.Column(name="TARGET_DEC", format="D", array=rng.uniform(-90, 90, n_fibers)),
        fits.Column(name="OBJTYPE", format="3A",
                    array=np.array(["TGT", "TGT", "SKY", "TGT", "TGT"][:n_fibers])),
        fits.Column(name="EBV", format="E", array=rng.uniform(0, 0.1, n_fibers).astype(np.float32)),
        fits.Column(name="FLUX_R", format="E", array=rng.uniform(1, 100, n_fibers).astype(np.float32)),
    ]
    fibermap = fits.BinTableHDU.from_columns(fibermap_cols, name="FIBERMAP")

    hdul = fits.HDUList([fits.PrimaryHDU(), fibermap])
    hdul.extend(_make_camera_hdus("B", 50, n_fibers=n_fibers, seed=1))
    hdul.extend(_make_camera_hdus("R", 40, n_fibers=n_fibers, seed=2))
    hdul.extend(_make_camera_hdus("Z", 60, n_fibers=n_fibers, seed=3))
    hdul.writeto(path, overwrite=True)


@pytest.fixture
def fake_raw_root(tmp_path):
    root = tmp_path / "DESI"
    root.mkdir()
    _write_fake_coadd(str(root / "coadd-sv3-backup-10001.fits"))
    _write_fake_coadd(str(root / "coadd-sv3-backup-10002.fits"))
    return str(root)


class TestFindRawFiles:
    def test_finds_files(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        assert len(files) == 2

    def test_max_files(self, fake_raw_root):
        assert len(build.find_raw_files(fake_raw_root, max_files=1)) == 1


class TestReadCoadd:
    def test_basic_columns(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_coadd(files[0])
        assert table is not None
        names = table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "spectrum" in names
        assert "EBV" in names
        assert "FLUX_R" in names

    def test_quality_cuts_drop_bad_fibers(self, fake_raw_root):
        # Of 5 fake fibers, the only one that fails the cuts is fiber index 2,
        # which is BOTH a SKY (OBJTYPE='SKY') AND has nonzero COADD_FIBERSTATUS.
        # So we should keep 4.
        files = build.find_raw_files(fake_raw_root)
        table = build.read_coadd(files[0])
        assert table.num_rows == 4

    def test_spectrum_struct_concatenated(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_coadd(files[0])
        spec_type = table.schema.field("spectrum").type
        names = [spec_type.field(i).name for i in range(spec_type.num_fields)]
        assert {"flux", "ivar", "lambda", "mask"} == set(names)
        # Length should be B(50)+R(40)+Z(60) = 150
        first_spec = table.column("spectrum")[0].as_py()
        assert len(first_spec["flux"]) == 150
        assert len(first_spec["lambda"]) == 150

    def test_object_id_is_string(self, fake_raw_root):
        files = build.find_raw_files(fake_raw_root)
        table = build.read_coadd(files[0])
        assert table.schema.field("object_id").type == pa.string()
