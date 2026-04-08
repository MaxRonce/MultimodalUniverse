"""Fast unit tests for scripts/manga/build_parent_sample_hats.py.

The full process_cube path needs real LOGCUBE/MAPS FITS files which are
hard to fabricate, so we test the pure-Python helpers (path construction,
padding, struct assembly) and exercise build_table with hand-rolled fake
records that match the expected dict shape.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "manga", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_manga_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


class TestPathConstruction:
    def test_cube_path(self):
        p = build.cube_path("/data", "8485-1901")
        assert p.endswith("dr17/manga/spectro/redux/v3_1_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz")

    def test_maps_path(self):
        p = build.maps_path("/data", "8485-1901")
        assert "spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/8485/1901" in p
        assert p.endswith("manga-8485-1901-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz")


class TestPadSpatial:
    def test_pad_2d(self):
        arr = np.zeros((40, 40), dtype=np.float32)
        out = build._pad_spatial(arr)
        assert out.shape == (96, 96)

    def test_pad_3d(self):
        arr = np.ones((4563, 40, 40), dtype=np.float32)
        out = build._pad_spatial(arr)
        assert out.shape == (4563, 96, 96)
        # Center should still be 1, edges 0
        assert out[0, 48, 48] == 1
        assert out[0, 0, 0] == 0

    def test_pad_already_full(self):
        arr = np.ones((96, 96), dtype=np.float32)
        out = build._pad_spatial(arr)
        assert out.shape == (96, 96)


def _fake_record(plateifu="8485-1901", n_maps=3):
    """Build a synthetic record matching the dict shape produced by process_cube."""
    nspaxels = build.IMAGE_SIZE * build.IMAGE_SIZE
    nwave = 10  # tiny for tests
    spaxels = {
        "flux":   np.ones((nspaxels, nwave), dtype=np.float32),
        "ivar":   np.ones((nspaxels, nwave), dtype=np.float32),
        "mask":   np.zeros((nspaxels, nwave), dtype=np.int64),
        "lsf":    np.ones((nspaxels, nwave), dtype=np.float32),
        "lambda": np.tile(np.linspace(3500, 9000, nwave, dtype=np.float32), (nspaxels, 1)),
        "x": np.arange(nspaxels, dtype=np.int8),
        "y": np.arange(nspaxels, dtype=np.int8),
        "spaxel_idx": np.arange(nspaxels, dtype=np.int16),
        "flux_units":   ["1e-17 erg/s/cm^2/Å"] * nspaxels,
        "lambda_units": ["Angstrom"] * nspaxels,
        "skycoo_x":     np.zeros(nspaxels, dtype=np.float32),
        "skycoo_y":     np.zeros(nspaxels, dtype=np.float32),
        "ellcoo_r":     np.zeros(nspaxels, dtype=np.float32),
        "ellcoo_rre":   np.zeros(nspaxels, dtype=np.float32),
        "ellcoo_rkpc":  np.zeros(nspaxels, dtype=np.float32),
        "ellcoo_theta": np.zeros(nspaxels, dtype=np.float32),
        "skycoo_units":      ["arcsec"] * nspaxels,
        "ellcoo_r_units":    ["arcsec"] * nspaxels,
        "ellcoo_rre_units":  [""] * nspaxels,
        "ellcoo_rkpc_units": ["kpc"] * nspaxels,
        "ellcoo_theta_units": ["degree"] * nspaxels,
    }
    images = {
        "filter": list(build.BANDS),
        "flux": np.ones((build.N_BANDS, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
        "flux_units": ["nanomaggies/pixel"] * build.N_BANDS,
        "psf": np.ones((build.N_BANDS, build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
        "psf_units": ["nanomaggies/pixel"] * build.N_BANDS,
        "scale": [build.SPAXEL_SIZE_ARCSEC] * build.N_BANDS,
        "scale_units": ["arcsec"] * build.N_BANDS,
    }
    maps = []
    for i in range(n_maps):
        maps.append({
            "group": f"group_{i}",
            "label": f"label_{i}",
            "flux": np.ones((build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
            "ivar": np.ones((build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
            "mask": np.zeros((build.IMAGE_SIZE, build.IMAGE_SIZE), dtype=np.float32),
            "array_units": "unit",
        })
    return {"plateifu": plateifu, "spaxels": spaxels, "images": images, "maps": maps}


def _fake_catalog(plateifus=("8485-1901",)):
    return Table({
        "plateifu": np.array(list(plateifus)),
        "ifura": np.full(len(plateifus), 150.0, dtype=np.float64),
        "ifudec": np.full(len(plateifus), 2.0, dtype=np.float64),
        "nsa_z": np.full(len(plateifus), 0.05, dtype=np.float32),
    })


class TestBuildTable:
    def test_top_level_columns(self):
        rec = _fake_record()
        cat = _fake_catalog(["8485-1901"])
        table = build.build_table([rec], cat)
        assert table.num_rows == 1
        names = set(table.schema.names)
        assert {"ra", "dec", "object_id", "z", "spaxel_size",
                "spaxel_size_units", "spaxels", "images", "maps"} <= names

    def test_spaxels_struct_subfields(self):
        table = build.build_table([_fake_record()], _fake_catalog(["8485-1901"]))
        spx = table.schema.field("spaxels").type
        assert pa.types.is_struct(spx)
        names = {f.name for f in spx}
        assert {"flux", "ivar", "mask", "lsf", "lambda", "x", "y",
                "spaxel_idx", "flux_units", "lambda_units",
                "skycoo_x", "skycoo_y", "ellcoo_r", "ellcoo_rre",
                "ellcoo_rkpc", "ellcoo_theta"} <= names

    def test_images_struct_subfields(self):
        table = build.build_table([_fake_record()], _fake_catalog(["8485-1901"]))
        ims = table.schema.field("images").type
        assert pa.types.is_struct(ims)
        names = {f.name for f in ims}
        assert names == {"filter", "flux", "flux_units", "psf", "psf_units", "scale", "scale_units"}

    def test_maps_struct_subfields(self):
        table = build.build_table([_fake_record(n_maps=5)], _fake_catalog(["8485-1901"]))
        mps = table.schema.field("maps").type
        assert pa.types.is_struct(mps)
        names = {f.name for f in mps}
        assert names == {"group", "label", "flux", "ivar", "mask", "array_units"}

    def test_first_row_shapes(self):
        rec = _fake_record(n_maps=3)
        table = build.build_table([rec], _fake_catalog(["8485-1901"]))
        nspaxels = build.IMAGE_SIZE * build.IMAGE_SIZE
        spx = table.column("spaxels")[0].as_py()
        assert len(spx["x"]) == nspaxels
        assert len(spx["flux"]) == nspaxels
        # Per-spaxel flux is 10 elements (the tiny test nwave).
        assert len(spx["flux"][0]) == 10

        ims = table.column("images")[0].as_py()
        assert ims["filter"] == build.BANDS
        flux = np.asarray(ims["flux"])
        assert flux.shape == (build.N_BANDS, build.IMAGE_SIZE, build.IMAGE_SIZE)

        mps = table.column("maps")[0].as_py()
        assert len(mps["group"]) == 3
        assert mps["group"] == ["group_0", "group_1", "group_2"]
        flux_maps = np.asarray(mps["flux"])
        assert flux_maps.shape == (3, build.IMAGE_SIZE, build.IMAGE_SIZE)

    def test_drops_records_not_in_catalog(self):
        rec = _fake_record(plateifu="9999-9999")  # not in catalog
        with pytest.raises(RuntimeError, match="No records joined"):
            build.build_table([rec], _fake_catalog(["8485-1901"]))
