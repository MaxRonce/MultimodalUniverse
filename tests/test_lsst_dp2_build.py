"""Tests for the restartable Rubin DP2 image ingestion pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

from scripts.lsst_dp2 import build_parent_sample_hats as build
from scripts.lsst_dp2 import download_coadds as download
from scripts.lsst_dp2 import query_catalog as query
from scripts.lsst_dp2 import validate_parent_sample as validate
from scripts.lsst_dp2.common import BANDS, IMAGE_SIZE, coadd_path


def _wcs_header(size: int = 256) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = size / 2 + 0.5
    header["CRPIX2"] = size / 2 + 0.5
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 2.0
    header["CDELT1"] = -0.2 / 3600.0
    header["CDELT2"] = 0.2 / 3600.0
    return header


def _write_maskedimage(path: Path, value: float, size: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size), value, dtype=np.float32)
    variance = np.full((size, size), 4.0, dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.int32)
    mask[size // 2, size // 2] = 1 << 0
    image_hdu = fits.ImageHDU(image, header=_wcs_header(size), name="IMAGE")
    image_hdu.header["BUNIT"] = "nJy"
    mask_hdu = fits.ImageHDU(mask, name="MASK")
    mask_hdu.header["MP_BAD"] = 0
    mask_hdu.header["MP_SAT"] = 1
    mask_hdu.header["MP_NO_DATA"] = 2
    mask_hdu.header["MP_SUSPECT"] = 3
    mask_hdu.header["MP_UNMASKEDNAN"] = 4
    variance_hdu = fits.ImageHDU(variance, header=_wcs_header(size), name="VARIANCE")
    variance_hdu.header["BUNIT"] = "nJy2"
    yy, xx = np.indices((build.PSF_SIZE, build.PSF_SIZE), dtype=np.float32)
    center = (build.PSF_SIZE - 1) / 2
    kernel = np.exp(-0.5 * ((xx - center) ** 2 + (yy - center) ** 2) / 2.0**2)
    kernel /= kernel.sum()
    psf = np.broadcast_to(kernel, (2, 2, build.PSF_SIZE, build.PSF_SIZE)).copy()
    psf_hdu = fits.ImageHDU(psf.astype(np.float32), name="PSF")
    metadata = {
        "image": {"yx0": [0, 0]},
        "psf": {
            "bounds": {
                "grid": {
                    "bbox": {
                        "y": {"start": 0, "stop": size},
                        "x": {"start": 0, "stop": size},
                    },
                    "cell_shape": [size // 2, size // 2],
                }
            }
        },
    }
    payload = np.frombuffer(json.dumps(metadata).encode("ascii"), dtype=np.uint8)
    json_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="JSON", format=f"PB({len(payload)})", array=[payload])
    ], name="JSON")
    fits.HDUList([
        fits.PrimaryHDU(), image_hdu, mask_hdu, variance_hdu, psf_hdu, json_hdu,
    ]).writeto(path)


def _catalog(path: Path) -> Table:
    data = {
        "objectId": np.array([101, 102], dtype=np.int64),
        "coord_ra": np.array([150.0, 150.0001], dtype=np.float64),
        "coord_dec": np.array([2.0, 2.0001], dtype=np.float64),
        "tract": np.array([1234, 1234], dtype=np.int64),
        "patch": np.array([56, 56], dtype=np.int64),
        "refBand": np.array(["i", "r"]),
        "refExtendedness": np.array([1.0, 0.0], dtype=np.float32),
        "detect_isIsolated": np.array([True, False]),
    }
    for i, band in enumerate(BANDS):
        data[f"{band}_ixxPSF"] = np.full(2, 4.0 + i, dtype=np.float32)
        data[f"{band}_iyyPSF"] = np.full(2, 4.0 + i, dtype=np.float32)
        data[f"{band}_ixyPSF"] = np.zeros(2, dtype=np.float32)
        for suffix in build.PHOTOMETRY_SUFFIXES:
            data[f"{band}_{suffix}"] = np.full(2, i + 1, dtype=np.float32)
    table = Table(data)
    table.write(path, format="parquet")
    return table


def _manifest(path: Path, mirror: Path) -> None:
    con = download.connect_manifest(str(path))
    for i, band in enumerate(BANDS):
        image_path = coadd_path(mirror, 1234, 56, band)
        _write_maskedimage(image_path, float(i + 1))
        con.execute(
            """
            INSERT INTO coadds (
                tract, patch, band, ra, dec, output_path, status, attempts,
                bytes, sha256, dataset_id, s_resolution, mask_planes, updated_at
            ) VALUES (?, ?, ?, 150.0, 2.0, ?, 'complete', 1, ?, ?, ?, 0.7, ?, ?)
            """,
            (
                1234, 56, band, str(image_path), image_path.stat().st_size,
                build.sha256_file(image_path), f"ivo://dp2/{band}",
                "BAD,NO_DATA,SAT,SUSPECT,UNMASKEDNAN", download.utcnow(),
            ),
        )
    con.commit()
    con.close()


def test_query_column_discovery_and_polygon():
    available = {
        "objectId", "coord_ra", "coord_dec", "tract", "patch", "refBand",
        "i_psfFlux", "i_ixxPSF", "i_iyyPSF", "i_ixyPSF",
    }
    columns = query.select_columns(available)
    assert columns[:5] == ["objectId", "coord_ra", "coord_dec", "tract", "patch"]
    assert "i_ixxPSF" in columns
    args = query.argparse.Namespace(polygon="0,0 1,0 1,1", ra=None, dec=None, radius_deg=None)
    assert "POLYGON('ICRS', 0.0, 0.0, 1.0, 0.0, 1.0, 1.0)" in query.spatial_predicate(args)
    assert query.build_query(columns, "1=1", None, 4).startswith("SELECT TOP 4 ")


def test_patch_psf_fallback_ignores_invalid_object_moments(tmp_path):
    catalog = _catalog(tmp_path / "objects.parquet")
    catalog["u_ixxPSF"][0] = np.nan
    expected = build.catalog_psf_fwhm(catalog, catalog[1], "u")
    result = build.patch_psf_fwhm(catalog, catalog)
    assert result["u"] == pytest.approx(expected)


def test_manifest_is_restartable(tmp_path):
    catalog_path = tmp_path / "objects.parquet"
    _catalog(catalog_path)
    con = download.connect_manifest(str(tmp_path / "manifest.sqlite"))
    assert download.initialize_tasks(con, str(catalog_path), str(tmp_path / "mirror")) == 6
    assert download.initialize_tasks(con, str(catalog_path), str(tmp_path / "mirror")) == 6
    assert con.execute("SELECT COUNT(*) FROM coadds").fetchone()[0] == 6
    con.close()


def test_manifest_status_does_not_require_token(tmp_path, monkeypatch, capsys):
    catalog_path = tmp_path / "objects.parquet"
    _catalog(catalog_path)
    monkeypatch.delenv("RSP_TOKEN", raising=False)
    result = download.main([
        "--catalog", str(catalog_path),
        "--mirror-root", str(tmp_path / "mirror"),
        "--manifest", str(tmp_path / "manifest.sqlite"),
        "--status-only",
    ])
    assert result == 0
    assert "pending=6" in capsys.readouterr().out


def test_select_sia_record_requires_exact_identity():
    table = Table({
        "lsst_tract": [1234, 1234],
        "lsst_patch": [55, 56],
        "lsst_band": ["i", "i"],
    })
    assert download.select_sia_record(table, 1234, 56, "i") == 1


def test_select_full_product_url_requires_this_semantics():
    datalink = Table({
        "semantics": ["#preview", "#this"],
        "access_url": ["https://example/preview", "https://example/deep-coadd"],
    })
    assert download.select_full_product_url(datalink) == "https://example/deep-coadd"


def test_mask_decode_and_stamp(tmp_path):
    path = tmp_path / "masked.fits"
    _write_maskedimage(path, 3.0)
    with build.open_maskedimage(str(path)) as coadd:
        assert set(build.CORE_REQUIRED_MASK_PLANES) <= set(coadd["mask_mapping"])
        flux, ivar, mask, bits = build.make_stamp(coadd, 150.0, 2.0)
    assert flux.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.all(flux == 3.0)
    assert np.all(ivar[mask] == 0.25)
    assert np.all(ivar[~mask] == 0.0)
    assert np.count_nonzero(bits) == 1
    assert np.count_nonzero(~mask) == 1


def test_dp2_mask_header_decode_and_sat_alias():
    header = fits.Header()
    header["MSKN0000"] = "NO_DATA"
    header["MSKM0000"] = 1
    header["MSKN0003"] = "SATURATED"
    header["MSKM0003"] = 8
    mapping = build.mask_plane_mapping(header)
    assert mapping == {"NO_DATA": 0, "SATURATED": 3}
    bits = np.array([0, 1, 8], dtype=np.int32)
    clean = build.clean_mask_from_bits(bits, mapping)
    assert clean.tolist() == [True, False, False]


def test_end_to_end_parquet_and_hats(tmp_path):
    catalog_path = tmp_path / "objects.parquet"
    _catalog(catalog_path)
    manifest_path = tmp_path / "manifest.sqlite"
    _manifest(manifest_path, tmp_path / "mirror")
    scratch = tmp_path / "scratch"
    output = tmp_path / "hats"

    result = build.main([
        "--catalog", str(catalog_path),
        "--manifest", str(manifest_path),
        "--scratch-dir", str(scratch),
        "--output-root", str(output),
        "--objects-per-shard", "1",
        "--pixel-threshold", "32",
        "--ingest-workers", "1",
        "--verify-checksums",
    ])
    assert result == 0
    shards = sorted(scratch.glob("*.parquet"))
    assert len(shards) == 2
    table = pq.read_table(shards[0])
    image = table.column("image")[0].as_py()
    assert image["band"] == list(BANDS)
    assert np.asarray(image["flux"]).shape == (6, IMAGE_SIZE, IMAGE_SIZE)
    assert np.asarray(image["ivar"]).shape == (6, IMAGE_SIZE, IMAGE_SIZE)
    psf_image = np.asarray(image["psf_image"])
    assert psf_image.shape == (6, build.PSF_SIZE, build.PSF_SIZE)
    assert np.allclose(psf_image.sum(axis=(-2, -1)), 1.0)
    assert table.schema.field("image").type.field("mask_bits").type.value_type.value_type.value_type == build.pa.int32()
    assert all(image["band_present"])
    assert image["dataset_id"][3] == "ivo://dp2/i"
    assert list(output.rglob("hats.properties"))
    assert validate.main([
        "--catalog", str(catalog_path),
        "--manifest", str(manifest_path),
        "--scratch-dir", str(scratch),
        "--hats-root", str(output),
        "--verify-checksums",
    ]) == 0


def test_missing_band_is_explicitly_padded(tmp_path):
    catalog_path = tmp_path / "objects.parquet"
    catalog = _catalog(catalog_path)
    manifest_path = tmp_path / "manifest.sqlite"
    _manifest(manifest_path, tmp_path / "mirror")
    con = sqlite3.connect(manifest_path)
    con.execute("DELETE FROM coadds WHERE band='u'")
    con.commit()
    con.close()
    columns = build.validate_catalog(catalog)
    manifest = build.load_manifest(str(manifest_path))
    rows = build.grouped_rows(catalog, columns)[0]
    build.process_patch(
        catalog, rows, columns, manifest, str(tmp_path / "scratch"), 2,
        build.DEFAULT_REJECT_MASK_PLANES, False,
    )
    table = pq.read_table(next((tmp_path / "scratch").glob("*.parquet")))
    image = table.column("image")[0].as_py()
    assert image["band_present"][0] is False
    assert not np.asarray(image["mask"])[0].any()
    assert not np.asarray(image["ivar"])[0].any()
    assert not np.asarray(image["psf_image"])[0].any()


def test_build_contract_rejects_changed_manifest(tmp_path):
    catalog_path = tmp_path / "objects.parquet"
    _catalog(catalog_path)
    manifest_path = tmp_path / "manifest.sqlite"
    _manifest(manifest_path, tmp_path / "mirror")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build.ensure_build_contract(
        str(scratch), str(catalog_path), str(manifest_path), 64,
        build.DEFAULT_REJECT_MASK_PLANES,
    )
    con = sqlite3.connect(manifest_path)
    con.execute("UPDATE coadds SET sha256='changed' WHERE band='u'")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="contract differs"):
        build.ensure_build_contract(
            str(scratch), str(catalog_path), str(manifest_path), 64,
            build.DEFAULT_REJECT_MASK_PLANES,
        )
