"""Tiny synthetic-FITS checks; no network or HATS cluster required."""

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from scripts.jades.audit_catalog import summarize_snr
from scripts.jades.build_parent_sample_hats import (
    HST_BANDS,
    JWST_BANDS,
    apparent_mag_from_njy,
    flux_conversion,
    _pixel_cutout,
    process_field,
    variance_to_ivar,
)
from scripts.jades.make_manifest import make_manifest
from scripts.jades.inspect_cutouts import inspect


def test_real_inverse_variance_only():
    x = np.array([[2.0, 0.0, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(variance_to_ivar(x, "sigma"), [[0.25, 0, 0]])
    np.testing.assert_allclose(variance_to_ivar(x, "variance"), [[0.5, 0, 0]])
    np.testing.assert_allclose(variance_to_ivar(x, "ivar"), [[2, 0, 0]])
    with pytest.raises(ValueError):
        variance_to_ivar(x, "synthetic")


def test_unit_conversion_requires_pixel_area():
    with pytest.raises(ValueError, match="PIXAR_SR"):
        flux_conversion(fits.Header(), "MJy/sr")
    header = fits.Header({"PIXAR_SR": 1e-14})
    assert flux_conversion(header, "MJy/sr") == pytest.approx(10.0)


def test_instrument_separation_and_ab_magnitude():
    assert len(JWST_BANDS) == 26
    assert len(HST_BANDS) == 9
    assert not (JWST_BANDS & HST_BANDS)  # Distinct instrument namespace (see below).
    assert apparent_mag_from_njy([1000.0])[0] == pytest.approx(23.9)
    assert np.isnan(apparent_mag_from_njy([0.0])[0])


def _wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [53.0, -27.0]
    wcs.wcs.crpix = [100, 100]
    wcs.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _catalog(path):
    id_col = fits.Column(name="ID", format="J", array=np.array([1, 2], dtype=np.int32))
    ra_col = fits.Column(name="RA", format="D", array=np.array([53.0, 53.0]))
    dec_col = fits.Column(name="DEC", format="D", array=np.array([-27.0, -27.0]))
    a_col = fits.Column(name="A", format="E", array=np.array([0.1, 0.2]))
    b_col = fits.Column(name="B", format="E", array=np.array([0.08, 0.18]))
    f_col = fits.Column(name="F277W_CIRC3", format="E", array=np.array([10.0, 2.0]))
    e_col = fits.Column(name="F277W_CIRC3_e", format="E", array=np.array([2.0, 2.0]))
    t_col = fits.Column(name="F277W_TEXP", format="E", array=np.array([200., 200.]))
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([id_col, ra_col, dec_col, a_col, b_col], name="SIZE"),
        fits.BinTableHDU.from_columns([id_col, ra_col, dec_col, f_col, e_col], name="CIRC"),
        fits.BinTableHDU.from_columns([id_col, ra_col, dec_col, t_col], name="FLAG"),
    ]).writeto(path)


def test_snr_and_cutouts(tmp_path):
    catalog = tmp_path / "cat.fits"
    mosaic = tmp_path / "mosaic.fits"
    _catalog(catalog)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((200, 200), np.float32), _wcs().to_header(), name="SCI"),
        fits.ImageHDU(np.full((200, 200), 2.0, np.float32), name="ERR"),
    ]).writeto(mosaic)
    result = summarize_snr([catalog], ["F277W"])
    assert result["cat"]["snr_ge_5"]["at_least_1_band"] == 1
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    entries = [{"path": "mosaic.fits", "field": "GOODS-S", "band": "F277W",
                "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                "uncertainty_kind": "sigma", "unit": "nJy"}]
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch, 1, None) == 2
    import pyarrow.parquet as pq
    table = pq.read_table(sorted(scratch.glob("*.parquet"))[0])
    image = table.column("image")[0].as_py()
    assert image["band"] == ["F277W"]
    assert image["ivar"][0][0][0] == pytest.approx(0.25)
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch, 1, None) == 2


def test_native_cutout_matches_astropy_partial_edges():
    from astropy.nddata import Cutout2D

    data = np.arange(200, dtype=np.float32).reshape(10, 20)
    for x, y in [(0, 0), (0.5, 3.5), (5, 5), (18.9, 9), (19.5, 1.1)]:
        actual = _pixel_cutout(data, x, y, np.nan, 6)
        expected = Cutout2D(data, (x, y), (6, 6), mode="partial", fill_value=np.nan).data
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert _pixel_cutout(data, -20, 5, np.nan, 6) is None


def test_miri_photometry_mosaic_overlap_photoz_and_checkpoint(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _catalog(catalog)
    ids = fits.Column(name="ID", format="J", array=np.array([1, 2], dtype=np.int32))
    with fits.open(catalog, mode="update") as hdus:
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="F770W_CIRC3", format="E", array=[101., 202.]),
            fits.Column(name="F770W_CIRC3_e", format="E", array=[4., 5.]),
        ], name="MIRI"))
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="z_spec", format="E", array=[5.1, -99.]),
            fits.Column(name="z_peak", format="E", array=[5.2, 4.7]),
        ], name="PHOTOZ"))
    entries = []
    for i, sigma in enumerate((4., 1.)):
        path = tmp_path / f"miri{i}.fits"
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(np.full((200, 200), 3 + i, np.float32), _wcs().to_header(), name="SCI"),
            fits.ImageHDU(np.full((200, 200), sigma, np.float32), name="ERR"),
        ]).writeto(path)
        entries.append({"path": path.name, "field": "GOODS-S", "band": "F770W",
                        "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                        "uncertainty_kind": "sigma", "unit": "nJy"})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=2, rows_per_file=1, image_size=8) == 2
    import pyarrow.parquet as pq
    first = pq.read_table(scratch / "GOODS-S_00000.parquet").to_pylist()[0]
    assert first["image"]["band"] == ["F770W"]
    assert first["image_sources"] == ["miri1.fits"]
    assert first["photometry"]["flux_circ3_njy"] == [101.0]
    assert first["photometry"]["flux_err_circ3_njy"] == [4.0]
    assert first["image"]["ivar"][0][0][0] == pytest.approx(1.)
    assert first["redshift"]["z_peak"] == pytest.approx(5.2)
    second = pq.read_table(scratch / "GOODS-S_00001.parquet").to_pylist()[0]
    assert np.isnan(second["redshift"]["z_spec"])

    # Simulate a crash after the first committed Parquet shard: the second is
    # present but not checkpointed, and must be safely rebuilt on resume.
    (scratch / "GOODS-S.done").unlink()
    progress = scratch / "GOODS-S.progress.json"
    checkpoint = json.loads(progress.read_text())
    checkpoint.update(next_index=1, next_shard=1, count=1)
    progress.write_text(json.dumps(checkpoint))
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=2, rows_per_file=1, image_size=8) == 2
    assert len(list(scratch.glob("GOODS-S_*.parquet"))) == 2
    with pytest.raises(ValueError, match="configuration differs"):
        process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                      chunk_size=1, max_objects=1, rows_per_file=1, image_size=8)
    report, _ = inspect(scratch)
    assert report["total_objects"] == 2
    assert report["sample_band_occurrences"]["F770W"] == 2


def test_miri_focused_sample_filters_catalog_exposures(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _catalog(catalog)
    with fits.open(catalog, mode="update") as hdus:
        old_flag = hdus["FLAG"]
        columns = old_flag.columns + fits.ColDefs([
            fits.Column(name="F770W_TEXP", format="E", array=[0., 100.])])
        hdus["FLAG"] = fits.BinTableHDU.from_columns(columns, name="FLAG")
    mosaic = tmp_path / "miri.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((200, 200), np.float32), _wcs().to_header(), name="SCI"),
        fits.ImageHDU(np.ones((200, 200), np.float32), name="ERR"),
    ]).writeto(mosaic)
    scratch = tmp_path / "scratch_miri"
    scratch.mkdir()
    entries = [{"field": "GOODS-S", "band": "F770W", "path": "miri.fits",
                "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                "uncertainty_kind": "sigma", "unit": "nJy"}]
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=1, image_size=8,
                         sample_mode="miri") == 1
    import pyarrow.parquet as pq
    row = pq.read_table(scratch / "GOODS-S_00000.parquet").to_pylist()[0]
    assert row["object_id"] == "GOODS-S:2"


def test_manifest_detects_correct_products_and_requires_complete_set(tmp_path):
    for field in ("GOODS-S", "GOODS-N"):
        folder = tmp_path / "catalogs"
        folder.mkdir(exist_ok=True)
        (folder / f"{field}.fits").write_bytes(b"catalog-placeholder")
    folder = tmp_path / "mosaics" / "GOODS-S"
    folder.mkdir(parents=True)
    filename = "hlsp_jades_jwst_nircam_goods-s_f277w_v5.0_drz.fits"
    header = _wcs().to_header()
    header["BUNIT"] = "MJy/sr"
    header["PIXAR_SR"] = 1e-14
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((10, 10), np.float32), header, name="SCI"),
        fits.ImageHDU(np.ones((10, 10), np.float32), fits.Header({"BUNIT": "MJy/sr"}), name="ERR"),
    ]).writeto(folder / filename)
    manifest = make_manifest(tmp_path, allow_partial=True)
    assert len(manifest["catalogs"]) == 2
    assert manifest["mosaics"][0]["band"] == "F277W"
    assert manifest["mosaics"][0]["uncertainty_kind"] == "sigma"
    with pytest.raises(ValueError, match="Unexpected input counts"):
        make_manifest(tmp_path)
