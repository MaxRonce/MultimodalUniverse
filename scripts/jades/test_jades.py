"""Tiny synthetic-FITS checks; no network or HATS cluster required."""

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from scripts.jades.audit_catalog import summarize_snr
from scripts.jades.build_parent_sample_hats import (
    flux_conversion,
    process_field,
    variance_to_ivar,
)


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
