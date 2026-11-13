"""Tiny synthetic-FITS checks; no network or HATS cluster required."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
    shard_directory,
    variance_to_ivar,
    verify_completed_shards,
)
from scripts.jades.make_manifest import make_manifest
from scripts.jades.inspect_cutouts import inspect
from scripts.jades import build_parent_sample_hats as builder


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


def _wcs(scale_arcsec=0.03):
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [53.0, -27.0]
    wcs.wcs.crpix = [100, 100]
    wcs.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
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
    assert image["band"] == ["f277w"]
    assert image["ivar"][0][0][0] == pytest.approx(0.25)
    assert image["psf_fwhm"] == [None]  # Never fabricate an unmeasured PSF.
    assert image["scale"] == pytest.approx([0.03])
    assert table.column("phot_f277w_circ3_flux_njy")[0].as_py() == pytest.approx(10.0)
    assert "photometry" not in table.column_names
    assert "redshift" not in table.column_names
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch, 1, None) == 2


def test_native_cutout_matches_astropy_partial_edges():
    from astropy.nddata import Cutout2D

    data = np.arange(200, dtype=np.float32).reshape(10, 20)
    for x, y in [(0, 0), (0.5, 3.5), (5, 5), (18.9, 9), (19.5, 1.1)]:
        actual = _pixel_cutout(data, x, y, np.nan, 6)
        expected = Cutout2D(data, (x, y), (6, 6), mode="partial", fill_value=np.nan).data
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert _pixel_cutout(data, -20, 5, np.nan, 6) is None


def test_partial_mosaic_without_covered_galaxy_center_is_not_a_band(tmp_path):
    catalog = tmp_path / "catalog.fits"
    mosaic = tmp_path / "edge.fits"
    _catalog(catalog)
    wcs = _wcs()
    wcs.wcs.crpix = [-1, 6]  # Galaxy centroid lies just left of the mosaic.
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((10, 10), np.float32), wcs.to_header(), name="SCI"),
        fits.ImageHDU(np.ones((10, 10), np.float32), name="ERR"),
    ]).writeto(mosaic)
    scratch = tmp_path / "edge_scratch"
    scratch.mkdir()
    entries = [{"field": "GOODS-S", "band": "F277W", "path": mosaic.name,
                "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                "uncertainty_kind": "sigma", "unit": "nJy"}]
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=2, image_size=8) == 0
    assert list(scratch.glob("*.parquet")) == []


def test_distributed_shards_cover_each_source_once_and_validate_completion(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.fits"
    mosaic = tmp_path / "nircam.fits"
    _catalog(catalog)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((200, 200), np.float32), _wcs().to_header(), name="SCI"),
        fits.ImageHDU(np.ones((200, 200), np.float32), name="ERR"),
    ]).writeto(mosaic)
    root = tmp_path / "distributed"
    entries = [{"field": "GOODS-S", "band": "F277W", "path": "nircam.fits",
                "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                "uncertainty_kind": "sigma", "unit": "nJy"}]
    for shard_index in range(3):
        shard = shard_directory(root, shard_index, 3)
        shard.mkdir(parents=True)
        for field in ("GOODS-S", "GOODS-N"):
            assert process_field(field, catalog, entries, tmp_path, shard,
                                 chunk_size=1, max_objects=None, rows_per_file=1,
                                 image_size=8, num_shards=3, shard_index=shard_index) in (0, 1)
    summary = verify_completed_shards(root, ["GOODS-S", "GOODS-N"], num_shards=3)
    assert summary["total_objects"] == 4
    assert summary["field_counts"] == {"GOODS-S": 2, "GOODS-N": 2}
    report, samples = inspect(root, max_samples=4, expected_size=8)
    assert report["sampled_objects"] == 4
    assert {r["object_id"] for r in samples} == {
        "GOODS-S:1", "GOODS-S:2", "GOODS-N:1", "GOODS-N:2",
    }

    # CLI ingest-only must consume the recursive shard tree, not re-extract
    # the catalog. Mock the HATS package (not needed for extraction tests).
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "release": "JADES_DR5",
        "catalogs": [{"field": f, "path": catalog.name} for f in ("GOODS-S", "GOODS-N")],
        "mosaics": [],
    }))
    calls = []
    fake_importer = ModuleType("mmu.hats_import")
    fake_importer.write_hats_from_parquet_dir = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "mmu.hats_import", fake_importer)
    monkeypatch.setattr(sys, "argv", [
        "build_parent_sample_hats", "--instrument", "jwst", "--manifest", str(manifest),
        "--raw-root", str(tmp_path), "--scratch-dir", str(root),
        "--output-root", str(tmp_path / "hats"), "--num-shards", "3", "--ingest-only",
    ])
    builder.main()
    assert len(calls) == 1
    assert calls[0][0] == (str(root),)
    assert calls[0][1]["catalog_name"] == "jades_jwst"
    assert calls[0][1]["highest_healpix_order"] == 13
    assert calls[0][1]["expected_total_rows"] == 4
    hats_output = tmp_path / "hats"
    hats_output.mkdir()
    (hats_output / "user_data.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="already contains data"):
        builder.main()
    assert (hats_output / "user_data.txt").read_text() == "preserve"

    # Resuming a nonempty output requires an existing staging identity.
    resume_tmp = tmp_path / "resume_tmp"
    monkeypatch.setattr(sys, "argv", [
        "build_parent_sample_hats", "--instrument", "jwst", "--manifest", str(manifest),
        "--raw-root", str(tmp_path), "--scratch-dir", str(root),
        "--output-root", str(hats_output), "--num-shards", "3", "--ingest-only",
        "--tmp-dir", str(resume_tmp), "--hats-resume", "--hats-mapping-retries", "1",
    ])
    with pytest.raises(FileExistsError, match="already contains data"):
        builder.main()
    identity = resume_tmp / "hats_import_jades_jwst_resume" / "_mmu_resume_identity.json"
    identity.parent.mkdir(parents=True)
    identity.write_text("{}")
    builder.main()  # The mocked importer receives the configuration to verify.
    assert calls[-1][1]["resumable"] is True
    assert calls[-1][1]["mapping_retries"] == 1

    missing_done = shard_directory(root, 1, 3) / "GOODS-N.done"
    missing_done.unlink()
    with pytest.raises(RuntimeError, match="Incomplete JADES extraction"):
        verify_completed_shards(root, ["GOODS-S", "GOODS-N"], num_shards=3)


def test_miri_photometry_mosaic_overlap_photoz_and_checkpoint(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _catalog(catalog)
    ids = fits.Column(name="ID", format="J", array=np.array([1, 2], dtype=np.int32))
    with fits.open(catalog, mode="update") as hdus:
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="F770W_CIRC3", format="E", array=[101., 202.]),
            fits.Column(name="F770W_CIRC3_e", format="E", array=[4., 5.]),
            fits.Column(name="F770W_KRON", format="E", array=[131., 231.]),
            fits.Column(name="F770W_KRON_e", format="E", array=[7., 8.]),
            fits.Column(name="F1280W_CIRC3", format="E", array=[33., np.nan]),
            fits.Column(name="F1280W_CIRC3_e", format="E", array=[2., -99.]),
        ], name="MIRI"))
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="z_spec", format="E", array=[5.1, -99.]),
            fits.Column(name="z_a", format="E", array=[5.3, 4.8]),
            fits.Column(name="z_peak", format="E", array=[5.2, 4.7]),
            fits.Column(name="l68", format="E", array=[4.9, 4.1]),
            fits.Column(name="u68", format="E", array=[5.5, 5.2]),
            fits.Column(name="z_bins", format="22D", array=np.tile(np.linspace(0, 11, 22), (2, 1))),
            fits.Column(name="Prob_z_bins", format="22D", array=np.tile(np.ones(22) / 22, (2, 1))),
        ], name="PHOTOZ"))
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="z_a", format="E", array=[5.6, 4.6]),
            fits.Column(name="z_peak", format="E", array=[5.4, 4.5]),
        ], name="PHOTOZ_KRON"))
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="F277W_KRON", format="E", array=[16., 3.]),
            fits.Column(name="F277W_KRON_e", format="E", array=[2., 1.]),
        ], name="KRON"))
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="F277W_KRON", format="E", array=[15., 4.]),
            fits.Column(name="F277W_KRON_e", format="E", array=[2.5, 1.5]),
        ], name="KRON_CONV"))
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
                        "uncertainty_kind": "sigma", "unit": "nJy",
                        "psf_fwhm_arcsec": 0.25 + 0.01 * i})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=2, rows_per_file=1, image_size=8) == 2
    import pyarrow.parquet as pq
    first = pq.read_table(scratch / "GOODS-S_00000.parquet").to_pylist()[0]
    assert first["image"]["band"] == ["f770w"]
    assert first["image_sources"] == ["miri1.fits"]
    assert first["image"]["psf_fwhm"] == pytest.approx([0.26])
    assert first["phot_f770w_circ3_flux_njy"] == pytest.approx(101.0)
    assert first["phot_f770w_circ3_err_njy"] == pytest.approx(4.0)
    assert first["phot_f770w_kron_flux_njy"] == pytest.approx(131.0)
    assert first["phot_f1280w_circ3_flux_njy"] == pytest.approx(33.0)
    assert first["phot_f277w_circ3_flux_njy"] == pytest.approx(10.0)
    assert first["phot_f277w_kron_conv_flux_njy"] == pytest.approx(15.0)
    assert "phot_f770w_kron_conv_flux_njy" not in first
    assert first["image"]["ivar"][0][0][0] == pytest.approx(1.)
    assert first["photoz"]["z_peak"] == pytest.approx(5.2)
    assert first["photoz"]["z_a"] == pytest.approx(5.3)
    assert first["photoz_kron"]["z_a"] == pytest.approx(5.6)
    assert len(first["photoz"]["z_bins"]) == 22
    assert len(first["photoz"]["prob_z_bins"]) == 22
    assert first["z_spec"] == pytest.approx(5.1)
    second = pq.read_table(scratch / "GOODS-S_00001.parquet").to_pylist()[0]
    assert second["z_spec"] is None
    assert second["phot_f1280w_circ3_flux_njy"] is None
    assert second["phot_f1280w_circ3_err_njy"] is None

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
    assert report["sample_band_occurrences"]["f770w"] == 2
    assert report["sample_photometric_band_occurrences"]["f1280w"] == 1
    assert report["sample_photoz_occurrences"]["photoz_kron"] == 2


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


def test_jwst_one_row_combines_nircam_and_miri_at_native_scales(tmp_path):
    catalog = tmp_path / "cat.fits"
    _catalog(catalog)
    ids = fits.Column(name="ID", format="J", array=np.array([1, 2], dtype=np.int32))
    with fits.open(catalog, mode="update") as hdus:
        hdus.append(fits.BinTableHDU.from_columns([
            ids,
            fits.Column(name="F770W_CIRC3", format="E", array=[10., 20.]),
            fits.Column(name="F770W_CIRC3_e", format="E", array=[1., 2.]),
            fits.Column(name="F1280W_CIRC3", format="E", array=[3., 4.]),
        ], name="MIRI"))
    entries = []
    for band, scale, psf in (("F770W", 0.06, 0.27), ("F277W", 0.03, None)):
        path = tmp_path / f"{band.lower()}.fits"
        sci_header = _wcs(scale).to_header()
        if band == "F770W":
            sci_header["PIXAR_SR"] = 1e-14
            sci_header["BUNIT"] = "MJy/sr"
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((200, 200), dtype=np.float32), sci_header, name="SCI"),
            fits.ImageHDU(np.ones((200, 200), dtype=np.float32), name="ERR"),
        ]).writeto(path)
        entry = {"path": path.name, "field": "GOODS-S", "band": band,
                 "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                 "uncertainty_kind": "sigma",
                 "unit": "MJy/sr" if band == "F770W" else "nJy"}
        if psf is not None:
            entry["psf_fwhm_arcsec"] = psf
        entries.append(entry)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert process_field("GOODS-S", catalog, entries, tmp_path, scratch,
                         chunk_size=1, max_objects=2, image_size=8) == 2
    import pyarrow.parquet as pq
    table = pq.read_table(scratch / "GOODS-S_00000.parquet")
    row = table.to_pylist()[0]
    image = row["image"]
    assert image["band"] == ["f277w", "f770w"]
    assert image["scale"] == pytest.approx([0.03, 0.06])
    assert image["psf_fwhm"][0] is None
    assert image["psf_fwhm"][1] == pytest.approx(0.27)
    assert row["image_sources"] == ["f277w.fits", "f770w.fits"]
    assert len(image["flux"]) == len(image["ivar"]) == len(image["mask"]) == 2
    assert np.asarray(image["flux"]).shape == (2, 8, 8)
    assert image["flux"][0][0][0] == pytest.approx(1.)
    assert image["flux"][1][0][0] == pytest.approx(10.)  # MJy/sr -> nJy/pixel.
    assert image["ivar"][1][0][0] == pytest.approx(0.01)  # Sigma rescales too.
    assert "phot_f1280w_circ3_flux_njy" in table.column_names
    assert row["phot_f1280w_circ3_flux_njy"] == pytest.approx(3.0)
    assert row["photoz"]["z_a"] is None
    assert row["photoz_kron"]["z_a"] is None
    assert row["z_spec"] is None
    # Both rows share a single Parquet shard. The audit must sample rows,
    # rather than merely inspecting the first row of each file.
    report, samples = inspect(scratch, max_samples=2, expected_size=8)
    assert report["sampled_objects"] == 2
    assert report["sample_field_occurrences"] == {"GOODS-S": 2}
    assert report["sample_miri_objects"] == 2
    assert report["sample_joint_nircam_miri_objects"] == 2
    assert {row["object_id"] for row in samples} == {"GOODS-S:1", "GOODS-S:2"}
    with pytest.raises(ValueError, match="Incorrect cutout size"):
        inspect(scratch, max_samples=1, expected_size=160)


def test_manifest_rejects_invalid_psf_calibration(tmp_path):
    catalog = tmp_path / "cat.fits"
    _catalog(catalog)
    mosaic = tmp_path / "img.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((200, 200), np.float32), _wcs().to_header(), name="SCI"),
        fits.ImageHDU(np.ones((200, 200), np.float32), name="ERR"),
    ]).writeto(mosaic)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    entry = {"path": "img.fits", "field": "GOODS-S", "band": "F277W",
             "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
             "uncertainty_kind": "sigma", "unit": "nJy",
             "psf_fwhm_arcsec": -1.0}
    with pytest.raises(ValueError, match="Invalid psf_fwhm_arcsec"):
        process_field("GOODS-S", catalog, [entry], tmp_path, scratch, 1, 1, image_size=8)


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


def test_threaded_spatial_extraction_resumes_old_checkpoint_without_changing_rows(
    tmp_path, monkeypatch,
):
    """Switching worker strategy must preserve old Parquet and original row order."""
    catalog = tmp_path / "catalog.fits"
    wcs = _wcs()
    x = np.array([40, 150, 80, 120, 55, 135, 60, 130, 70, 115, 90, 100], dtype=float)
    y = np.array([140, 40, 95, 60, 130, 50, 120, 70, 110, 80, 100, 90], dtype=float)
    ra, dec = wcs.all_pix2world(x, y, 0)
    ids = fits.Column(name="ID", format="J", array=np.arange(1, 13, dtype=np.int32))
    table_cols = [ids,
                  fits.Column(name="RA", format="D", array=ra),
                  fits.Column(name="DEC", format="D", array=dec)]
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([
            *table_cols,
            fits.Column(name="A", format="E", array=np.full(12, 0.2)),
            fits.Column(name="B", format="E", array=np.full(12, 0.1)),
        ], name="SIZE"),
        fits.BinTableHDU.from_columns(table_cols, name="CIRC"),
    ]).writeto(catalog)
    arr = np.arange(200 * 200, dtype=np.float32).reshape((200, 200))
    mosaic = tmp_path / "image.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(arr, wcs.to_header(), name="SCI"),
        fits.ImageHDU(np.ones_like(arr), name="ERR"),
    ]).writeto(mosaic)
    entries = [{"field": "GOODS-S", "band": "F277W", "path": "image.fits",
                "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                "uncertainty_kind": "sigma", "unit": "nJy"}]

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    assert process_field("GOODS-S", catalog, entries, tmp_path, baseline,
                         chunk_size=1, max_objects=None, rows_per_file=2,
                         image_size=8) == 12

    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    original = builder._record

    def fail_after_committed_prefix(index, *args, **kwargs):
        if index == 5:
            raise RuntimeError("simulated old worker interruption")
        return original(index, *args, **kwargs)

    monkeypatch.setattr(builder, "_record", fail_after_committed_prefix)
    with pytest.raises(RuntimeError, match="simulated old worker interruption"):
        process_field("GOODS-S", catalog, entries, tmp_path, interrupted,
                      chunk_size=1, max_objects=None, rows_per_file=2,
                      image_size=8)
    checkpoint = json.loads((interrupted / "GOODS-S.progress.json").read_text())
    assert checkpoint["next_index"] == 4
    assert checkpoint["count"] == 4
    first_parquet_bytes = (interrupted / "GOODS-S_00000.parquet").read_bytes()

    monkeypatch.setattr(builder, "_record", original)
    assert process_field("GOODS-S", catalog, entries, tmp_path, interrupted,
                         chunk_size=1, max_objects=None, rows_per_file=2,
                         image_size=8, extract_threads=4, spatial_batch_size=7) == 12
    assert (interrupted / "GOODS-S_00000.parquet").read_bytes() == first_parquet_bytes

    import pyarrow.parquet as pq

    def saved_rows(root):
        return [row for part in sorted(root.glob("GOODS-S_*.parquet"))
                for row in pq.read_table(part).to_pylist()]

    assert saved_rows(interrupted) == saved_rows(baseline)
    assert [row["object_id"] for row in saved_rows(interrupted)] == [
        f"GOODS-S:{i}" for i in range(1, 13)
    ]


def test_field_split_selects_only_requested_catalog(tmp_path, monkeypatch):
    paths = []
    for field in ("GOODS-S", "GOODS-N"):
        path = tmp_path / f"{field}.fits"
        _catalog(path)
        paths.append({"field": field, "path": path.name})
    image = tmp_path / "image.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((200, 200), np.float32), _wcs().to_header(), name="SCI"),
        fits.ImageHDU(np.ones((200, 200), np.float32), name="ERR"),
    ]).writeto(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "release": "JADES_DR5", "catalogs": paths,
        "mosaics": [{"field": field, "band": "F277W", "path": image.name,
                     "sci_hdu": "SCI", "uncertainty_hdu": "ERR",
                     "uncertainty_kind": "sigma", "unit": "nJy"}
                    for field in ("GOODS-S", "GOODS-N")],
    }))
    scratch = tmp_path / "split"
    command = ["build_parent_sample_hats", "--instrument", "jwst",
               "--manifest", str(manifest), "--raw-root", str(tmp_path),
               "--scratch-dir", str(scratch), "--skip-hats",
               "--extract-threads", "2", "--spatial-batch-size", "4"]
    monkeypatch.setattr(sys, "argv", command + ["--field", "GOODS-S"])
    builder.main()
    assert (scratch / "GOODS-S.done").exists()
    assert not (scratch / "GOODS-N.done").exists()
    monkeypatch.setattr(sys, "argv", command + ["--field", "GOODS-N"])
    builder.main()
    assert verify_completed_shards(scratch, ["GOODS-S", "GOODS-N"], 1)["total_objects"] == 4


@pytest.mark.parametrize("task_index,field,shard", [
    (0, "GOODS-S", 0), (31, "GOODS-S", 31),
    (32, "GOODS-N", 0), (63, "GOODS-N", 31),
])
def test_slurm_fast_array_mapping_plan_only(task_index, field, shard):
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(repo),
               SLURM_ARRAY_TASK_ID=str(task_index), JADES_NUM_SHARDS="32",
               JADES_RAW="/not/required", JADES_RUN="/not/required",
               JADES_SPLIT_FIELDS="1", JADES_PLAN_ONLY="1",
               JADES_EXTRACT_THREADS="4", JADES_SPATIAL_BATCH_SIZE="256")
    result = subprocess.run(["bash", str(repo / "scripts/jades/slurm_extract.sh")],
                            env=env, cwd=repo, text=True, capture_output=True, check=True)
    assert f"shard={shard}/32 field={field}" in result.stdout
    assert f"--shard-index {shard}" in result.stdout
    assert f"--field {field}" in result.stdout
    assert "--extract-threads 4 --spatial-batch-size 256" in result.stdout
