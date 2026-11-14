"""Scientific count invariants and read-only JADES audit integration checks."""

import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from scripts.jades.audit_selection import (
    Catalog, discover_bands, extracted_ids, read_catalog, summarize,
)


def _catalog(path, *, misalign=False):
    ids = np.arange(1, 8, dtype=np.int64)
    circ = fits.BinTableHDU.from_columns([
        fits.Column(name="ID", format="K", array=ids),
        fits.Column(name="F277W_CIRC3", format="D", unit="nJy", array=[2, 0, -2, 3, 5, 10, 1]),
        fits.Column(name="F277W_CIRC3_e", format="D", unit="nJy", array=[1, 1, 1, 0, -1, np.nan, 1]),
    ], name="CIRC")
    kron = fits.BinTableHDU.from_columns([
        fits.Column(name="ID", format="K", array=ids[::-1] if misalign else ids),
        fits.Column(name="F277W_KRON", format="D", unit="uJy", array=[1, 0, -1, np.nan, 2, 3, 4]),
    ], name="KRON")
    miri = fits.BinTableHDU.from_columns([
        fits.Column(name="ID", format="K", array=ids),
        fits.Column(name="F770W_CIRC3", format="D", unit="uJy", array=[-2, 2, 3, 4, 5, 6, 7]),
        fits.Column(name="F770W_CIRC3_e", format="D", unit="nJy", array=[1000] * 7),
    ], name="MIRI")
    fits.HDUList([fits.PrimaryHDU(), circ, kron, miri]).writeto(path)


def _toy(field="GOODS-S"):
    return Catalog(field, None, np.arange(1, 8),
                   np.array([20, 21, 21.5, 22, 29, np.nan, 30]),
                   {"F277W": np.array([-1, 0, 2, 3, 5, 10, np.nan]),
                    "F770W": np.array([np.nan, 5, 5, 10, 20, np.nan, -2])}, [])


def _select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def test_fits_units_miri_and_signed_snr(tmp_path):
    path = tmp_path / "cat.fits"
    _catalog(path)
    entries = [("GOODS-S", path)]
    assert discover_bands(entries, "CIRC3") == ["F277W", "F770W"]
    catalog = read_catalog("GOODS-S", path, "F277W", ["F277W", "F770W", "F444W"], "CIRC3")
    assert catalog.mag[0] == pytest.approx(23.9)  # 1 uJy = 1000 nJy
    assert np.isnan(catalog.mag[1:4]).all()
    assert catalog.snr["F277W"][:3].tolist() == [2, 0, -2]
    assert np.isnan(catalog.snr["F277W"][3:6]).all()
    assert catalog.snr["F770W"][:3] == pytest.approx([-2, 2, 3])
    assert np.isnan(catalog.snr["F444W"]).all()
    assert not catalog.assumed_njy


def test_misaligned_hdus_fail_instead_of_mixing_sources(tmp_path):
    path = tmp_path / "cat.fits"
    _catalog(path, misalign=True)
    with pytest.raises(ValueError, match="not aligned"):
        read_catalog("GOODS-S", path, "F277W", ["F277W"], "CIRC3")


def test_magnitude_boundaries_and_histogram_partition():
    tables = summarize([_toy()], ["F277W", "F770W"], [21, 22, 29], [0, 2, 3, 5, 10])
    counts = _select(tables["counts"], field="ALL")
    cumulative = _select(counts, selection="cumulative")
    assert [r["n_selected"] for r in cumulative] == [1, 3, 4]
    intervals = _select(counts, selection="interval")
    assert [r["n_selected"] for r in intervals] == [1, 2, 1, 2]
    all_row = _select(counts, selection="all")[0]
    assert all_row["n_missing_mag"] == 1
    assert sum(r["n_selected"] for r in intervals) + all_row["n_missing_mag"] == all_row["n_selected"]
    bins = _select(tables["snr_bins"], field="ALL", selection="all", metric="F277W")
    assert {r["snr_bin"]: r["n_bin"] for r in bins} == {
        "<0": 1, "[0,2)": 1, "[2,3)": 1, "[3,5)": 1, "[5,10)": 1, ">=10": 1, "missing": 1,
    }
    assert sum(r["percent_selected"] for r in bins) == pytest.approx(100)
    assert sum(r["percent_valid_snr"] or 0 for r in bins) == pytest.approx(100)
    # The same partition invariant holds in every field and magnitude selection.
    grouped = {}
    for row in tables["snr_bins"]:
        key = tuple(row[k] for k in ("population", "field", "selection", "mag_lower", "mag_upper", "metric"))
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        assert sum(r["n_bin"] for r in rows) == rows[0]["n_selected"]


def test_combined_fields_multiband_counts_and_empty_denominators():
    first, second = _toy(), _toy("GOODS-N")
    tables = summarize([first, second], ["F277W", "F770W"], [19, 21, 22], [0, 2, 5],
                       retained={"GOODS-S": np.array([True, False, False, False, False, False, False]),
                                 "GOODS-N": np.zeros(7, dtype=bool)})
    rows = _select(tables["detections"], population="catalog", field="ALL", selection="all",
                   instrument="JWST", snr_threshold=5, min_bands=1)
    assert rows[0]["n_pass"] == 10  # Five sources per field, without counting bands as sources.
    rows = _select(tables["detections"], population="catalog", field="ALL", selection="all",
                   instrument="JWST", snr_threshold=5, min_bands=2)
    assert rows[0]["n_pass"] == 2
    rows = _select(tables["counts"], population="extracted", field="ALL", selection="all")
    assert rows[0]["n_selected"] == 1
    rows = _select(tables["snr_bins"], population="extracted", field="GOODS-N", selection="all")
    assert all(r["percent_selected"] is None for r in rows)
    # All output is strict JSON-safe, including empty selections.
    json.dumps(tables, allow_nan=False)


def test_snapshot_ignores_uncommitted_files_and_rejects_duplicates(tmp_path, monkeypatch):
    shard = tmp_path / "shard_0000_of_0032"
    shard.mkdir()
    checkpoint = shard / "GOODS-S.progress.json"
    checkpoint.write_text(json.dumps({"next_shard": 1, "count": 2}))
    table = pa.table({"field": ["GOODS-S", "GOODS-S"], "object_id": ["GOODS-S:1", "GOODS-S:2"]})
    pq.write_table(table, shard / "GOODS-S_00000.parquet")
    # Both are deliberately unreadable. The audit must not open them.
    (shard / "GOODS-S_00001.parquet").write_bytes(b"uncommitted")
    (shard / "GOODS-S_00002.parquet.part").write_bytes(b"in progress")
    original = pq.read_table

    def read_scalars_only(*args, **kwargs):
        assert kwargs["columns"] == ["field", "object_id"]
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_scalars_only)
    masks, metadata = extracted_ids(tmp_path, [_toy()])
    assert masks["GOODS-S"].tolist() == [True, True, False, False, False, False, False]
    assert metadata["committed_parquet_files"] == 1
    # A retry accidentally mixing duplicate IDs must fail instead of inflating N.
    pq.write_table(table, shard / "GOODS-S_00001.parquet")
    checkpoint.write_text(json.dumps({"next_shard": 2, "count": 4}))
    with pytest.raises(ValueError, match="duplicate extracted"):
        extracted_ids(tmp_path, [_toy()])


def test_cli_outputs_and_plots(tmp_path):
    raw = tmp_path / "raw" / "catalogs"
    raw.mkdir(parents=True)
    for field in ("GOODS-S", "GOODS-N"):
        _catalog(raw / f"{field}.fits")
    out = tmp_path / "audit"
    command = [sys.executable, "-m", "scripts.jades.audit_selection", "--raw-root", str(raw.parent),
               "--output-dir", str(out), "--mag-cuts", "21", "24", "29", "--plots"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["magnitude_band"] == "F277W"
    assert [r["n_selected"] for r in summary["counts"] if r["field"] == "ALL" and r["selection"] == "all"] == [14]
    for name in ("counts.csv", "snr_bins.csv", "detections.csv", "report.txt",
                 "selection_catalog.png", "selection_catalog.pdf"):
        assert (out / name).stat().st_size > 0
    failed = subprocess.run([*command, "--mag-cuts", "22", "21"], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "strictly increasing" in failed.stderr
