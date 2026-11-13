"""Exact population, cache invalidation, and bounded image-free Parquet I/O."""

import io
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from scripts.jades import hats_healpix_preflight as audit


def extraction(tmp_path, shards=2):
    scratch = tmp_path / "scratch"
    for index in range(shards):
        directory = scratch if shards == 1 else scratch / f"shard_{index:04d}_of_{shards:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        for field in audit.FIELDS:
            # Same numeric IDs in different fields; a gap represents rejection
            # by the cutout validator, which FITS catalog cuts cannot reproduce.
            ids = [1 + 4 * index, 3 + 4 * index]
            ra = 53.0 if field == "GOODS-S" else 189.0
            dec = -27.0 if field == "GOODS-S" else 62.0
            table = pa.table({"object_id": [f"{field}:{i}" for i in ids],
                              "ra": [ra + i * 0.0001 for i in ids],
                              "dec": [dec] * 2,
                              "field": [field] * 2,
                              "image": [b"cutout"] * 2})
            pq.write_table(table, directory / f"{field}_00000.parquet", row_group_size=1)
            state = dict(fingerprint=f"{field}-{index}", count=2, next_index=4, next_shard=1)
            if shards > 1:
                state.update(num_shards=shards, shard_index=index,
                             schema_version="jades-v2-multimodal-2")
            (directory / f"{field}.progress.json").write_text(json.dumps(state))
            (directory / f"{field}.done").write_text("2")
    return scratch, tmp_path / "coords.npz", 4 * shards


def test_exact_rows_order_and_healpix_match_legacy_scanner(tmp_path):
    scratch, cache, count = extraction(tmp_path)
    columns = audit.load_coordinates(scratch, cache, 2, count)
    reference = ds.dataset(sorted(str(p) for p in scratch.rglob("*.parquet")),
                           format="parquet").to_table(columns=audit.COLUMNS)
    for c in audit.COLUMNS:
        np.testing.assert_array_equal(columns[c], reference[c].to_numpy())
    for order in (0, 10, 13, 16):
        np.testing.assert_array_equal(
            audit.pixel_occupancies(columns["ra"], columns["dec"], order),
            audit.pixel_occupancies(reference["ra"].to_numpy(), reference["dec"].to_numpy(), order))
    assert "GOODS-S:2" not in columns["object_id"]


def test_warm_cache_opens_no_parquet(tmp_path, monkeypatch):
    scratch, cache, count = extraction(tmp_path)
    first = audit.load_coordinates(scratch, cache, 2, count)
    monkeypatch.setattr(audit.pq, "ParquetFile", lambda *a, **k: pytest.fail("Parquet reopened"))
    second = audit.load_coordinates(scratch, cache, 2, count)
    for c in audit.COLUMNS:
        np.testing.assert_array_equal(first[c], second[c])


def test_changed_file_with_same_row_count_invalidates_cache(tmp_path):
    scratch, cache, count = extraction(tmp_path)
    first = audit.load_coordinates(scratch, cache, 2, count)
    path = sorted(scratch.rglob("*.parquet"))[0]
    table = pq.ParquetFile(path).read()
    old_stat = path.stat()
    table = table.set_column(table.schema.get_field_index("ra"), "ra", pa.array([190.1, 190.2]))
    pq.write_table(table, path)
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 10_000_000))
    second = audit.load_coordinates(scratch, cache, 2, count)
    assert not np.array_equal(first["ra"], second["ra"])


@pytest.mark.parametrize("damage", ["done", "missing", "extra", "field", "shard", "count"])
def test_incomplete_or_inconsistent_extraction_fails_even_with_cache(tmp_path, damage):
    scratch, cache, count = extraction(tmp_path)
    audit.load_coordinates(scratch, cache, 2, count)
    checkpoint = sorted(scratch.rglob("*.progress.json"))[0]
    field = checkpoint.name.removesuffix(".progress.json")
    path = checkpoint.parent / f"{field}_00000.parquet"
    if damage == "done":
        checkpoint.with_name(f"{field}.done").unlink()
    elif damage == "missing":
        path.unlink()
    elif damage == "extra":
        (scratch / "uncommitted.parquet").write_bytes(b"bad")
    elif damage == "field":
        # Remove an entire field, leaving a plausible but incomplete shard tree.
        for path in scratch.rglob(f"{field}*"):
            path.unlink()
    elif damage == "shard":
        state = json.loads(checkpoint.read_text())
        state["shard_index"] = 1
        checkpoint.write_text(json.dumps(state))
    else:
        state = json.loads(checkpoint.read_text())
        state["count"] += 1
        checkpoint.write_text(json.dumps(state))
        checkpoint.with_name(f"{field}.done").write_text(str(state["count"]))
    with pytest.raises(ValueError):
        audit.load_coordinates(scratch, cache, 2, count)


@pytest.mark.parametrize("column,values,error", [
    ("object_id", ["GOODS-N:1", "GOODS-N:1"], "Duplicate"),
    ("object_id", ["GOODS-S:5", "GOODS-S:7"], "prefix"),
    ("ra", [np.nan, 189.0], "Invalid"),
    ("ra", [360.0, 189.0], "Invalid"),
    ("dec", [-91.0, 62.0], "Invalid"),
    ("ra", [None, 189.0], "null"),
    ("field", ["GOODS-S", "GOODS-S"], "field differs"),
])
def test_bad_scalar_rows_are_rejected(tmp_path, column, values, error):
    scratch, cache, count = extraction(tmp_path, shards=1)
    path = scratch / "GOODS-N_00000.parquet"
    table = pq.ParquetFile(path).read()
    table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=error):
        audit.load_coordinates(scratch, cache, 2, count)
    assert not cache.exists()


def test_per_checkpoint_count_not_just_grand_total(tmp_path):
    scratch, cache, count = extraction(tmp_path)
    paths = sorted(scratch.rglob("GOODS-N_00000.parquet"))
    tables = [pq.ParquetFile(p).read() for p in paths]
    pq.write_table(tables[0].slice(0, 1), paths[0])
    pq.write_table(pa.concat_tables([tables[0].slice(1), tables[1]]), paths[1])
    with pytest.raises(ValueError, match="Parquet rows differ"):
        audit.load_coordinates(scratch, cache, 2, count)


def test_mid_scan_mutation_does_not_publish_cache(tmp_path, monkeypatch):
    scratch, cache, count = extraction(tmp_path)
    original = audit.read_coordinates

    def change(path):
        columns = original(path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        return columns

    monkeypatch.setattr(audit, "read_coordinates", change)
    with pytest.raises(ValueError, match="changed during"):
        audit.load_coordinates(scratch, cache, 2, count)
    assert not cache.exists()


def test_only_scalar_ranges_read_from_image_heavy_row_groups(tmp_path, monkeypatch):
    path = tmp_path / "image.parquet"
    table = pa.table({"field": ["GOODS-S"] * 4,
                      "object_id": [f"GOODS-S:{i}" for i in range(4)],
                      "ra": [53.0] * 4, "dec": [-27.0] * 4,
                      "image": [os.urandom(2 * 1024 * 1024) for _ in range(4)]})
    pq.write_table(table, path, compression=None, row_group_size=1)

    class CountingFile(io.BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            data = super().read(size)
            self.bytes_read += len(data)
            return data

    reader = CountingFile(path.read_bytes())
    original = pq.ParquetFile

    def tracked(*args, **kwargs):
        assert kwargs == {"pre_buffer": False, "buffer_size": 0}
        return original(reader, **kwargs)

    monkeypatch.setattr(audit.pq, "ParquetFile", tracked)
    assert len(audit.read_coordinates(path)["ra"]) == 4
    # Allow the footer's 64 KiB tail read, but no multi-MiB image ranges.
    assert reader.bytes_read < path.stat().st_size * 0.02


def test_fits_checks_exact_subset_not_catalog_count(tmp_path):
    scratch, cache, count = extraction(tmp_path, shards=1)
    columns = audit.load_coordinates(scratch, cache, 2, count)
    raw = tmp_path / "raw"
    (raw / "catalogs").mkdir(parents=True)
    for field in audit.FIELDS:
        ra = 53.0 if field == "GOODS-S" else 189.0
        dec = -27.0 if field == "GOODS-S" else 62.0
        size = fits.BinTableHDU.from_columns([
            fits.Column(name="ID", format="K", array=[3, 2, 1]),
            fits.Column(name="RA", format="D", array=[ra + i * .0001 for i in [3, 2, 1]]),
            fits.Column(name="DEC", format="D", array=[dec] * 3),
        ], name="SIZE")
        fits.HDUList([fits.PrimaryHDU(), size]).writeto(raw / "catalogs" / f"{field}.fits")
    audit.validate_catalogs(columns, raw, None)
    # Same row count and valid IDs, different scientific selection: must fail.
    columns["object_id"][0] = "GOODS-N:2"
    with pytest.raises(ValueError, match="differs from catalog"):
        audit.validate_catalogs(columns, raw, None)
    columns["object_id"][0] = "GOODS-N:99"
    with pytest.raises(ValueError, match="absent from catalog"):
        audit.validate_catalogs(columns, raw, None)


def test_cli_pass_then_cached_threshold_failure(tmp_path):
    scratch, cache, count = extraction(tmp_path)
    command = [sys.executable, "-m", "scripts.jades.hats_healpix_preflight",
               "--scratch-dir", str(scratch), "--coordinate-cache", str(cache),
               "--expected-rows", str(count), "--orders", "0", "--highest-order", "0",
               "--pixel-threshold", "4"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASS:" in result.stdout and "selected ID SHA256:" in result.stdout
    command[-1] = "1"
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no Parquet opened" in result.stdout
    assert "exceeding" in result.stderr


def test_full_size_population_484219_of_485510(tmp_path):
    """Synthetic full-size selection: never substitute the 485,510 FITS rows."""
    scratch, raw = tmp_path / "scratch", tmp_path / "raw"
    scratch.mkdir()
    (raw / "catalogs").mkdir(parents=True)
    rejected = set(np.linspace(0, 485509, 1291, dtype=int).tolist())
    expected_ids = []
    offset = 0
    for field, nparent in zip(audit.FIELDS, (300000, 185510)):
        ids = np.arange(1, nparent + 1, dtype=np.int64)
        keep = np.array([offset + i not in rejected for i in range(nparent)])
        ra = (53.0 if field == "GOODS-S" else 189.0) + ids * 1e-7
        dec = np.full(nparent, -27.0 if field == "GOODS-S" else 62.0)
        selected_ids = [f"{field}:{i}" for i in ids[keep]]
        expected_ids.extend(selected_ids)
        pq.write_table(pa.table({"object_id": selected_ids, "ra": ra[keep],
                                 "dec": dec[keep], "field": [field] * int(keep.sum())}),
                       scratch / f"{field}_00000.parquet")
        state = dict(fingerprint=field, count=int(keep.sum()), next_index=nparent, next_shard=1)
        (scratch / f"{field}.progress.json").write_text(json.dumps(state))
        (scratch / f"{field}.done").write_text(str(state["count"]))
        size = fits.BinTableHDU.from_columns([
            fits.Column(name="ID", format="K", array=ids),
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="DEC", format="D", array=dec),
        ], name="SIZE")
        fits.HDUList([fits.PrimaryHDU(), size]).writeto(raw / "catalogs" / f"{field}.fits")
        offset += nparent
    columns = audit.load_coordinates(scratch, tmp_path / "coords.npz", 2, 484219)
    assert len(columns["ra"]) == 484219
    assert set(columns["object_id"]) == set(expected_ids)
    audit.validate_catalogs(columns, raw, None)
