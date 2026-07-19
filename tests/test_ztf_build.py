"""Tests for scripts/ztf/build_parent_sample_hats.py."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "ztf", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_ztf_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _write_nested_ztf_hats(root, npix=1):
    dataset = root / "ztf_dr23_lc-hats" / "dataset"
    part = dataset / "Norder=3" / "Dir=0" / f"Npix={npix}"
    part.mkdir(parents=True)
    lc_type = pa.struct(
        [
            pa.field("hmjd", pa.list_(pa.float64())),
            pa.field("mag", pa.list_(pa.float32())),
            pa.field("magerr", pa.list_(pa.float32())),
            pa.field("clrcoeff", pa.list_(pa.float32())),
            pa.field("catflags", pa.list_(pa.int32())),
        ]
    )

    rows = [
        {
            "objectid": 1001,
            "filterid": 1,
            "fieldid": 12,
            "rcid": 3,
            "objra": 150.1,
            "objdec": 2.1,
            "hmjd": [60000.0, 60001.0, np.nan],
            "mag": [16.0, 14.8, 17.0],
            "magerr": [0.10, 0.20, 0.30],
            "clrcoeff": [0.1, 0.1, 0.1],
            "catflags": [0, 0, 0],
        },
        {
            # Same objectid as the first row but a different ZTF/IPAC series.
            # The MMU conversion must preserve both rows and not merge them.
            "objectid": 1001,
            "filterid": 2,
            "fieldid": 12,
            "rcid": 4,
            "objra": 150.1002,
            "objdec": 2.1001,
            "hmjd": [60100.0 + i for i in range(30)],
            "mag": [18.0] * 30,
            "magerr": [0.05] * 30,
            "clrcoeff": [0.2] * 30,
            "catflags": [0] * 30,
        },
        {
            "objectid": 2002,
            "filterid": 3,
            "fieldid": 99,
            "rcid": 1,
            "objra": 10.0,
            "objdec": -5.0,
            "hmjd": [60200.0, 60201.0, 60202.0, 60203.0],
            "mag": [14.0, 15.0, 16.0, 17.0],
            "magerr": [0.1, -0.1, 0.1, 0.1],
            "clrcoeff": [0.3, 0.3, 0.3, 0.3],
            "catflags": [0, 0, 32768, 2],
        },
    ]
    lightcurves = [
        {
            "hmjd": row["hmjd"],
            "mag": row["mag"],
            "magerr": row["magerr"],
            "clrcoeff": row["clrcoeff"],
            "catflags": row["catflags"],
        }
        for row in rows
    ]
    table = pa.table(
        {
            "_healpix_29": pa.array([1, 2, 3], type=pa.int64()),
            "objectid": pa.array([row["objectid"] for row in rows], type=pa.int64()),
            "filterid": pa.array([row["filterid"] for row in rows], type=pa.int8()),
            "fieldid": pa.array([row["fieldid"] for row in rows], type=pa.int16()),
            "rcid": pa.array([row["rcid"] for row in rows], type=pa.int8()),
            "objra": pa.array([row["objra"] for row in rows], type=pa.float32()),
            "objdec": pa.array([row["objdec"] for row in rows], type=pa.float32()),
            "nepochs": pa.array([len(row["hmjd"]) for row in rows], type=pa.int64()),
            "lightcurve": pa.array(lightcurves, type=lc_type),
        }
    )
    pq.write_table(table, part / "part0.snappy.parquet")
    return root / "ztf_dr23_lc-hats"


def _read_single_converted_parquet(scratch):
    parts = sorted(scratch.glob("*.parquet"))
    assert len(parts) == 1
    return pq.read_table(parts[0])


def test_resolve_scan_dir_primary(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    scan_dir, warning = build.resolve_scan_dir(root, "auto")
    assert scan_dir.name == "dataset"
    assert warning is None


def test_convert_batch_preserves_native_series_and_quality_flags(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    files = build.discover_parquet_files(root / "dataset")
    batch = next(pq.ParquetFile(files[0]).iter_batches(batch_size=10))
    table = build.convert_batch(batch, build.ConvertConfig())

    assert table.num_rows == 3
    assert table.column("object_id").to_pylist() == ["1001", "1001", "2002"]
    assert table.column("band").to_pylist() == ["ztf_g", "ztf_r", "ztf_i"]
    assert table.column("n_total").to_pylist() == [3, 30, 4]
    assert table.column("n_relaxed").to_pylist() == [2, 30, 2]
    assert table.column("n_strict").to_pylist() == [2, 30, 1]
    assert table.column("n_bright_relaxed").to_pylist() == [1, 0, 1]
    assert table.column("pass_relaxed_ge30").to_pylist() == [False, True, False]
    assert table.column("aion_training_candidate").to_pylist() == [False, True, False]
    assert table.column("aion_training_candidate_v1").to_pylist() == [False, True, False]
    assert set(table.column("quality_policy_version").to_pylist()) == {
        "ztf_aion_relaxed_ge30_v1"
    }

    lc0 = table.column("lightcurve")[0].as_py()
    assert lc0["time"][:2] == [60000.0, 60001.0]
    assert np.isnan(lc0["time"][2])
    assert lc0["band"] == ["ztf_g", "ztf_g", "ztf_g"]
    assert lc0["finite_mask"] == [True, True, False]
    assert lc0["poserr_mask"] == [True, True, False]
    assert lc0["relaxed_mask"] == [True, True, False]
    assert lc0["strict_mask"] == [True, True, False]
    assert lc0["bright_mask"] == [False, True, False]

    lc2 = table.column("lightcurve")[2].as_py()
    assert lc2["poserr_mask"] == [True, False, True, True]
    assert lc2["relaxed_mask"] == [True, False, False, True]
    assert lc2["strict_mask"] == [True, False, False, False]


def test_convert_files_to_parquet_dir_streaming(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    scratch = tmp_path / "scratch"
    files = build.discover_parquet_files(root / "dataset")
    summary = build.convert_files_to_parquet_dir(
        files,
        scratch,
        build.ConvertConfig(),
        batch_size=2,
        num_processes=1,
    )

    assert summary["files_written"] == 1
    assert summary["rows"] == 3
    assert summary["observations"] == 37
    table = _read_single_converted_parquet(scratch)
    assert table.num_rows == 3
    assert int(table.column("n_total").to_numpy().sum()) == 37


def test_sharded_outputs_use_global_file_indices(tmp_path):
    root_a = _write_nested_ztf_hats(tmp_path / "ztf_a", npix=1)
    root_b = _write_nested_ztf_hats(tmp_path / "ztf_b", npix=2)
    files = [
        build.discover_parquet_files(root_a / "dataset")[0],
        build.discover_parquet_files(root_b / "dataset")[0],
    ]
    scratch = tmp_path / "scratch"

    build.convert_files_to_parquet_dir(
        [(0, files[0])],
        scratch,
        build.ConvertConfig(),
        batch_size=10,
        num_processes=1,
        shard_idx=0,
    )
    build.convert_files_to_parquet_dir(
        [(1, files[1])],
        scratch,
        build.ConvertConfig(),
        batch_size=10,
        num_processes=1,
        shard_idx=1,
    )

    assert sorted(path.name for path in scratch.glob("*.parquet")) == [
        "part-00000000.parquet",
        "part-00000001.parquet",
    ]


def test_main_writes_small_hats_catalog(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    rc = build.main(
        [
            "--input-dir", str(root),
            "--output-root", str(out),
            "--scratch-dir", str(scratch),
            "--num-processes", "1",
            "--batch-size", "2",
            "--pixel-threshold", "10",
            "--ingest-workers", "1",
            "--ingest-chunksize", "2",
        ]
    )

    assert rc == 0
    catalog_dir = out / "ztf" / "ztf"
    assert (catalog_dir / "hats.properties").is_file()
    parts = sorted((catalog_dir / "dataset").rglob("*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    assert "aion_training_candidate" in table.schema.names
    assert "lightcurve" in table.schema.names


def test_main_direct_hats_preserves_partitions_and_metadata(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    out = tmp_path / "direct_out"
    manifest_dir = tmp_path / "direct_manifests"
    rc = build.main(
        [
            "--input-dir", str(root),
            "--output-root", str(out),
            "--scratch-dir", str(manifest_dir),
            "--direct-hats",
            "--num-processes", "1",
            "--batch-size", "2",
        ]
    )

    assert rc == 0
    catalog_dir = out / "ztf" / "ztf"
    part = (
        catalog_dir
        / "dataset"
        / "Norder=3"
        / "Dir=0"
        / "Npix=1.parquet"
    )
    assert part.is_file()
    assert (out / "ztf" / "collection.properties").is_file()
    assert (catalog_dir / "hats.properties").is_file()
    assert (catalog_dir / "properties").is_file()
    assert (catalog_dir / "partition_info.csv").read_text().strip() == "Norder,Npix\n3,1"
    assert (catalog_dir / "dataset" / "_common_metadata").is_file()
    assert (catalog_dir / "dataset" / "_metadata").is_file()
    assert (manifest_dir / "manifest-direct-shard-000.json").is_file()
    policy = build.json.loads((catalog_dir / "aion_quality_policy.json").read_text())
    assert policy["version"] == "ztf_aion_relaxed_ge30_v1"
    assert policy["candidate_mask"] == "relaxed_mask"

    table = pq.read_table(part)
    assert table.column("_healpix_29").to_pylist() == [1, 2, 3]
    assert table.column("n_total").to_pylist() == [3, 30, 4]
    assert table.column("aion_training_candidate").to_pylist() == [False, True, False]


def test_direct_hats_shards_write_distinct_partition_paths(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf", npix=1)
    _write_nested_ztf_hats(tmp_path / "ztf", npix=2)
    files = build.discover_parquet_files(root / "dataset")
    out_dataset = tmp_path / "direct_dataset"

    build.convert_files_to_direct_hats(
        [(0, files[0])],
        root / "dataset",
        out_dataset,
        build.ConvertConfig(),
        batch_size=10,
        num_processes=1,
        shard_idx=0,
    )
    build.convert_files_to_direct_hats(
        [(1, files[1])],
        root / "dataset",
        out_dataset,
        build.ConvertConfig(),
        batch_size=10,
        num_processes=1,
        shard_idx=1,
    )

    assert sorted(path.relative_to(out_dataset).as_posix() for path in out_dataset.rglob("*.parquet")) == [
        "Norder=3/Dir=0/Npix=1.parquet",
        "Norder=3/Dir=0/Npix=2.parquet",
    ]


def test_quality_policy_version_tracks_nondefault_candidate_threshold(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    files = build.discover_parquet_files(root / "dataset")
    batch = next(pq.ParquetFile(files[0]).iter_batches(batch_size=10))
    table = build.convert_batch(batch, build.ConvertConfig(aion_relaxed_threshold=20))
    assert set(table.column("quality_policy_version").to_pylist()) == {
        "ztf_aion_relaxed_ge20_v1"
    }


def test_direct_manifest_config_rejects_mixed_quality_policies(tmp_path):
    (tmp_path / "manifest-direct-shard-000.json").write_text(
        build.json.dumps({"config": {"aion_relaxed_threshold": 20}})
    )
    (tmp_path / "manifest-direct-shard-001.json").write_text(
        build.json.dumps({"config": {"aion_relaxed_threshold": 30}})
    )
    try:
        build.direct_manifest_config(tmp_path)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("Mixed quality policies must fail before HATS finalization")


def test_resume_rejects_output_from_a_different_quality_policy(tmp_path):
    root = _write_nested_ztf_hats(tmp_path / "ztf")
    files = build.discover_parquet_files(root / "dataset")
    out_dataset = tmp_path / "direct_dataset"
    build.convert_files_to_direct_hats(
        [(0, files[0])],
        root / "dataset",
        out_dataset,
        build.ConvertConfig(aion_relaxed_threshold=20),
        batch_size=10,
        num_processes=1,
        shard_idx=0,
    )
    summary = build.convert_files_to_direct_hats(
        [(0, files[0])],
        root / "dataset",
        out_dataset,
        build.ConvertConfig(aion_relaxed_threshold=30),
        batch_size=10,
        num_processes=1,
        shard_idx=0,
        resume=True,
    )
    assert summary["files_written"] == 1
    assert summary["files_skipped"] == 0
