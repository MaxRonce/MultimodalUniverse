import json
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.lsst_dp2.common import sha256_file
from scripts.lsst_dp2.publish_review import publish_review
from scripts.lsst_dp2.review_dataset import (
    check_completed_run,
    export_review,
    read_selected,
    sample_locations,
)
from scripts.lsst_dp2.schema import build_image_struct


@pytest.fixture
def completed_run(tmp_path):
    root = tmp_path / "run"
    inputs = root / "inputs"
    dataset = root / "hats/lsst_dp2/dataset"
    inputs.mkdir(parents=True)
    dataset.mkdir(parents=True)
    plan = inputs / "plan.json"
    plan.write_text(json.dumps({"objects": 5}))
    report = {
        "status": "PASS",
        "plan_sha256": sha256_file(plan),
        "catalog_rows": 5,
        "parquet_rows": 5,
        "hats_rows": 5,
        "unique_object_ids": 5,
        "center_checks": 30,
        "band_present_counts": dict.fromkeys("ugrizy", 5),
    }
    (root / "validation_report.json").write_text(json.dumps(report))
    (dataset.parent / "hats.properties").write_text(
        "obs_collection=lsst_dp2\nhats_nrows=5\n"
    )
    records = []
    for index in range(5):
        psf = np.zeros((6, 35, 35), dtype=np.float32)
        psf[:, 17, 17] = 1
        record = {
            "flux": np.full((6, 160, 160), index, dtype=np.float32),
            "ivar": np.ones((6, 160, 160), dtype=np.float32),
            "mask": np.ones((6, 160, 160), dtype=bool),
            "mask_bits": np.zeros((6, 160, 160), dtype=np.int32),
            "psf_image": psf,
            "psf_image_valid": np.ones(6, dtype=bool),
            "psf_fwhm": np.full(6, 0.7, dtype=np.float32),
            "scale": np.full(6, 0.2, dtype=np.float32),
            "band_present": np.ones(6, dtype=bool),
            "psf_source": ["catalog"] * 6,
            "dataset_id": ["test-dataset"] * 6,
            "sha256": ["a" * 64] * 6,
            "mask_plane_map": [json.dumps({"BAD": 0, "SAT": 1, "NO_DATA": 2})] * 6,
        }
        record["mask"][0, 0, 0] = False
        record["mask_bits"][0, 0, 0] = 1
        record["ivar"][0, 0, 0] = 0
        records.append(record)
    table = pa.table(
        {
            "object_id": [str(i) for i in range(5)],
            "image": build_image_struct(records),
            "i_cModelMag": [20.0] * 5,
            "sersic_reff_major": [0.8] * 5,
            "ra": [53.0] * 5,
            "dec": [-28.0] * 5,
            "tract": [5063] * 5,
            "patch": [25] * 5,
        }
    )
    pq.write_table(table, dataset / "part.parquet", row_group_size=2)
    return root, dataset / "part.parquet", records


def test_sample_locations_uniform_rows():
    counts = [0, 1, 100, 0, 3]
    expected = sorted(np.random.default_rng(12).choice(104, 15, replace=False))
    offsets = np.cumsum([0, *counts])
    locations = sample_locations(counts, 15, 12)
    assert locations == sample_locations(counts, 15, 12)
    assert [
        offsets[file] + row for file, rows in locations.items() for row in rows
    ] == expected
    assert sample_locations([0, 1, 0, 2], 10, 12) == {1: [0], 3: [0, 1]}
    with pytest.raises(ValueError):
        sample_locations([0], 2, 12)


def test_read_selected_row_groups(completed_run):
    _, path, records = completed_run
    selected = list(read_selected(path, [0, 3, 4]))
    assert [r["object_id"] for r in selected] == ["0", "3", "4"]
    for record in selected:
        np.testing.assert_array_equal(
            record["image"]["flux"], records[int(record["object_id"])]["flux"]
        )


def test_check_completed_run_rejects_stale_report_and_missing_data(completed_run):
    root, path, _ = completed_run
    report, paths, counts = check_completed_run(root)
    assert report["status"] == "PASS" and paths == [path] and counts == [5]
    (root / "inputs/plan.json").write_text('{"objects": 6}')
    with pytest.raises(ValueError, match="frozen plan"):
        check_completed_run(root)
    (root / "inputs/plan.json").write_text(json.dumps({"objects": 5}))
    pq.write_table(pq.read_table(path).slice(0, 4), path)
    with pytest.raises(ValueError, match="HATS metadata/data rows"):
        check_completed_run(root)


def test_export_preserves_pixels_and_validates_masks(completed_run, tmp_path):
    root, path, records = completed_run
    original_sha = sha256_file(path)
    output = tmp_path / "review"
    report = export_review(root, output, 2, 12, "linear")
    assert report["sample_rows"] == 2 and report["dataset_rows"] == 5
    assert report["rendering"]["raw_arrays_modified"] is False
    assert (output / "index.html").is_file()
    for item in report["items"]:
        assert (output / item["png"]).stat().st_size > 1000
        with np.load(output / item["npz"], allow_pickle=False) as raw:
            for key, value in records[int(item["object_id"])].items():
                np.testing.assert_array_equal(raw[key], value)
    assert sha256_file(path) == original_sha
    with pytest.raises(ValueError, match="already exists"):
        export_review(root, output, 2, 12, "linear")


def test_export_handles_missing_bands_and_rejects_bad_ivar(
    completed_run, tmp_path, monkeypatch
):
    root, path, records = completed_run
    monkeypatch.setattr("scripts.lsst_dp2.review_dataset.render", lambda *args: None)
    for record in records:
        for key in (
            "flux",
            "ivar",
            "mask",
            "mask_bits",
            "psf_image",
            "psf_image_valid",
            "psf_fwhm",
            "band_present",
        ):
            record[key][0] = 0
        record["psf_source"][0] = "missing"
        record["dataset_id"][0] = record["sha256"][0] = ""
        record["mask_plane_map"][0] = "{}"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("image"), "image", build_image_struct(records)
    )
    pq.write_table(table, path, row_group_size=2)
    report_path = root / "validation_report.json"
    report = json.loads(report_path.read_text())
    report["center_checks"] = 25
    report["band_present_counts"]["u"] = 0
    report_path.write_text(json.dumps(report))
    review = export_review(root, tmp_path / "missing", 5, 12, "linear")
    assert all(
        item["band_present"] == [False, True, True, True, True, True]
        for item in review["items"]
    )
    records[0]["ivar"][1, 0, 0] = -1
    table = table.set_column(
        table.schema.get_field_index("image"), "image", build_image_struct(records)
    )
    pq.write_table(table, path, row_group_size=2)
    with pytest.raises(ValueError, match="ivar/mask relation"):
        export_review(root, tmp_path / "invalid", 5, 12, "linear")


def test_publish_only_review_panels(tmp_path, monkeypatch):
    calls = {}

    class Run:
        dir = str(tmp_path)

        def __init__(self):
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, values):
            calls["log"] = values

    class Table:
        def __init__(self, columns):
            self.columns = columns
            self.data = []

        def add_data(self, *values):
            self.data.append(values)

    def init(**kwargs):
        calls["init"] = kwargs
        return Run()

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, Table=Table, Image=str)
    )
    panel = tmp_path / "object.png"
    panel.touch()
    review = {
        "status": "PASS",
        "seed": 12,
        "sampling": "uniform rows",
        "rendering": {},
        "scope": "sample",
        "dataset_rows": 5,
        "sample_rows": 1,
        "items": [
            {
                "png": "object.png",
                "npz": "never_uploaded.npz",
                "object_id": "123",
                "i_cModelMag": 20.0,
                "sersic_reff_major": 0.8,
                "clean_fraction": 0.9,
                "band_present": [False, True, True, True, True, True],
            }
        ],
    }
    (tmp_path / "review.json").write_text(json.dumps(review))
    publish_review(tmp_path, "test", "team", "offline")
    assert calls["init"]["mode"] == "offline"
    assert calls["log"]["cutouts"].data[0][-2:] == ("grizy", str(panel))
    review["items"][0]["png"] = "../outside.png"
    (tmp_path / "review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="invalid review panel"):
        publish_review(tmp_path, "test", "team", "offline")
