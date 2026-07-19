from __future__ import annotations

import importlib.util
import sys

import pandas as pd


def _load_module():
    path = __file__.replace("tests/test_plasticc_hats_build.py", "scripts/plasticc/build_parent_sample_hats.py")
    spec = importlib.util.spec_from_file_location("_plasticc_hats_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_module()


def _group(object_id=1, n=6):
    return pd.DataFrame(
        {
            "object_id": [object_id] * n,
            "mjd": [60000.0 + index for index in range(n)],
            "passband": [index % 6 for index in range(n)],
            "flux": [10.0 + index for index in range(n)],
            "flux_err": [0.5] * n,
            "detected": [1, 1, 0, 1, 0, 1][:n],
        }
    )


def test_convert_object_is_jagged_and_preserves_all_six_bands():
    policy = build.QualityPolicy(min_quality_epochs=6, min_quality_bands=6)
    metadata = pd.Series({"ra": 10.0, "decl": -5.0, "true_target": 90, "true_z": 0.2})
    row = build.convert_object(_group(), metadata, "plasticc_train", policy)
    assert row["lightcurve"]["band"] == ["u", "g", "r", "i", "z", "y"]
    assert row["lightcurve"]["quality_mask"] == [True] * 6
    assert row["lightcurve"]["detected_mask"] == [True, True, False, True, False, True]
    assert row["n_total"] == 6
    assert row["n_quality_bands"] == 6
    assert row["aion_training_candidate_v1"] is True
    table = build.rows_to_table([row])
    lightcurve = table.column("lightcurve")[0].as_py()
    assert len(lightcurve["time"]) == 6
    assert not any(
        time == flux == error == 0
        for time, flux, error in zip(lightcurve["time"], lightcurve["flux"], lightcurve["flux_err"])
    )


def test_quality_policy_keeps_bad_epochs_but_masks_them():
    group = _group(n=3)
    group.loc[1, "flux_err"] = 0.0
    group.loc[2, "flux"] = float("nan")
    policy = build.QualityPolicy(min_quality_epochs=1, min_quality_bands=1)
    row = build.convert_object(group, pd.Series({"ra": 1.0, "decl": 2.0}), "plasticc_train", policy)
    assert row["n_total"] == 3
    assert row["n_quality"] == 1
    assert row["lightcurve"]["quality_mask"] == [True, False, False]
    assert len(row["lightcurve"]["time"]) == 3


def test_rows_to_table_keeps_schema_when_optional_metadata_is_all_null():
    policy = build.QualityPolicy(min_quality_epochs=1, min_quality_bands=1)
    row = build.convert_object(
        _group(n=3),
        pd.Series({"ra": 1.0, "decl": 2.0}),
        "plasticc_train",
        policy,
    )
    table = build.rows_to_table([row])
    assert table.schema.field("redshift").type == build.pa.float64()
    assert table.schema.field("obj_type").type == build.pa.string()


def test_chunk_iterator_does_not_split_objects(tmp_path):
    frame = pd.concat([_group(1, 4), _group(2, 4)], ignore_index=True)
    path = tmp_path / "lightcurves.csv"
    frame.to_csv(path, index=False)
    groups = list(build.iter_complete_object_groups(path, chunksize=3))
    assert [len(group) for group in groups] == [4, 4]
    assert [int(group["object_id"].iloc[0]) for group in groups] == [1, 2]
