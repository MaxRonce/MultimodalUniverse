"""Publication tests use tiny scalar Parquet files, never the production images."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.jades import prepare_release as release


@pytest.fixture
def sample(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    root.mkdir()
    (root / "collection.properties").write_text("obs_collection=test\n")
    values = {"object_id": ["GOODS-S:2", "GOODS-N:1", "GOODS-S:1"],
              "field": ["GOODS-S", "GOODS-N", "GOODS-S"],
              "ra": [53.1, 189.1, 53.2], "dec": [-27.1, 62.1, -27.2]}
    for name in ("jades_jwst", "margin"):
        catalog = root / name
        (catalog / "dataset").mkdir(parents=True)
        (catalog / "hats.properties").write_text("metadata\n")
        pq.write_table(pa.table(values), catalog / "dataset/Npix=0.parquet")
    cache = tmp_path / "coordinates.npz"
    np.savez(cache, **{k: np.asarray(v) for k, v in values.items()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"inputs": []}')
    monkeypatch.setattr(release, "FIELD_COUNTS", {"GOODS-S": 2, "GOODS-N": 1})
    monkeypatch.setattr(release, "EXPECTED_PARTITIONS", 1)
    monkeypatch.setattr(release, "SELECTED_SHA256", hashlib.sha256(
        "\n".join(sorted(values["object_id"])).encode()).hexdigest())
    import hats
    from hats.catalog.dataset.collection_properties import CollectionProperties
    monkeypatch.setattr(hats, "read_hats", lambda path: None)
    monkeypatch.setattr(CollectionProperties, "read_from_dir", lambda path: SimpleNamespace(
        hats_primary_table_url="jades_jwst", all_margins=["margin"], default_margin="margin"))
    return root, cache, manifest, values


def test_stage_is_relocatable_and_preserves_exact_selection(sample, tmp_path):
    root, cache, manifest, _ = sample
    stage = tmp_path / "stage"
    report = release.prepare(root, stage, manifest, cache)
    assert report["validation"]["rows"] == 3
    assert report["validation"]["coordinates_equal_preflight"] is True
    assert report["validation"]["catalogs"]["margin"]["partitions"] == 1
    assert (stage / "README.md").read_text(encoding="utf-8").startswith("---")
    data = "jades_jwst/dataset/Npix=0.parquet"
    assert (stage / data).samefile(root / data)
    assert not (stage / "collection.properties").samefile(root / "collection.properties")
    paths = {entry["path"] for entry in json.loads((stage / "release/collection_files.json").read_text())}
    assert "margin/dataset/Npix=0.parquet" in paths
    with pytest.raises(ValueError, match="Stage exists"):
        release.prepare(root, stage, manifest, cache)


def test_equal_row_count_does_not_hide_wrong_ids(sample):
    root, cache, _, values = sample
    values["object_id"][0] = "GOODS-S:999"
    pq.write_table(pa.table(values), root / "jades_jwst/dataset/Npix=0.parquet")
    with pytest.raises(ValueError, match="ID hash differs"):
        release.validate(root, cache)


def test_coordinate_corruption_is_rejected(sample):
    root, cache, _, values = sample
    values["ra"][0] += .001
    pq.write_table(pa.table(values), root / "jades_jwst/dataset/Npix=0.parquet")
    with pytest.raises(ValueError, match="Final ra differs"):
        release.validate(root, cache)


def test_missing_margin_is_rejected(sample):
    root, cache, _, _ = sample
    (root / "margin/hats.properties").unlink()
    with pytest.raises(ValueError, match="Missing catalogue metadata"):
        release.validate(root, cache)
