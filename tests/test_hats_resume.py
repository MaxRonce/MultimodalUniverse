"""Checkpoint and retry behavior for large, distributed HATS imports."""

from pathlib import Path

import pytest

from mmu import hats_import as importer


class _FakeCollectionArguments:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.catalog_options = {}

    def catalog(self, **kwargs):
        self.catalog_options = kwargs
        return self

    def add_margin(self, **kwargs):
        return self


def _mock_hats(monkeypatch, pipeline):
    monkeypatch.setattr(importer, "CollectionArguments", _FakeCollectionArguments)
    monkeypatch.setattr(importer, "ParquetPyarrowReader", lambda **kwargs: object())
    monkeypatch.setattr(importer, "pipeline_with_client", pipeline)


def test_mapping_retry_reuses_checkpoints_and_preserves_output(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    for index in range(2):
        (inputs / f"{index}.parquet").touch()

    output = tmp_path / "output"
    partial = output / "jades_jwst" / "jades_jwst" / "partial.parquet"
    partial.parent.mkdir(parents=True)
    partial.write_text("incomplete output must not be deleted")
    tmp_dir = tmp_path / "hats_tmp"
    attempts = []

    def fake_pipeline(args, client):
        attempts.append(args)
        if len(attempts) == 1:
            histogram = Path(args.options["tmp_dir"]) / "resume" / "row_count" / "map_0.npz"
            histogram.parent.mkdir(parents=True)
            histogram.touch()
            raise RuntimeError("some map stages did not complete successfully.")

    _mock_hats(monkeypatch, fake_pipeline)
    result = importer.write_hats_from_parquet_dir(
        str(inputs), str(output), "jades_jwst", tmp_dir=str(tmp_dir),
        client=object(), resumable=True, mapping_retries=1,
        pixel_threshold=256, highest_healpix_order=15, expected_total_rows=2,
    )

    assert result == str(partial.parent)
    assert len(attempts) == 2
    assert attempts[0].options["tmp_dir"] == attempts[1].options["tmp_dir"]
    assert attempts[0].catalog_options["highest_healpix_order"] == 15
    assert attempts[0].catalog_options["expected_total_rows"] == 2
    assert partial.read_text() == "incomplete output must not be deleted"
    assert not (tmp_dir / "hats_import_jades_jwst_resume").exists()


def test_failed_import_keeps_staging_and_refuses_changed_config(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "0.parquet").touch()
    output = tmp_path / "output"
    tmp_dir = tmp_path / "hats_tmp"
    calls = []

    def fake_pipeline(args, client):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("some map stages did not complete successfully.")

    _mock_hats(monkeypatch, fake_pipeline)
    options = dict(
        parquet_dir=str(inputs), output_path=str(output), catalog_name="jades_jwst",
        tmp_dir=str(tmp_dir), client=object(), resumable=True, mapping_retries=0,
        pixel_threshold=256,
    )
    with pytest.raises(RuntimeError, match="some map stages"):
        importer.write_hats_from_parquet_dir(**options)

    staging = tmp_dir / "hats_import_jades_jwst_resume"
    assert (staging / "_mmu_resume_identity.json").is_file()
    with pytest.raises(ValueError, match="identity mismatch"):
        importer.write_hats_from_parquet_dir(**(options | {"pixel_threshold": 512}))

    importer.write_hats_from_parquet_dir(**options)
    assert len(calls) == 2
    assert not staging.exists()
