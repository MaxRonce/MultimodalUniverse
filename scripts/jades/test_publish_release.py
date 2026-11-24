"""Exercise upload guards and first-transfer restart identity without a network."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.jades import publish_release as release


@pytest.fixture
def context(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / "release").mkdir(parents=True)
    provenance = stage / "release/provenance.json"
    provenance.write_text(json.dumps({"validation": {"coordinates_equal_preflight": True}}))
    (stage / "README.md").write_text("card")
    monkeypatch.setattr(sys, "argv", ["publish_release", "upload", "--repo", "user/jades",
                                      "--stage", str(stage)])
    import huggingface_hub
    state = SimpleNamespace(private=True, existing=[], events=[], sizes=release.local_files(stage))

    class Api:
        def create_repo(self, *args, **kwargs):
            state.events.append("create-private")

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=state.private, sha="pinned-revision")

        def list_repo_files(self, *args, **kwargs):
            return state.existing

        def upload_file(self, **kwargs):
            assert kwargs["path_in_repo"] == "release/provenance.json"
            state.events.append("identity")

        def upload_large_folder(self, **kwargs):
            state.events.append("payload")

        def list_repo_tree(self, *args, **kwargs):
            return [SimpleNamespace(path=p, size=n) for p, n in state.sizes.items()]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(provenance))
    return state


def test_identity_precedes_payload_for_restartability(context):
    release.main()
    assert context.events == ["create-private", "identity", "payload"]


def test_refuse_public_repository(context):
    context.private = False
    with pytest.raises(ValueError, match="PRIVATE"):
        release.main()
    assert "payload" not in context.events


def test_refuse_unidentified_existing_data(context):
    context.existing = ["old.parquet"]
    with pytest.raises(ValueError, match="nonempty"):
        release.main()


def test_remote_size_mismatch_is_not_success(context):
    context.sizes["README.md"] += 1
    with pytest.raises(ValueError, match="Remote inventory differs"):
        release.main()
