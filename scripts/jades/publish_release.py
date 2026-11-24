"""Upload a prepared collection privately, or verify its remote file inventory.

Upload is restartable with the same stage and repository. No visibility change
is performed. Verify reports a pinned repository revision for training configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def local_files(stage: Path) -> dict[str, int]:
    return {p.relative_to(stage).as_posix(): p.stat().st_size
            for p in stage.rglob("*") if p.is_file()
            and ".cache" not in p.relative_to(stage).parts}


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "verify"])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo", required=True, help="namespace/jades-jwst")
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    provenance = stage / "release/provenance.json"
    expected = json.loads(provenance.read_text(encoding="utf-8"))
    if not expected["validation"]["coordinates_equal_preflight"]:
        raise ValueError("Release needs exact-coordinate validation")
    api = HfApi()
    if args.action == "upload":
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="dataset")
    if args.action == "upload":
        if not info.private:
            raise ValueError("This preparatory uploader requires a PRIVATE repository")
        existing = set(api.list_repo_files(args.repo, repo_type="dataset"))
        if "release/provenance.json" in existing:
            previous = Path(hf_hub_download(args.repo, "release/provenance.json", repo_type="dataset"))
            if json.loads(previous.read_text(encoding="utf-8")) != expected:
                raise ValueError("Repository contains a different release; use a fresh repository")
        elif existing - {".gitattributes"}:
            raise ValueError("Repository is nonempty without matching provenance; inspect it first")
        else:
            # Publish the release identity first, so even an interruption during
            # the first large-folder call can be resumed unambiguously.
            api.upload_file(repo_id=args.repo, repo_type="dataset",
                            path_or_fileobj=str(provenance), path_in_repo="release/provenance.json",
                            commit_message="Record validated JADES release identity")
        api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(stage),
                                num_workers=4, ignore_patterns=[".cache/**"])
        info = api.repo_info(args.repo, repo_type="dataset")
    revision = info.sha
    remote = {item.path: item.size for item in api.list_repo_tree(
        args.repo, repo_type="dataset", revision=revision, recursive=True)
              if hasattr(item, "size")}
    local = local_files(stage)
    mismatches = {p: {"local": size, "remote": remote.get(p)}
                  for p, size in local.items() if remote.get(p) != size}
    extra = set(remote) - set(local) - {".gitattributes"}
    if mismatches or extra:
        raise ValueError(f"Remote inventory differs: {mismatches}; extra files: {sorted(extra)}")
    saved = Path(hf_hub_download(args.repo, "release/provenance.json", repo_type="dataset", revision=revision))
    if saved.read_bytes() != provenance.read_bytes():
        raise ValueError("Remote provenance differs")
    print(json.dumps({"status": "PASS: remote paths, sizes and provenance match",
                      "repo": args.repo, "revision": revision, "files": len(local),
                      "bytes": sum(local.values()), "private": info.private}, indent=2))
    print("Inventory verification is not a full read-back checksum of image payloads.")


if __name__ == "__main__":
    main()
