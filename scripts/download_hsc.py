#!/usr/bin/env python3
"""Download complete, authenticated HATS repositories to Ceph.

Requires huggingface_hub>=1.33,<2 and hf-xet>=1.8,<2.
See docs/DOWNLOAD_HSC.md and docs/DOWNLOAD_LEGACY.md for setup and usage.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time

from huggingface_hub import HfApi, RepoFile, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


DATASETS = {
    "wide": "hugging-science/mmu_hsc_pdr3_wide_21",
    "dud": "UniverseTBD/mmu_hsc_pdr3_dud_22.5",
}
MANIFEST = ".hf-download-manifest.json"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("La valeur doit etre >= 1.")
    return number


def volume(size):
    return f"{size / 1e9:,.2f} Go ({size / 2**30:,.2f} Gio)"


def validated_repo_id(value):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or any(
        part in (".", "..") for part in value.split("/")
    ):
        raise argparse.ArgumentTypeError("Utiliser un depot au format organisation/nom.")
    return value


def get_plan(api, repo_id, directory, verify_only=False):
    manifest = directory / MANIFEST
    if manifest.exists():
        print(f"Lecture du manifeste local : {manifest}", flush=True)
        plan = json.loads(manifest.read_text())
        if plan["repo_id"] != repo_id:
            raise RuntimeError(f"Depot incompatible dans {manifest}")
        return plan
    if verify_only:
        raise RuntimeError(f"Manifeste absent : {manifest}")
    # The siblings array in dataset_info is capped at 100,000 files on large repos.
    # Resolve once, then page through the tree of this exact immutable revision.
    info = api.dataset_info(repo_id, expand=["sha"])
    if not info.sha:
        raise RuntimeError(f"Commit inconnu pour {repo_id}")
    print(f"Inventaire complet de {repo_id} (pagination)...", flush=True)
    files = []
    for entry in api.list_repo_tree(
        repo_id, repo_type="dataset", revision=info.sha, recursive=True
    ):
        if not isinstance(entry, RepoFile):
            continue
        if entry.size is None:
            raise RuntimeError(f"Taille inconnue : {repo_id}/{entry.path}")
        files.append({"path": entry.path, "size": entry.size})
        if len(files) % 20000 == 0:
            print(f"  {len(files):,} fichiers inventories...", flush=True)
    if not files:
        raise RuntimeError(f"Inventaire vide pour {repo_id}")
    return {"repo_id": repo_id, "revision": info.sha, "files": files}


def save_plan(directory, plan):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(directory / MANIFEST)


def incomplete_files(directory, plan):
    total = len(plan["files"])
    print(f"Verification locale des presences et tailles : {total:,} fichiers "
          "(sans lecture du contenu)...", flush=True)
    if not directory.exists():
        print("  Dossier absent : tous les fichiers sont a telecharger.", flush=True)
        return plan["files"]
    missing = []
    started = last_report = time.monotonic()
    for checked, entry in enumerate(plan["files"], 1):
        try:
            local = (directory / entry["path"]).stat()
        except (FileNotFoundError, NotADirectoryError):
            missing.append(entry)
        else:
            if not stat.S_ISREG(local.st_mode) or local.st_size != entry["size"]:
                missing.append(entry)
        now = time.monotonic()
        if now - last_report >= 5 or checked == total:
            print(f"  {checked:,}/{total:,} controles, {len(missing):,} a telecharger "
                  f"({now - started:.0f} s)", flush=True)
            last_report = now
    return missing


def download(plan, directory, workers, attempts):
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=plan["repo_id"],
                repo_type="dataset",
                revision=plan["revision"],
                local_dir=directory,
                token=True,
                max_workers=workers,
            )
            return
        except Exception as error:
            if isinstance(error, HfHubHTTPError) and error.response is not None:
                if error.response.status_code in (401, 403, 404):
                    raise
            if attempt == attempts:
                raise
            delay = min(15 * 2 ** (attempt - 1), 60)
            print(f"Tentative {attempt}/{attempts} : {error}", file=sys.stderr)
            print(f"Reprise dans {delay} s...", flush=True)
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    selection.add_argument("--repo-id", type=validated_repo_id,
                           help="Autre depot HATS ; necessite --root explicite.")
    parser.add_argument(
        "--root", type=Path,
        help="Dossier parent : organisation/nom sera ajoute automatiquement.",
    )
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--attempts", type=positive_int, default=5)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true",
                       help="Inventaire distant uniquement, aucun fichier telecharge.")
    modes.add_argument("--verify-only", action="store_true",
                       help="Controle local des presences/tailles, sans reseau.")
    args = parser.parse_args()
    if args.repo_id and args.root is None:
        parser.error("--repo-id necessite --root pour choisir le dossier de destination.")
    if args.root is None:
        args.root = Path(os.environ.get(
            "HSC_DEST", "/mnt/ceph/users/polymathic/mronceray/hsc_pdr3"
        ))

    api = HfApi(token=True)
    if not args.verify_only:
        account = api.whoami()
        print(f"Compte Hugging Face : {account['name']}", flush=True)

    if args.repo_id:
        selected = {"custom": args.repo_id}
    elif args.dataset == "all":
        selected = DATASETS
    else:
        selected = {args.dataset: DATASETS[args.dataset]}
    plans = []
    remaining = 0
    has_incomplete = False
    for repo_id in selected.values():
        directory = args.root / repo_id
        plan = get_plan(api, repo_id, directory, args.verify_only)
        missing = incomplete_files(directory, plan)
        has_incomplete = has_incomplete or bool(missing)
        pending_size = sum(entry["size"] for entry in missing)
        remaining += pending_size
        print(f"\n{repo_id}\n  Commit : {plan['revision']}\n  Dossier : {directory}")
        print(f"  Total : {len(plan['files'])} fichiers, "
              f"{volume(sum(entry['size'] for entry in plan['files']))}")
        print(f"  Absents ou de taille incorrecte : {len(missing)}, "
              f"{volume(pending_size)}", flush=True)
        plans.append((directory, plan))

    if args.verify_only:
        if has_incomplete:
            raise RuntimeError("Verification echouee : fichiers absents ou tailles incorrectes.")
        print("\nVerification OK (presences et tailles ; sans calcul de checksum).")
        return
    if args.dry_run:
        print(f"\nVolume restant estime : {volume(remaining)}")
        print("Estimation par taille : un fichier partiel est compte en entier.")
        return

    args.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.root).free
    print(f"\nEspace libre du filesystem : {volume(free)}", flush=True)
    if remaining > free:
        raise RuntimeError(f"Espace insuffisant : il faut encore environ {volume(remaining)}")

    # Persist all revisions before starting the first large transfer.
    for directory, plan in plans:
        save_plan(directory, plan)
    for directory, plan in plans:
        print(f"\nTelechargement de {plan['repo_id']} ({args.workers} workers)...", flush=True)
        download(plan, directory, args.workers, args.attempts)
        missing = incomplete_files(directory, plan)
        if missing:
            examples = ", ".join(entry["path"] for entry in missing[:3])
            raise RuntimeError(f"{len(missing)} fichiers absents ou tailles incorrectes : {examples}")
        print(f"OK : {plan['repo_id']} (presences et tailles verifiees)", flush=True)
    print("\nLes datasets demandes sont telecharges.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu. Relancer la meme commande pour reprendre.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\nERREUR ({type(error).__name__}) : {error}", file=sys.stderr)
        print("Verifiez la connexion avec hf auth whoami, puis relancez la meme commande.",
              file=sys.stderr)
        sys.exit(1)
