#!/usr/bin/env python3
"""Download complete, authenticated HSC HATS repositories to Ceph.

Requires huggingface_hub>=1.0,<2 and hf-xet>=1.2,<2.
See docs/DOWNLOAD_HSC.md for installation, login and background execution.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from huggingface_hub import HfApi, snapshot_download
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


def get_plan(api, repo_id, directory, verify_only=False):
    manifest = directory / MANIFEST
    if manifest.exists():
        plan = json.loads(manifest.read_text())
        if plan["repo_id"] != repo_id:
            raise RuntimeError(f"Depot incompatible dans {manifest}")
        return plan
    if verify_only:
        raise RuntimeError(f"Manifeste absent : {manifest}")
    info = api.dataset_info(repo_id, files_metadata=True)
    if not info.sha or not info.siblings:
        raise RuntimeError(f"Inventaire vide pour {repo_id}")
    files = []
    for entry in info.siblings:
        if entry.size is None:
            raise RuntimeError(f"Taille inconnue : {repo_id}/{entry.rfilename}")
        files.append({"path": entry.rfilename, "size": entry.size})
    return {"repo_id": repo_id, "revision": info.sha, "files": files}


def save_plan(directory, plan):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n")
    temporary.replace(directory / MANIFEST)


def incomplete_files(directory, plan):
    return [
        entry for entry in plan["files"]
        if not (directory / entry["path"]).is_file()
        or (directory / entry["path"]).stat().st_size != entry["size"]
    ]


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
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument(
        "--root", type=Path,
        default=Path(os.environ.get(
            "HSC_DEST", "/mnt/ceph/users/polymathic/mronceray/hsc_pdr3"
        )),
    )
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--attempts", type=positive_int, default=5)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true",
                       help="Inventaire distant uniquement, aucun fichier telecharge.")
    modes.add_argument("--verify-only", action="store_true",
                       help="Controle local des presences/tailles, sans reseau.")
    args = parser.parse_args()

    api = HfApi(token=True)
    if not args.verify_only:
        account = api.whoami()
        print(f"Compte Hugging Face : {account['name']}", flush=True)

    selected = DATASETS if args.dataset == "all" else {
        args.dataset: DATASETS[args.dataset]
    }
    plans = []
    remaining = 0
    for repo_id in selected.values():
        directory = args.root / repo_id
        plan = get_plan(api, repo_id, directory, args.verify_only)
        missing = incomplete_files(directory, plan)
        pending_size = sum(entry["size"] for entry in missing)
        remaining += pending_size
        print(f"\n{repo_id}\n  Commit : {plan['revision']}\n  Dossier : {directory}")
        print(f"  Total : {len(plan['files'])} fichiers, "
              f"{volume(sum(entry['size'] for entry in plan['files']))}")
        print(f"  Absents ou de taille incorrecte : {len(missing)}, "
              f"{volume(pending_size)}", flush=True)
        plans.append((directory, plan))

    if args.verify_only:
        if any(incomplete_files(directory, plan) for directory, plan in plans):
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

    # Persist both revisions before starting the first large transfer.
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
