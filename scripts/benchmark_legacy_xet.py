#!/usr/bin/env python3
"""Benchmark native Xet on a bounded batch of missing Legacy Parquet files.
Default is planning only. --run writes the selected files at their final paths.
No bandwidth cap: file workers and Xet connection concurrency are different.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import time


def select_batch(directory, plan, requested_bytes, workers):
    chosen, byte_count = [], 0
    for entry in plan["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Chemin invalide dans le manifeste.")
        if relative.suffix != ".parquet" or not 256_000_000 <= entry["size"] <= 2_000_000_000:
            continue
        target = directory / relative
        try:
            finished = target.is_file() and target.stat().st_size == entry["size"]
        except FileNotFoundError:
            finished = False
        if finished:
            continue
        # Preserve curl progress; benchmark on other missing files instead.
        if target.with_name(target.name + ".curl.incomplete").exists():
            continue
        chosen.append(entry)
        byte_count += entry["size"]
        if byte_count >= requested_bytes and len(chosen) >= 2 * workers:
            break
    return chosen, byte_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--sample-gb", type=float, default=8)
    parser.add_argument("--high-performance", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.sample_gb <= 64:
        parser.error("--sample-gb doit etre compris entre 0 et 64.")
    repo = "hugging-science/mmu_legacysurvey_dr10_south_21"
    directory = args.root / repo
    plan = json.loads((directory / ".hf-download-manifest.json").read_text())
    if plan["repo_id"] != repo:
        raise ValueError("Manifeste incompatible.")

    # Read Xet flags before importing Hugging Face / hf_xet.
    os.environ.pop("HF_HUB_DISABLE_XET", None)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1" if args.high_performance else "0"
    os.environ["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = str(args.workers)
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import tqdm as hf_tqdm
    import hf_xet
    from tqdm import tqdm

    class QuietFileProgress(hf_tqdm):
        def __init__(self, *a, **kw):
            kw["disable"] = True
            super().__init__(*a, **kw)

    print(f"Noeud : {socket.gethostname()} ; job : {os.environ.get('SLURM_JOB_ID', '?')}", flush=True)
    print(f"Versions : Hub {version('huggingface_hub')} ; Xet {version('hf_xet')}", flush=True)
    print(f"Fichiers simultanes : {args.workers} ; HP : {args.high_performance}", flush=True)
    print(f"Cache Xet : {os.environ.get('HF_XET_CACHE', 'valeur par defaut')}", flush=True)
    print("ATTENTION : aucun plafond de bande passante dans ce benchmark.", flush=True)
    with (directory / ".serial-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Selection des fichiers manquants (ceci peut prendre quelques minutes)...", flush=True)
        entries, amount = select_batch(directory, plan, int(args.sample_gb * 1e9), args.workers)
        if not entries:
            raise RuntimeError("Aucun fichier manquant eligible.")
        print(f"Lot : {len(entries)} fichiers, {amount / 1e9:.3f} Go.", flush=True)
        print("Au moins deux fichiers par worker si le catalogue en contient assez.", flush=True)
        if not args.run:
            print("Plan uniquement. Ajouter --run pour lancer ce lot.", flush=True)
            return
        print(f"Compte : {HfApi(token=True).whoami()['name']}", flush=True)

        def fetch(entry):
            path = Path(hf_hub_download(
                repo_id=repo, repo_type="dataset", revision=plan["revision"],
                filename=entry["path"], local_dir=directory, token=True,
                tqdm_class=QuietFileProgress,
            ))
            if path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Taille incorrecte : {entry['path']}")
            return entry["size"]

        started = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=args.workers)
        try:
            with tqdm(total=amount, unit="B", unit_scale=True, unit_divisor=1000,
                      desc="Lot Xet termine", smoothing=0, dynamic_ncols=True) as progress:
                futures = [pool.submit(fetch, entry) for entry in entries]
                for future in as_completed(futures):
                    progress.update(future.result())
        except BaseException:
            # Abort active native transfers as HF does for Ctrl+C in its Xet path.
            try:
                from huggingface_hub.utils._xet import abort_xet_session
                abort_xet_session()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        elapsed = time.monotonic() - started
        rate = amount / elapsed / 1e6
        print(f"\nRESULTAT : {amount / 1e9:.3f} Go / {elapsed:.2f} s = {rate:.1f} Mo/s utiles")
        print(f"Fichiers : {len(entries)} ; workers : {args.workers} ; HP : {args.high_performance}")
        print("Mesure : octets de fichiers verifies / temps mural, pas debit reseau compresse.")
        print("Cible 720 Mo/s : " + ("atteinte sur ce lot" if rate >= 720 else "NON atteinte sur ce lot"))
        print("Les fichiers du lot sont conserves dans le dataset ; rien a recopier.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrompu. Fichiers termines conserves.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERREUR : {error}", file=sys.stderr)
        sys.exit(1)
