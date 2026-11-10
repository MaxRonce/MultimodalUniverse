#!/usr/bin/env python3
"""Resumable Legacy downloads with bounded curl parallelism.
Default: dry-run. Pass --run to download. No Xet; 1, 2 or 4 curl transfers.
Requires an existing .hf-download-manifest.json from download_hsc.py.
The 100,000,000 B/s cap is ~0.8 Gb/s, leaving headroom below 1 Gb/s.
curl rate limiting is averaged over several seconds, not kernel traffic shaping.
With 2 or 4 workers, caps sum to 200 MB/s (~1.6 Gb/s).
--unlimited removes the rate cap for the selected number of workers.
"""
import argparse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from huggingface_hub import HfApi, get_token, hf_hub_url
from tqdm import tqdm

REPO = "hugging-science/mmu_legacysurvey_dr10_south_21"
RATE = 100_000_000


class TransferFailed(RuntimeError):
    """Retries exhausted for one file; other files may still be accessible."""


def transfer_url(plan, entry):
    # A fresh resolver request avoids reusing an expired cached signed redirect.
    # The immutable revision and destination path stay unchanged.
    return hf_hub_url(REPO, filename=entry["path"], repo_type="dataset",
                      revision=plan["revision"]) + "?download=true&mmu_retry=" + uuid.uuid4().hex


def stop(*_):
    raise KeyboardInterrupt


def run(args):
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl est requis.")
    directory = args.root / REPO
    manifest = directory / ".hf-download-manifest.json"
    plan = json.loads(manifest.read_text())
    if plan["repo_id"] != REPO or not plan.get("revision"):
        raise RuntimeError("Manifeste incompatible.")
    # Hold a repository lock; the previous downloader must be stopped.
    with (directory / ".serial-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        total = len(plan["files"])
        print(f"Depot : {REPO}\nCommit : {plan['revision']}", flush=True)
        rate = None if args.unlimited else (
            RATE if args.workers == 1 else 200_000_000 // args.workers
        )
        if rate is None:
            print(f"{args.workers} transfert(s). Aucun plafond de debit curl.", flush=True)
        else:
            print(f"{args.workers} transfert(s) : {rate / 1e6:g} Mo/s maximum chacun ; "
                  f"{rate * args.workers / 1e6:g} Mo/s au total.", flush=True)
        if not args.run:
            print("Mode inventaire seulement. Ajouter --run pour telecharger.", flush=True)
        count = present = 0
        pending = []
        expected_total = sum(entry["size"] for entry in plan["files"])
        print("Inventaire local : fichiers termines et partiels curl.", flush=True)
        with tqdm(total=total, desc="Inventaire", unit="fichier",
                  mininterval=1, dynamic_ncols=True) as scan:
            for entry in plan["files"]:
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Chemin incompatible dans le manifeste.")
                target = directory / relative
                expected = entry["size"]
                try:
                    complete = target.is_file() and target.stat().st_size == expected
                except FileNotFoundError:
                    complete = False
                if complete:
                    count += 1
                    present += expected
                else:
                    partial = target.with_name(target.name + ".curl.incomplete")
                    initial_partial = partial_size(partial, expected)
                    present += initial_partial
                    pending.append((entry, target, initial_partial))
                scan.update(1)
        print(f"{count:,}/{total:,} fichiers termines ; "
              f"{present / 1e12:.3f}/{expected_total / 1e12:.3f} To presents "
              "(partiels curl inclus).", flush=True)
        if not args.run:
            print(f"A reprendre : {len(pending):,} fichiers ; "
                  f"{(expected_total - present) / 1e12:.3f} To restants.", flush=True)
            return
        # The denominator is the whole repository, not just the current batch.
        # Initial bytes are excluded from the rate/ETA calculation by tqdm.
        with tqdm(total=expected_total, initial=present, desc="Legacy global",
                  unit="B", unit_scale=True, unit_divisor=1000,
                  mininterval=1, miniters=1, smoothing=0,
                  dynamic_ncols=True) as progress:
            progress.set_postfix_str(f"{count}/{total} fichiers", refresh=False)
            tasks = iter(pending)
            task_lock = threading.Lock()
            cancelled = threading.Event()
            failures = []
            failure_path = directory / ".curl-failures.json"

            def save_failures():
                temporary = failure_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({
                    "repo_id": REPO, "revision": plan["revision"], "files": failures
                }, indent=2) + "\n")
                temporary.replace(failure_path)

            def worker():
                nonlocal count
                while not cancelled.is_set():
                    with task_lock:
                        task = next(tasks, None)
                    if task is None:
                        return
                    entry, target, initial_partial = task
                    try:
                        download(curl, plan, entry, target, progress, initial_partial,
                                 rate=rate, cancelled=cancelled)
                    except TransferFailed as error:
                        if not getattr(args, "keep_going", False):
                            raise
                        with task_lock:
                            failures.append({**entry, "error": str(error)})
                            save_failures()
                        progress.write(f"Fichier laisse incomplet : {entry['path']}")
                        continue
                    with tqdm.get_lock():
                        count += 1
                        progress.set_postfix_str(f"{count}/{total} fichiers", refresh=True)

            # Only submit workers, not one Future per catalog file.
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = [pool.submit(worker) for _ in range(args.workers)]
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in done:
                    future.result()
            except BaseException:
                cancelled.set()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        # A run with skipped failures must never be reported as complete.
        if failures:
            raise RuntimeError(f"{len(failures)} fichier(s) encore incomplet(s). "
                               f"Liste : {failure_path}. Relancer pour les retenter.")
        if failure_path.exists():
            failure_path.unlink()
        print("Termine : presences et tailles verifiees, sans checksum.", flush=True)


def partial_size(path, expected):
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    if not 0 <= size <= expected:
        raise RuntimeError(f"Fichier partiel de taille incorrecte : {path}")
    return size


def download(curl, plan, entry, target, progress, initial_partial, rate=RATE, cancelled=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not treat Xet's preallocated incomplete files as resumable HTTP data.
    partial = target.with_name(target.name + ".curl.incomplete")
    expected = entry["size"]
    reported = initial_partial

    def report():
        nonlocal reported
        current = partial_size(partial, expected)
        with tqdm.get_lock():
            progress.update(current - reported)
            reported = current

    for attempt in range(1, 6):
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("Transfert arrete.")
        if partial.exists():
            if partial.stat().st_size > expected:
                raise RuntimeError(f"Fichier partiel trop grand : {partial}")
            if partial.stat().st_size == expected:
                partial.replace(target)
                return
        # whoami lets the Hub manage OAuth refresh before retrieving the token.
        # Authorization is stored in a private temporary file, not process arguments.
        HfApi(token=True).whoami()
        token = get_token()
        if not token:
            raise RuntimeError("Login requis : hf auth login")
        if "\n" in token or "\r" in token:
            raise RuntimeError("Token invalide.")
        url = transfer_url(plan, entry)
        with tempfile.NamedTemporaryFile(mode="w", prefix="hf-header-", encoding="utf-8") as headers:
            headers.write("Authorization: Bearer " + token + "\nCache-Control: no-cache\n")
            headers.flush()
            command = [
                curl, "--disable", "--silent", "--show-error", "--fail", "--location", "--globoff",
                "--proto", "=https", "--proto-redir", "=https",
                "--connect-timeout", "60", "--speed-time", "300", "--speed-limit", "1024",
                *([] if rate is None else ["--limit-rate", str(rate)]),
                "--continue-at", "-",
                "--header", "@" + headers.name, "--output", str(partial), url,
            ]
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Transfert arrete.")
            child = subprocess.Popen(command)
            try:
                while True:
                    if cancelled is not None and cancelled.is_set():
                        raise InterruptedError("Transfert arrete.")
                    try:
                        result = child.wait(timeout=1)
                        report()
                        break
                    except subprocess.TimeoutExpired:
                        report()
                        with tqdm.get_lock():
                            progress.refresh()

            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
        if result == 0 and partial.exists() and partial.stat().st_size == expected:
            partial.replace(target)
            return
        if attempt < 5:
            progress.write(f"Echec curl ({result}) : {entry['path']}, nouvelle tentative dans 30 s.")
            if cancelled is None:
                time.sleep(30)
            elif cancelled.wait(30):
                raise InterruptedError("Transfert arrete.")
    raise TransferFailed(f"Telechargement incomplet : {entry['path']}. "
                       "Verifier le login, le quota et la connexion avant de reprendre.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="Telecharger ; sans cette option, inventaire local uniquement.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Poursuivre les autres fichiers apres epuisement des tentatives ; "
                             "enregistrer les echecs dans .curl-failures.json.")
    parser.add_argument("--unlimited", action="store_true",
                        help="Retirer le plafond curl pour tous les transferts selectionnes.")
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=1,
                        help="Transferts simultanes ; a 2 ou 4, plafond total 200 Mo/s.")
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    try:
        run(arguments)
    except KeyboardInterrupt:
        print("\nArrete. Fichiers termines conserves.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERREUR : {error}", file=sys.stderr)
        sys.exit(1)
