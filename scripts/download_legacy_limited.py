#!/usr/bin/env python3
"""Serial Legacy restart using curl's average payload rate cap.
Default: dry-run. Pass --run to download. No Xet; one curl at a time.
Requires an existing .hf-download-manifest.json from download_hsc.py.
The 100,000,000 B/s cap is ~0.8 Gb/s, leaving headroom below 1 Gb/s.
curl rate limiting is averaged over several seconds, not kernel traffic shaping.
Pass --unlimited to omit the curl rate cap while keeping transfers serial.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from huggingface_hub import HfApi, get_token, hf_hub_url
from tqdm import tqdm

REPO = "hugging-science/mmu_legacysurvey_dr10_south_21"
RATE = 100_000_000


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
    # Hold a lock for this serial downloader; the previous HF job must be stopped.
    with (directory / ".serial-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        total = len(plan["files"])
        print(f"Depot : {REPO}\nCommit : {plan['revision']}", flush=True)
        if args.unlimited:
            print("Un fichier a la fois. Aucun plafond de debit curl.", flush=True)
        else:
            print("Un fichier a la fois. Plafond curl : 100 Mo/s (~0.8 Gb/s).", flush=True)
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
            for entry, target, initial_partial in pending:
                download(curl, plan, entry, target, progress, initial_partial,
                         unlimited=args.unlimited)
                count += 1
                progress.set_postfix_str(f"{count}/{total} fichiers", refresh=True)
        print("Termine : presences et tailles verifiees, sans checksum.", flush=True)


def partial_size(path, expected):
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0
    if not 0 <= size <= expected:
        raise RuntimeError(f"Fichier partiel de taille incorrecte : {path}")
    return size


def download(curl, plan, entry, target, progress, initial_partial, unlimited=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not treat Xet's preallocated incomplete files as resumable HTTP data.
    partial = target.with_name(target.name + ".curl.incomplete")
    expected = entry["size"]
    reported = initial_partial

    def report():
        nonlocal reported
        current = partial_size(partial, expected)
        progress.update(current - reported)
        reported = current

    for attempt in range(1, 6):
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
        url = hf_hub_url(REPO, filename=entry["path"],
                         repo_type="dataset", revision=plan["revision"])
        with tempfile.NamedTemporaryFile(mode="w", prefix="hf-header-", encoding="utf-8") as headers:
            headers.write("Authorization: Bearer " + token + "\n")
            headers.flush()
            command = [
                curl, "--disable", "--silent", "--show-error", "--fail", "--location", "--globoff",
                "--proto", "=https", "--proto-redir", "=https",
                "--connect-timeout", "60", "--speed-time", "300", "--speed-limit", "1024",
                *([] if unlimited else ["--limit-rate", str(RATE)]),
                "--continue-at", "-",
                "--header", "@" + headers.name, "--output", str(partial), url,
            ]
            child = subprocess.Popen(command)
            try:
                while True:
                    try:
                        result = child.wait(timeout=1)
                        report()
                        break
                    except subprocess.TimeoutExpired:
                        report()
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
            progress.write(f"Echec curl ({result}), nouvelle tentative dans 30 s.")
            time.sleep(30)
    raise RuntimeError(f"Telechargement incomplet : {entry['path']}. "
                       "Verifier le login, le quota et la connexion avant de reprendre.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="Telecharger ; sans cette option, inventaire local uniquement.")
    parser.add_argument("--unlimited", action="store_true",
                        help="Retirer le plafond curl ; un seul transfert a la fois.")
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
