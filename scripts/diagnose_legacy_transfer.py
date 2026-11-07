#!/usr/bin/env python3
"""Bounded HTTP comparison: /dev/null, local /tmp, Ceph, then /dev/null.
No dataset files are modified. Reads the existing download manifest.
Four sequential 256 MiB range requests, each capped at 100 MB/s.
Private temporary files are removed automatically. No Xet traffic.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

from huggingface_hub import HfApi, get_token, hf_hub_url

REPO = "hugging-science/mmu_legacysurvey_dr10_south_21"
SAMPLE_BYTES = 256 * 1024 * 1024
RATE = 100_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl absent")
    directory = args.root / REPO
    manifest = json.loads((directory / ".hf-download-manifest.json").read_text())
    if manifest["repo_id"] != REPO:
        raise RuntimeError("Manifeste incompatible")
    entry = next(f for f in manifest["files"]
                 if f["path"].endswith(".parquet") and f["size"] >= SAMPLE_BYTES)
    account = HfApi(token=True).whoami()
    token = get_token()
    if not token or "\n" in token or "\r" in token:
        raise RuntimeError("Authentification HF invalide")
    url = hf_hub_url(REPO, filename=entry["path"], repo_type="dataset",
                     revision=manifest["revision"])
    print(f"Noeud : {socket.gethostname()}", flush=True)
    print(f"Job : {os.environ.get('SLURM_JOB_ID', 'non renseigne')}", flush=True)
    print(f"CPUs demandes : {os.environ.get('SLURM_CPUS_PER_TASK', 'non renseigne')}", flush=True)
    print(f"Compte : {account['name']}", flush=True)
    print(subprocess.check_output([curl, "--version"], text=True).splitlines()[0], flush=True)
    print(f"Fichier distant : {entry['path']}", flush=True)
    print("4 requetes sequentielles de 256 Mio, plafond 100 Mo/s chacune.", flush=True)
    print("Arreter l'autre telechargement avant ce test.", flush=True)
    with tempfile.TemporaryDirectory(prefix="legacy-net-check-", dir="/tmp") as local, \
            tempfile.TemporaryDirectory(prefix=".legacy-net-check-", dir=directory) as ceph:
        header = Path(local) / "auth.headers"
        descriptor = os.open(header, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write("Authorization: Bearer " + token + "\n")
        cases = [
            ("RESEAU avant (/dev/null)", Path("/dev/null")),
            ("DISQUE LOCAL (/tmp)", Path(local) / "sample"),
            ("CEPH", Path(ceph) / "sample"),
            ("RESEAU apres (/dev/null)", Path("/dev/null")),
        ]
        for label, output in cases:
            print(f"\n{label}...", flush=True)
            command = [
                curl, "--disable", "--silent", "--show-error", "--fail", "--location",
                "--globoff", "--proto", "=https", "--proto-redir", "=https",
                "--connect-timeout", "30", "--max-time", "180",
                "--range", f"0-{SAMPLE_BYTES - 1}", "--max-filesize", str(SAMPLE_BYTES),
                "--limit-rate", str(RATE), "--header", "@" + str(header),
                "--output", str(output),
                "--write-out",
                "%{http_code} %{size_download} %{speed_download} %{time_total} %{time_starttransfer} %{remote_ip}",
                url,
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=200)
            if result.returncode:
                print(f"ECHEC curl code {result.returncode}: {result.stderr.strip()}", flush=True)
                continue
            fields = result.stdout.strip().split()
            if len(fields) != 6:
                raise RuntimeError("Statistiques curl inattendues")
            status, amount, speed, seconds, first_byte, address = fields
            if status != "206" or int(float(amount)) != SAMPLE_BYTES:
                raise RuntimeError(f"Reponse inattendue : HTTP {status}, {amount} octets")
            print(f"HTTP {status}; {float(amount) / 1e6:.1f} Mo; "
                  f"{float(speed) / 1e6:.1f} Mo/s; duree {float(seconds):.2f} s; "
                  f"premier octet {float(first_byte):.2f} s; serveur {address}", flush=True)
            if output != Path("/dev/null"):
                print("Debit curl : ne mesure pas un fsync durable du stockage.", flush=True)
        print("\nFin. Les fichiers temporaires du test vont etre supprimes.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Test interrompu.")
    except Exception as error:
        raise SystemExit(f"ERREUR : {error}")
