"""Read-only DESI campaign monitor. No scientific dependencies or job submission.

Examples: python3 scripts/desi/status_dr1_main.py smoke_local --watch
          python3 scripts/desi/status_dr1_main.py /path/to/run
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


BASE = Path("/mnt/ceph/users/polymathic/mronceray")
ALIASES = {"full": "desi_dr1_main", "smoke": "desi_dr1_main_smoke",
           "smoke_local": "desi_dr1_main_smoke_local"}


def tail(path, size=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            text = stream.read().decode("utf-8", errors="replace")
        return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
    except OSError:
        return ""


def load(path, issues):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        issues.append(f"Lecture impossible : {path}: {error}")
        return None


def jobs_from_logs(root):
    jobs = {}
    native = root / "workflow/.snakemake/slurm_logs"
    for path in native.rglob("*.log"):
        if re.fullmatch(r"\d+(?:_\d+)?", path.stem):
            jobs[path.stem] = path.relative_to(native).parts[0].removeprefix("rule_")
    # Submitted-but-pending jobs may not have created their Slurm logfile yet.
    controller = root / "workflow/.snakemake/log"
    for path in controller.glob("*.log"):
        for jobid in re.findall(r"submitted with SLURM jobid (\d+(?:_\d+)?)", tail(path, 262144), re.I):
            jobs.setdefault(jobid, "job")
    return jobs


def query_jobs(ids, runner=None):
    """Only query explicitly discovered campaign IDs; never query/cancel all jobs."""
    runner = runner or subprocess.run
    states, issues = {}, []
    if not ids:
        return states, issues
    try:
        queue = runner(["squeue", "--noheader", "--jobs", ",".join(sorted(ids)),
                        "--format", "%i|%T|%M|%R"], capture_output=True, text=True, timeout=10)
        for line in queue.stdout.splitlines():
            fields = line.strip().split("|", 3)
            if len(fields) == 4 and fields[0] in ids:
                states[fields[0]] = fields[1:]
        missing = set(ids) - set(states)
        if missing:
            history = runner(["sacct", "--jobs", ",".join(sorted(missing)), "--noheader", "--parsable2",
                              "--format=JobIDRaw,State,Elapsed,ExitCode"],
                             capture_output=True, text=True, timeout=10)
            for line in history.stdout.splitlines():
                fields = line.strip().split("|")
                if len(fields) >= 4 and fields[0] in missing:
                    states[fields[0]] = [fields[1], fields[2], "exit=" + fields[3]]
            if history.returncode:
                issues.append("Historique Slurm indisponible : " + history.stderr.strip())
        if queue.returncode and not states:
            issues.append("File Slurm indisponible : " + queue.stderr.strip())
    except FileNotFoundError:
        issues.append("Commandes Slurm indisponibles ; charger 'module load slurm'.")
    except (OSError, subprocess.TimeoutExpired) as error:
        issues.append(f"Interrogation Slurm impossible : {error}")
    return states, issues


def snapshot(root, runner=None):
    issues = []
    plan = load(root / "plan.json", issues)
    validation = load(root / "validation.json", issues)
    reports = []
    for path in sorted((root / "reports").glob("shard-*.json")):
        report = load(path, issues)
        if report is not None:
            reports.append(report)
    jobs = jobs_from_logs(root)
    states, slurm_issues = query_jobs(jobs, runner)
    issues.extend(slurm_issues)
    return {"root": root, "plan": plan, "validation": validation, "reports": reports,
            "jobs": jobs, "states": states, "issues": issues}


def render(state):
    root, plan, validation = state["root"], state["plan"], state["validation"]
    complete = bool(validation and validation.get("status") == "PASS")
    lines = [f"DESI DR1 Main — {datetime.now():%H:%M:%S}", f"Campagne : {root}"]
    if plan:
        scope = f"SMOKE ({len(plan['groups'])} coadds)" if plan["scope"] == "smoke" else "COMPLET"
        lines.append(f"Mode : {scope} | {len(plan['groups']):,} coadds | {plan['selected_rows']:,} cibles")
    lines += ["", "Étapes :",
              f"  {'OK' if plan else 'EN ATTENTE / EN COURS'} — préparation du catalogue et contrôle des entrées"]
    if plan:
        count = plan["config"]["shards"]
        reports = state["reports"]
        lines.append(f"  {'OK' if len(reports) == count else 'EN ATTENTE / EN COURS'} — extraction : "
                     f"{len(reports)}/{count} morceaux terminés, "
                     f"{sum(r['retained_rows'] for r in reports):,} spectres validés dans ces morceaux")
    else:
        lines.append("  EN ATTENTE — extraction des spectres")
    lines.append(f"  {'OK' if complete else 'EN ATTENTE / EN COURS' if (root / 'gather-config.json').exists() else 'EN ATTENTE'}"
                 " — assemblage HATS + marge 10″ + validation")
    if state["jobs"]:
        lines += ["", "Jobs de cette campagne (Slurm) :"]
        for jobid in sorted(state["jobs"], key=lambda value: int(value.split("_")[0])):
            status = state["states"].get(jobid, ["ÉTAT NON DISPONIBLE", "", "consulter le log"])
            lines.append(f"  {jobid:>10}  {state['jobs'][jobid]:<10}  {' | '.join(status)}")
    else:
        lines += ["", "Aucun identifiant Slurm trouvé dans les logs pour l'instant."]
    logs = [root / "logs/prepare.log"] + sorted((root / "logs").glob("scatter-*.log")) + [root / "logs/gather.log"]
    lines += ["", "Dernières nouvelles :"]
    for path in logs:
        messages = [line.strip() for line in tail(path).splitlines() if line.strip()]
        if messages:
            lines.append(f"  {path.name}: {messages[-1][-240:]}")
    if complete:
        lines += ["", f"RÉSULTAT : PASS — {validation['rows']:,} spectres",
                  f"Collection : {validation['collection']}"]
        lines += ["  Source : " + warning for warning in validation.get("source_warnings", [])]
    lines += ["", "Logs : " + str(root / "logs")]
    if not plan:
        lines.append("Préparation : voir prepare.log ; la lecture/empreinte du catalogue n'a pas de pourcentage précis.")
    lines.extend("Note : " + issue for issue in state["issues"])
    lines.append("Ctrl-C arrête uniquement cet affichage ; les jobs continuent.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default=os.environ.get("DESI_RUN_ROOT", "full"),
                        help="full, smoke, smoke_local, ou chemin du dossier de campagne")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10, help="Rafraîchissement en secondes (minimum 2)")
    args = parser.parse_args(argv)
    if args.interval < 2:
        parser.error("L'intervalle minimum est de deux secondes")
    root = BASE / ALIASES[args.run] if args.run in ALIASES else Path(args.run).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Campagne introuvable : {root}")
    try:
        while True:
            state = snapshot(root)
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(state), flush=True)
            if not args.watch or (state["validation"] and state["validation"].get("status") == "PASS"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nSuivi arrêté ; aucun job n'a été annulé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
