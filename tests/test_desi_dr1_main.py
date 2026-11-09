"""Selection, resumption, corruption handling and real FITS -> HATS integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

from scripts.desi import dr1_main as build


def catalog(n=4):
    result = {"TARGETID": np.arange(n, dtype=np.int64) + 100,
              "TARGET_RA": np.linspace(150, 150.2, n), "TARGET_DEC": np.full(n, 2.0),
              "SURVEY": np.full(n, "main", dtype="U10"),
              "PROGRAM": np.full(n, "dark", dtype="U10"), "HEALPIX": np.full(n, 1001),
              "MAIN_PRIMARY": np.ones(n, dtype=bool), "OBJTYPE": np.full(n, "TGT", dtype="U5"),
              "COADD_FIBERSTATUS": np.zeros(n, dtype=np.int64),
              "ZWARN": np.zeros(n, dtype=np.int64), "SPECTYPE": np.full(n, "GALAXY", dtype="U10")}
    result.update({name: np.ones(n, dtype=np.float32) for name in build.FLOAT_FEATURES})
    return result


def arrays(n=4):
    length = 7781
    return {"flux": np.ones((n, length), dtype=np.float32),
            "ivar": np.ones((n, length), dtype=np.float32),
            "mask": np.zeros((n, length), dtype=bool),
            "lambda": np.tile(np.linspace(3600, 9824, length, dtype=np.float32), (n, 1)),
            "lsf_sigma": np.ones((n, length), dtype=np.float32)}


def test_only_main_but_no_blanket_redshift_cut():
    cat = catalog(8)
    cat["SURVEY"][:3] = ["sv1", "sv2", "sv3"]
    cat["ZWARN"][3] = 4  # uncertain redshift, useful spectrum
    cat["ZWARN"][4] = 512  # NODATA
    cat["TARGET_RA"][5] = np.nan
    cat["MAIN_PRIMARY"][6] = False
    keep, counts = build.selection_masks(cat)
    assert np.flatnonzero(keep).tolist() == [3, 7]
    assert counts["after_main"] == 5
    assert counts["after_has_data"] == 3


def test_masks_and_exact_quality_flags():
    cat = catalog()
    cat["ZWARN"] = np.array([0, 4, 1024, 0])
    cat["Z"][3] = np.nan
    payload = arrays()
    payload["flux"][0, 0] = np.nan
    payload["ivar"][0, 1] = np.inf
    payload["mask"][1] = True
    table, counts = build.make_output(pa.table(cat), payload, {"SNR_B": np.arange(4)})
    assert counts["all_masked_rejected"] == 1
    assert table["TARGETID"].to_pylist() == [100, 102, 103]
    assert table["ZWARN_BITS"].to_pylist() == [0, 1024, 0]
    assert table["ZWARN"].to_pylist() == [True, False, True]
    assert table["redshift_good"].to_pylist() == [True, False, False]
    spec = table["spectrum"][0].as_py()
    assert spec["flux"][0] == 0
    assert spec["mask"][:2] == [True, True]
    assert spec["ivar"][:2] == [0, 0]
    assert np.isfinite(spec["flux"]).all()


@pytest.mark.parametrize("run,raw", [
    ("/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/desi", "/tmp/raw"),
    ("/mnt/ceph/users/polymathic/mronceray", "/tmp/raw"),
    ("/tmp/raw/outputs", "/tmp/raw"), ("/tmp", "/tmp/raw"),
    ("/tmp/external_data/test", "/tmp/raw"),
])
def test_refuses_shared_or_overlapping_targets(run, raw):
    with pytest.raises(ValueError):
        build.validate_paths(run, raw)


def fixture_plan(tmp_path, monkeypatch, *, missing=False, duplicate=False):
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    run.mkdir()
    cat = catalog()
    cat["PROGRAM"][:2] = "bright"
    cat["ZWARN"][1] = 4
    if duplicate:
        cat["TARGETID"][1] = cat["TARGETID"][0]
    zcat = run / "inputs/zall-pix-iron-v1.fits"
    zcat.parent.mkdir()
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(cat))]).writeto(zcat)
    receipt = {"url": build.ZCAT_URL, "sha256": build.digest(zcat)}
    monkeypatch.setattr(build, "fetch_catalog", lambda root: (zcat, receipt))
    if not missing:
        for program in ("bright", "dark"):
            (raw / f"coadd-main-{program}-1001.fits").touch()
    return run, raw, cat


def test_preflight_rejects_missing_files(tmp_path, monkeypatch):
    run, raw, _ = fixture_plan(tmp_path, monkeypatch, missing=True)
    with pytest.raises(RuntimeError, match="coadds missing"):
        build.prepare(run, raw, 2)
    assert not (run / "plan.json").exists()
    assert len(build.read_json(run / "preflight.json")["missing_coadds"]) == 2


def test_preflight_rejects_duplicate_primary(tmp_path, monkeypatch):
    run, raw, _ = fixture_plan(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(RuntimeError, match="duplicate TARGETIDs"):
        build.prepare(run, raw, 2)


def test_resume_and_input_changes_are_detected(tmp_path, monkeypatch):
    run, raw, cat = fixture_plan(tmp_path, monkeypatch)
    build.prepare(run, raw, 2)
    plan, plan_id = build.check_plan(run)
    (run / "parts").mkdir()
    build.init_worker(run, raw, plan_id)
    monkeypatch.setattr(build, "process_spectra", lambda path, ids: (arrays(len(ids)), {}))
    first = build.group_result(plan["groups"][0])
    monkeypatch.setattr(build, "process_spectra", lambda *args: pytest.fail("Resume reread coadd"))
    assert build.group_result(plan["groups"][0]) == first
    (raw / plan["groups"][0]["source"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        build.group_result(plan["groups"][0])
    with pytest.raises(RuntimeError, match="configuration"):
        build.prepare(run, raw, 3)


def test_finished_download_is_verified_and_reused(tmp_path, monkeypatch):
    import requests
    source = tmp_path / "source.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table(catalog()))]).writeto(source)
    content = source.read_bytes()
    monkeypatch.setattr(requests, "head", lambda *a, **kw: SimpleNamespace(
        headers={"Content-Length": str(len(content)), "ETag": "fixed-v1", "Last-Modified": "today"},
        raise_for_status=lambda: None))
    calls = []
    def download(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(content)
    monkeypatch.setattr(build.subprocess, "run", download)
    target, receipt = build.fetch_catalog(tmp_path / "run")
    assert receipt["url"] == build.ZCAT_URL
    assert receipt["sha256"] == build.digest(source)
    assert len(calls) == 1
    assert build.fetch_catalog(tmp_path / "run")[0] == target
    assert len(calls) == 1
    target.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="receipt"):
        build.fetch_catalog(tmp_path / "run")


def write_real_coadd(path, cat, indices, all_masked_id=None):
    from desispec.spectra import Spectra
    from desispec.io import write_spectra

    n = len(indices)
    grid = np.linspace(3600, 9824, 7781)
    slices = {"b": slice(0, 2800), "r": slice(2700, 5100), "z": slice(5000, None)}
    wave = {b: grid[s] for b, s in slices.items()}
    flux = {b: np.full((n, len(w)), 2, dtype=np.float32) for b, w in wave.items()}
    ivar = {b: np.ones_like(v) for b, v in flux.items()}
    mask = {b: np.zeros_like(v, dtype=np.uint32) for b, v in flux.items()}
    ids = cat["TARGETID"][indices]
    all_masked_ids = set(np.atleast_1d(all_masked_id)) if all_masked_id is not None else set()
    for b in slices:
        for i, tid in enumerate(ids):
            if tid in all_masked_ids:
                mask[b][i] = 1
        # Regression: flagged bright spikes must not leak into camera coadds.
        flux[b][0, 0] = 1e10
        mask[b][0, 0] = 1
    profile = np.exp(-0.5 * ((np.arange(11) - 5) / 1.2)**2).astype(np.float32)
    resolution = {b: np.broadcast_to(profile[None, :, None], (n, 11, len(w))).copy()
                  for b, w in wave.items()}
    scores = Table({f"MEDIAN_COADD_SNR_{b}": np.full(n, 3, dtype=np.float32) for b in "BRZ"})
    spectra = Spectra(bands=list(slices), wave=wave, flux=flux, ivar=ivar, mask=mask,
                      resolution_data=resolution, fibermap=Table({"TARGETID": ids}), scores=scores)
    write_spectra(str(path), spectra)


@pytest.mark.parametrize("fully_mask_dark", [False, True])
def test_real_fits_to_validated_hats(tmp_path, monkeypatch, fully_mask_dark):
    run, raw, cat = fixture_plan(tmp_path, monkeypatch)
    for program, indices in (("bright", [0, 1]), ("dark", [2, 3])):
        path = raw / f"coadd-main-{program}-1001.fits"
        path.unlink()  # Fixture sentinels only; no production data involved.
        write_real_coadd(path, cat, indices, all_masked_id=[102, 103] if fully_mask_dark else [103])
    build.prepare(run, raw, 2)
    build.scatter(run, raw, 0, 1)
    build.scatter(run, raw, 1, 1)
    assert sum(build.read_json(run / f"reports/shard-{i:03d}.json")["retained_rows"]
               for i in range(2)) == (2 if fully_mask_dark else 3)
    build.gather(run, workers=1, debug=True)
    report = build.read_json(run / "validation.json")
    assert report["status"] == "PASS"
    assert report["rows"] == (2 if fully_mask_dark else 3)
    assert report["quality"]["all_masked_rejected"] == (2 if fully_mask_dark else 1)
    assert report["quality"]["good_redshift_rows"] == (1 if fully_mask_dark else 2)
    collection = Path(report["collection"])
    assert (collection / "collection.properties").is_file()
    import lsdb
    loaded = lsdb.open_catalog(str(collection))
    assert loaded.margin is not None
    result = loaded.compute()
    assert sorted(result["TARGETID"].tolist()) == ([100, 101] if fully_mask_dark else [100, 101, 102])
    assert len(loaded.crossmatch(loaded, radius_arcsec=1, suffixes=("_left", "_right"),
                                suffix_method="all_columns").compute()) == report["rows"]
    # The flagged spike on the R band's first pixel has no weight; the valid
    # overlapping B measurement supplies the merged flux instead.
    spec = result.loc[result["TARGETID"] == 100, "spectrum"].iloc[0]
    assert spec["flux"].iloc[2700] == pytest.approx(2)
    # Idempotent gather validates the published output instead of overwriting it.
    build.gather(run, workers=1, debug=True)
    staged = run / "building" / build.NAME
    collection.rename(staged)
    with monkeypatch.context() as check:
        check.setattr(sys.modules["hats_import.pipeline"], "pipeline_with_client",
                      lambda *a: pytest.fail("Complete staged HATS was rebuilt"))
        build.gather(run, workers=1, debug=True)
    assert collection.exists() and not staged.exists()


def test_dag_dry_run_uses_personal_root(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "snakemake", "--snakefile",
               str(repo / "scripts/desi/dr1_main.smk"), "--cores", "32", "--dry-run", "--config",
               f"run_root={tmp_path / 'output'}", "shards=2", "max_groups=8"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "prepare" in result.stdout and "scatter" in result.stdout and "gather" in result.stdout
    assert "/MultimodalUniverse_v2_hats/desi" not in result.stdout


def test_launcher_slurm_dry_run_preserves_requested_threads(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bin_dir, raw = tmp_path / "bin", tmp_path / "raw"
    bin_dir.mkdir()
    raw.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/bin/sh\nexit 99\n")  # Must never be invoked in dry-run.
    sbatch.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "DESI_RAW_ROOT": str(raw), "DESI_RUN_ROOT": str(tmp_path / "output"),
           "DESI_WORKERS": "16", "DESI_CONCURRENT": "2"}
    result = subprocess.run(["bash", str(repo / "scripts/desi/launch_dr1_main.sh"),
                             "--smoke", "--dry-run"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "threads: 16" in result.stdout
    assert "mem_mb=64000" in result.stdout
