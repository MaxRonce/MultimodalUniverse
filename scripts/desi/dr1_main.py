"""Auditable DESI DR1 Main -> MMU HATS production (see docs/BUILD_DESI_DR1.md).

The shared coadds are read-only. A run owns its corrected v1 zcatalog,
compact selection, atomic/resumable group parquets, HATS collection and QA.
No S/N, redshift or spectral-class cut is imposed on otherwise usable spectra.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from filelock import FileLock

from mmu.hats_import import np_to_pyarrow_list, to_native_endian
from scripts.desi.build_parent_sample_hats import BAD_HANDLES, FLOAT_FEATURES, estimate_lsf_sigma


NAME = "desi_dr1_main"
ZCAT_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits"
RAW_ROOT = "/mnt/ceph/users/polymathic/external_data/astro/DESI_DR1"
DEFAULT_RUN = "/mnt/ceph/users/polymathic/mronceray/desi_dr1_main"
REQUIRED = ["TARGETID", "TARGET_RA", "TARGET_DEC", "SURVEY", "PROGRAM", "HEALPIX",
            "MAIN_PRIMARY", "OBJTYPE", "COADD_FIBERSTATUS", "ZWARN", "SPECTYPE"] + FLOAT_FEATURES
OPTIONAL = ["SUBTYPE", "DELTACHI2", "DESI_TARGET", "BGS_TARGET", "MWS_TARGET",
            "SCND_TARGET", "ZCAT_PRIMARY", "COADD_NUMEXP", "COADD_EXPTIME",
            "COADD_NUMNIGHT", "COADD_NUMTILE"]
_STATE = None


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def strings(values):
    return np.char.strip(np.asarray(values).astype(str))


def signature(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def software_versions():
    return {name: version(name) for name in ("numpy", "pyarrow", "astropy", "desispec",
                                              "hats", "hats-import", "lsdb")}


def validate_paths(run, raw):
    """Never put output inside the mirror, or use a shared production root."""
    run, raw = Path(run).resolve(), Path(raw).resolve()
    if run == raw or run in raw.parents or raw in run.parents:
        raise ValueError("Run and raw mirror must be disjoint directories")
    shared = Path("/mnt/ceph/users/polymathic")
    personal = shared / "mronceray"
    if shared in run.parents and (personal not in run.parents or run == personal):
        raise ValueError("Use a dedicated run below /mnt/ceph/users/polymathic/mronceray")
    if "external_data" in run.parts:
        raise ValueError("Run must not be in external_data")
    return run, raw


def fetch_catalog(run):
    """Download ONLY the official corrected v1 summary, with curl resume.

    The unknown-version shared summary is intentionally not used. A receipt
    pins both HTTP identity and the SHA256 of the completed personal copy.
    """
    import requests

    target = run / "inputs/zall-pix-iron-v1.fits"
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(".download.json")
    if target.exists():
        receipt = read_json(receipt_path)
        if (receipt["url"] != ZCAT_URL or receipt["sha256"] != digest(target)
                or receipt["bytes"] != target.stat().st_size):
            raise RuntimeError("Completed v1 catalog differs from its download receipt")
        return target, receipt
    response = requests.head(ZCAT_URL, headers={"Accept-Encoding": "identity"}, timeout=120)
    response.raise_for_status()
    identity = {"url": ZCAT_URL, "bytes": int(response.headers["Content-Length"]),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified")}
    partial = target.with_suffix(".fits.part")
    identity_path = target.with_suffix(".http.json")
    if partial.exists() and (not identity_path.exists() or read_json(identity_path) != identity):
        raise RuntimeError("Remote catalog identity changed; use a new run directory")
    atomic_json(identity_path, identity)
    if partial.exists() and partial.stat().st_size > identity["bytes"]:
        raise RuntimeError("Oversized partial catalog; use a new run directory")
    if not partial.exists() or partial.stat().st_size != identity["bytes"]:
        subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
                        "--connect-timeout", "60", "--speed-limit", "1024", "--speed-time", "180",
                        "--continue-at", "-", "--output", str(partial), ZCAT_URL], check=True)
    if partial.stat().st_size != identity["bytes"]:
        raise RuntimeError("Incomplete catalog download")
    with fits.open(partial, memmap=True) as hdus:
        hdus.verify("exception")
        if "MAIN_PRIMARY" not in hdus[1].columns.names:
            raise RuntimeError("Downloaded file is not the expected DR1 redshift catalog")
    receipt = {**identity, "sha256": digest(partial)}
    atomic_json(receipt_path, receipt)
    os.replace(partial, target)
    return target, receipt


def selection_masks(cat):
    """Cumulative selection, retaining uncertain redshifts and all Main programs."""
    n = len(cat["TARGETID"])
    good = np.ones(n, dtype=bool)
    counts = {"input_rows": n}
    tests = [
        ("main", strings(cat["SURVEY"]) == "main"),
        ("primary", np.asarray(cat["MAIN_PRIMARY"], dtype=bool)),
        ("science_target", strings(cat["OBJTYPE"]) == "TGT"),
        ("fiber_ok", np.asarray(cat["COADD_FIBERSTATUS"]) == 0),
        ("has_data", (np.asarray(cat["ZWARN"], dtype=np.int64) & (1 << 9)) == 0),
    ]
    ra, dec = cat["TARGET_RA"], cat["TARGET_DEC"]
    tests.append(("coordinates_ok", np.isfinite(ra) & np.isfinite(dec)
                  & (ra >= 0) & (ra < 360) & (dec >= -90) & (dec <= 90)))
    available = np.ones(n, dtype=bool)
    for program, pixels in BAD_HANDLES.items():
        available &= ~((strings(cat["PROGRAM"]) == program) & np.isin(cat["HEALPIX"], pixels))
    tests.append(("known_available_handles", available))
    for name, test in tests:
        good &= test
        counts["after_" + name] = int(good.sum())
    return good, counts


def arrow_columns(cat, names, mask):
    columns = {}
    for name in names:
        values = np.asarray(cat[name][mask])
        if values.dtype.kind in "SU":
            values = strings(values)
        else:
            values = to_native_endian(values)
        columns[name] = pa.array(values)
    return pa.table(columns)


def prepare(run, raw, shards, max_groups=None):
    plan_path = run / "plan.json"
    config = {"raw_root": str(raw), "shards": shards, "max_groups": max_groups,
              "builder_sha256": digest(__file__), "zcatalog_url": ZCAT_URL}
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["config"] != config or plan["selection_sha256"] != digest(run / "selection.parquet"):
            raise RuntimeError("Run configuration/code/selection changed; use a new run")
        if plan["catalog"]["sha256"] != digest(run / "inputs/zall-pix-iron-v1.fits"):
            raise RuntimeError("Catalog changed since preparation")
        print(f"Reusing plan: {len(plan['groups'])} groups, {plan['selected_rows']} targets", flush=True)
        return
    zcat, receipt = fetch_catalog(run)
    with fits.open(zcat, memmap=True) as hdus:
        cat = hdus[1].data
        missing = sorted(set(REQUIRED) - set(cat.names))
        if missing:
            raise RuntimeError(f"Missing catalog columns: {missing}")
        names = list(dict.fromkeys(REQUIRED + [n for n in cat.names
                         if n in OPTIONAL or n.startswith("TSNR2_")]))
        mask, counts = selection_masks(cat)
        table = arrow_columns(cat, names, mask)
        header = {key: hdus[1].header.get(key, hdus[0].header.get(key))
                  for key in ("SPECPROD", "SURVEY", "EXTNAME", "NAXIS2")}
    if table.num_rows == 0:
        raise RuntimeError("No targets selected")
    tids = table["TARGETID"].to_numpy()
    if len(np.unique(tids)) != len(tids):
        raise RuntimeError("MAIN_PRIMARY selection contains duplicate TARGETIDs")
    program = table["PROGRAM"].to_numpy()
    hp = table["HEALPIX"].to_numpy()
    order = np.lexsort((tids, hp, program))
    table = table.take(pa.array(order))
    program, hp = program[order], hp[order]
    boundaries = np.r_[0, np.flatnonzero((program[1:] != program[:-1]) | (hp[1:] != hp[:-1])) + 1,
                       table.num_rows]
    intervals = list(zip(boundaries[:-1], boundaries[1:]))
    if max_groups is not None:
        intervals = intervals[:max_groups]
    groups, missing = [], []
    for idx, (start, end) in enumerate(intervals):
        filename = f"coadd-main-{program[start]}-{hp[start]}.fits"
        source = raw / filename
        if not source.is_file():
            missing.append(filename)
            continue
        groups.append({"index": idx, "source": filename, "rows": int(end - start),
                       "source_stat": signature(source)})
    atomic_json(run / "preflight.json", {"selection_counts": counts, "missing_coadds": missing,
                                         "required_groups": len(intervals), "catalog": receipt})
    if missing:
        raise RuntimeError(f"{len(missing)} required coadds missing: see preflight.json")
    selection_path = run / "selection.parquet"
    temporary = selection_path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(temporary, table.schema, compression="zstd") as writer:
        for start, end in intervals:
            # One row group per input coadd: workers need no 21 GB catalog copies.
            writer.write_table(table.slice(int(start), int(end - start)), row_group_size=int(end - start))
    os.replace(temporary, selection_path)
    plan = {"config": config, "catalog": receipt, "fits_header": header,
            "software": software_versions(),
            "selection_counts": counts, "groups": groups,
            "selected_rows": sum(g["rows"] for g in groups),
            "selection_sha256": digest(selection_path), "schema": str(table.schema),
            "scope": "smoke" if max_groups else "full"}
    atomic_json(plan_path, plan)
    print(f"PREPARE PASS: {len(groups)} coadds, {plan['selected_rows']} targets", flush=True)


def clean_arrays(arrays):
    """Mask invalid pixels; exclude only entirely unusable spectra.

    Finite measured flux is preserved even on masked pixels. Non-finite flux
    becomes zero and all masked pixels have ivar=0. mask=True always means bad.
    """
    wave = arrays["lambda"]
    if not np.isfinite(wave).all() or not (np.diff(wave, axis=1) > 0).all():
        raise ValueError("Wavelength grid must be finite and strictly increasing")
    lsf = arrays["lsf_sigma"]
    if not np.isfinite(lsf).all() or not (lsf > 0).all():
        raise ValueError("Invalid LSF estimate")
    flux, ivar = arrays["flux"], arrays["ivar"]
    bad = arrays["mask"] | ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 1e-6)
    flux = np.where(np.isfinite(flux), flux, 0).astype(np.float32)
    ivar = np.where(bad, 0, ivar).astype(np.float32)
    keep = (~bad).any(axis=1)
    cleaned = {**arrays, "flux": flux, "ivar": ivar, "mask": bad}
    return cleaned, keep


def process_spectra(path, target_ids):
    import desispec.io
    from desispec.coaddition import coadd_cameras

    spectra = desispec.io.read_spectra(str(path)).select(targets=target_ids)
    # Do this BEFORE camera coaddition: desispec can clear the output mask
    # wherever output ivar>0. A flagged/non-finite native pixel must therefore
    # contribute zero statistical weight, including in B/R/Z overlaps.
    for band in spectra.bands:
        bad = ((spectra.mask[band] != 0) | ~np.isfinite(spectra.flux[band])
               | ~np.isfinite(spectra.ivar[band]) | (spectra.ivar[band] <= 1e-6))
        spectra.ivar[band][bad] = 0
        spectra.flux[band][~np.isfinite(spectra.flux[band])] = 0
    combined = coadd_cameras(spectra)
    ids = np.asarray(combined.target_ids())
    if len(ids) != len(target_ids) or len(np.unique(ids)) != len(ids) or set(ids) != set(target_ids):
        raise RuntimeError(f"Coadd TARGETIDs do not match selection: {path}")
    sort = np.argsort(ids)
    order = sort[np.searchsorted(ids[sort], target_ids)]
    flux = np.asarray(combined.flux["brz"], dtype=np.float32)[order]
    n, length = flux.shape
    sigma = estimate_lsf_sigma(np.asarray(combined.resolution_data["brz"])[order])
    arrays = {
        "flux": flux,
        "ivar": np.asarray(combined.ivar["brz"], dtype=np.float32)[order],
        "mask": np.asarray(combined.mask["brz"])[order] != 0,
        "lambda": np.tile(np.asarray(combined.wave["brz"], dtype=np.float32), (n, 1)),
        "lsf_sigma": np.full((n, length), sigma, dtype=np.float32),
    }
    snr = {f"SNR_{band}": np.asarray(combined.scores[f"MEDIAN_COADD_SNR_{band}"],
                                   dtype=np.float32)[order] for band in "BRZ"}
    return arrays, snr


def make_output(selected, arrays, snr):
    arrays, keep = clean_arrays(arrays)
    good_pixels = (~arrays["mask"]).sum(axis=1)
    total = len(keep)
    data = {n: selected[n].filter(pa.array(keep)) for n in selected.column_names}
    data["ra"], data["dec"] = data.pop("TARGET_RA"), data.pop("TARGET_DEC")
    ids = selected["TARGETID"].to_numpy()[keep]
    data["object_id"] = pa.array(ids.astype(str))
    # Keep the historical MMU boolean semantics AND the exact DESI bitmask.
    bits = selected["ZWARN"].to_numpy()[keep].astype(np.int64)
    data["ZWARN_BITS"] = pa.array(bits)
    data["ZWARN"] = pa.array(bits == 0)
    data["redshift_good"] = pa.array((bits == 0) & np.isfinite(selected["Z"].to_numpy()[keep]))
    data["n_good_pixels"] = pa.array(good_pixels[keep].astype(np.int32))
    data["good_pixel_fraction"] = pa.array((good_pixels[keep] / arrays["flux"].shape[1]).astype(np.float32))
    for name, values in snr.items():
        data[name] = pa.array(values[keep])
    fields = ["flux", "ivar", "lsf_sigma", "lambda", "mask"]
    data["spectrum"] = pa.StructArray.from_arrays(
        [np_to_pyarrow_list(arrays[f][keep]) for f in fields], names=fields)
    for name in FLOAT_FEATURES:
        data[name] = data[name].cast(pa.float32())
    return pa.table(data), {"input_rows": total, "retained_rows": int(keep.sum()),
                            "all_masked_rejected": int((~keep).sum()),
                            "good_redshift_rows": int(data["redshift_good"].to_numpy(zero_copy_only=False).sum()),
                            "masked_pixels": int(arrays["mask"][keep].sum())}


def init_worker(run, raw, plan_id):
    global _STATE
    _STATE = (Path(run), Path(raw), plan_id, pq.ParquetFile(Path(run) / "selection.parquet"))


def group_result(group):
    run, raw, plan_id, selection = _STATE
    idx = group["index"]
    path = run / "parts" / f"part-{idx:06d}.parquet"
    source = raw / group["source"]
    if signature(source) != group["source_stat"]:
        raise RuntimeError(f"Input coadd changed: {source}")
    if path.exists():
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"mmu_plan") != plan_id.encode():
            raise RuntimeError(f"Stale parquet: {path}; use a new run")
        result = json.loads(metadata[b"mmu_quality"])
        if (result["input_rows"] != group["rows"]
                or pq.ParquetFile(path).metadata.num_rows != result["retained_rows"]):
            raise RuntimeError(f"Invalid resumed parquet: {path}")
        return result
    selected = selection.read_row_group(idx)
    arrays, snr = process_spectra(source, selected["TARGETID"].to_numpy())
    table, result = make_output(selected, arrays, snr)
    table = table.replace_schema_metadata({b"mmu_plan": plan_id.encode(),
                                           b"mmu_quality": json.dumps(result).encode()})
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=256)
    os.replace(temporary, path)
    return result


def check_plan(run):
    plan = read_json(run / "plan.json")
    if plan["config"]["builder_sha256"] != digest(__file__):
        raise RuntimeError("Builder changed; use the original checkout or a new run")
    if plan["software"] != software_versions():
        raise RuntimeError("Scientific dependencies changed; restore the recorded versions")
    return plan, digest(run / "plan.json")


def scatter(run, raw, shard, workers):
    plan, plan_id = check_plan(run)
    if str(raw) != plan["config"]["raw_root"] or not 0 <= shard < plan["config"]["shards"]:
        raise ValueError("Wrong mirror or shard index")
    if digest(run / "selection.parquet") != plan["selection_sha256"]:
        raise RuntimeError("Selection parquet changed")
    groups = plan["groups"][shard::plan["config"]["shards"]]
    (run / "parts").mkdir(exist_ok=True)
    totals = {k: 0 for k in ("input_rows", "retained_rows", "all_masked_rejected",
                             "good_redshift_rows", "masked_pixels")}
    # Each child opens only its needed row groups; no large inherited FITS
    # catalog and no fork of PyArrow's multithreaded runtime.
    with mp.get_context("spawn").Pool(workers, initializer=init_worker,
                                     initargs=(str(run), str(raw), plan_id)) as pool:
        for i, result in enumerate(pool.imap_unordered(group_result, groups, chunksize=1), 1):
            for name in totals:
                totals[name] += result[name]
            if i % 25 == 0 or i == len(groups):
                print(f"shard {shard}: {i}/{len(groups)} groups, {totals['retained_rows']} rows", flush=True)
    atomic_json(run / f"reports/shard-{shard:03d}.json",
                {"plan_sha256": plan_id, "groups": len(groups), **totals})


def validate_collection(collection, expected_rows):
    import lsdb

    collection = Path(collection)
    catalog = lsdb.open_catalog(str(collection))
    if catalog.margin is None:
        raise RuntimeError("Collection has no default margin cache")
    main = collection / NAME
    for relative in ("hats.properties", "partition_info.csv", "skymap.fits",
                     "dataset/_metadata", "dataset/_common_metadata"):
        if not (main / relative).is_file():
            raise RuntimeError(f"Missing HATS artifact: {main / relative}")
    parts = sorted((main / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    if total != expected_rows or catalog.hc_structure.catalog_info.total_rows != total:
        raise RuntimeError(f"HATS row count {total} != expected {expected_rows}")
    for path in parts:
        for batch in pq.ParquetFile(path).iter_batches(
                columns=["SURVEY", "MAIN_PRIMARY", "COADD_FIBERSTATUS", "ZWARN", "ZWARN_BITS"]):
            values = batch.to_pydict()
            if (any(s != "main" for s in values["SURVEY"]) or not all(values["MAIN_PRIMARY"])
                    or any(values["COADD_FIBERSTATUS"]) or values["ZWARN"] != [v == 0 for v in values["ZWARN_BITS"]]):
                raise RuntimeError(f"Selection/quality inconsistency in {path}")
    for index in sorted({0, len(parts) // 2, len(parts) - 1}):
        rows = next(pq.ParquetFile(parts[index]).iter_batches(batch_size=4, columns=["spectrum"]))
        for spec in rows.to_pydict()["spectrum"]:
            lengths = {len(v) for v in spec.values()}
            if len(lengths) != 1 or not lengths or next(iter(lengths)) != 7781:
                raise RuntimeError("Unexpected spectrum schema/length")
            if not all(np.isfinite(spec[f]).all() for f in ("flux", "ivar", "lambda", "lsf_sigma")):
                raise RuntimeError("Non-finite spectral payload")
            if not np.any(~np.asarray(spec["mask"])):
                raise RuntimeError("Entirely masked spectrum reached final catalog")
    return {"status": "PASS", "rows": total, "partitions": len(parts),
            "default_margin_arcsec": 10, "full_scalar_scan": True,
            "spectral_sample_partitions": min(3, len(parts))}


def gather(run, workers, pixel_threshold=8192, debug=False):
    import dask
    from hats_import import CollectionArguments
    from hats_import.catalog.file_readers import ParquetPyarrowReader
    from hats_import.pipeline import pipeline_with_client
    from mmu.hats_import import _client_ctx

    plan, plan_id = check_plan(run)
    settings_path = run / "gather-config.json"
    settings = {"plan_sha256": plan_id, "pixel_threshold": pixel_threshold, "margin_arcsec": 10}
    if settings_path.exists() and read_json(settings_path) != settings:
        raise RuntimeError("HATS partition settings changed; use a new run")
    atomic_json(settings_path, settings)
    reports = [read_json(run / f"reports/shard-{i:03d}.json")
               for i in range(plan["config"]["shards"])]
    if any(r["plan_sha256"] != plan_id for r in reports):
        raise RuntimeError("Shard reports do not belong to the current plan")
    if sum(r["input_rows"] for r in reports) != plan["selected_rows"]:
        raise RuntimeError("Incomplete scatter")
    files = [run / "parts" / f"part-{g['index']:06d}.parquet" for g in plan["groups"]]
    expected = sum(r["retained_rows"] for r in reports)
    if expected == 0:
        raise RuntimeError("No usable spectra")
    if len(list((run / "parts").glob("*.parquet"))) != len(files):
        raise RuntimeError("Unexpected group parquet files in this run")
    for group, path in zip(plan["groups"], files):
        f = pq.ParquetFile(path)
        metadata = f.schema_arrow.metadata or {}
        result = json.loads(metadata[b"mmu_quality"])
        if (metadata.get(b"mmu_plan") != plan_id.encode() or result["input_rows"] != group["rows"]
                or result["retained_rows"] != f.metadata.num_rows):
            raise RuntimeError(f"Invalid group parquet: {path}")
    if sum(pq.ParquetFile(p).metadata.num_rows for p in files) != expected:
        raise RuntimeError("Shard reports and parquets disagree")
    final = run / "hats" / NAME
    if final.exists():
        if (final / "provenance.json").exists() and read_json(final / "provenance.json")["plan_sha256"] != plan_id:
            raise RuntimeError("Existing published collection belongs to a different plan")
        report = validate_collection(final, expected)
    else:
        # Stable work paths allow hats-import to resume a failed gather. The
        # complete collection becomes visible at its final path only after QA.
        staged = run / "building" / NAME
        # A crash after native HATS completion but before our QA/publish must
        # not try to overwrite the already complete staged collection.
        if not (staged / "collection.properties").is_file():
            args = (CollectionArguments(output_artifact_name=NAME, output_path=str(run / "building"),
                                        tmp_dir=str(run / "hats_work"),
                                        write_table_kwargs={"compression": "zstd"})
                    .catalog(input_file_list=[str(p) for p in files if pq.ParquetFile(p).metadata.num_rows],
                             file_reader=ParquetPyarrowReader(chunksize=256),
                             ra_column="ra", dec_column="dec", sort_columns="TARGETID",
                             expected_total_rows=expected, pixel_threshold=pixel_threshold,
                             lowest_healpix_order=4)
                    .add_margin(margin_threshold=10.0, is_default=True))
            with dask.config.set({"temporary-directory": str(run / "dask_tmp")}):
                with _client_ctx(None, workers, debug) as client:
                    pipeline_with_client(args, client)
        report = validate_collection(staged, expected)
        final.parent.mkdir(exist_ok=True)
        os.replace(staged, final)
    totals = {k: sum(r[k] for r in reports) for k in
              ("input_rows", "retained_rows", "all_masked_rejected", "good_redshift_rows", "masked_pixels")}
    atomic_json(final / "provenance.json", {"plan_sha256": plan_id, "plan": plan,
                                           "release": "DR1", "survey": "main", "license": "CC-BY-4.0",
                                           "quality": totals, "lsf": "per-coadd mean Gaussian sigma, pixels"})
    (final / "README.md").write_text(
        f"# DESI DR1 Main — MMU HATS\n\n{expected:,} spectra; default margin: 10 arcsec.\n\n"
        "Selection: SURVEY=main, MAIN_PRIMARY, OBJTYPE=TGT, COADD_FIBERSTATUS=0; "
        "NODATA and invalid coordinates excluded. Known missing handles excluded. "
        "All Main programs and spectral classes retained; no S/N or redshift-success cut. "
        "Entirely masked spectra excluded. See provenance.json for counts and inputs.\n\n"
        "spectrum.mask=True means bad; masked ivar=0, non-finite flux replaced by zero. "
        "ZWARN is the historical MMU boolean (True = no warning); ZWARN_BITS preserves "
        "the original DESI integer. redshift_good additionally requires a finite Z. "
        "LSF is the MMU approximate mean Gaussian sigma in pixels, not the full resolution matrix. "
        "No denoising, extinction correction, normalization or rest-frame shift.\n\n"
        "DESI data: CC BY 4.0. Cite DESI DR1 and Multimodal Universe; "
        "acknowledgments: https://data.desi.lbl.gov/doc/acknowledgments/\n"
        "Known DR1 caveats: https://data.desi.lbl.gov/doc/releases/dr1/known-issues/\n"
    )
    atomic_json(run / "validation.json", {**report, "collection": str(final),
                                          "scope": plan["scope"], "quality": totals})
    print(f"HATS PASS: {final} ({expected:,} rows)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "scatter", "gather"])
    parser.add_argument("--run-root", default=DEFAULT_RUN)
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.shards < 1 or args.workers < 1 or (args.max_groups is not None and args.max_groups < 1):
        parser.error("shards, workers and max-groups must be positive")
    run, raw = validate_paths(args.run_root, args.raw_root)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run / f".{args.stage}-{args.shard if args.stage == 'scatter' else 0}.lock"), timeout=0):
        if args.stage == "prepare":
            prepare(run, raw, args.shards, args.max_groups)
        elif args.stage == "scatter":
            scatter(run, raw, args.shard, args.workers)
        else:
            gather(run, args.workers, args.pixel_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
