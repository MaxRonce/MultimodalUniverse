"""Count JADES sources by AB magnitude and catalog aperture SNR, read-only.

Read scalar FITS columns only, so this can run alongside mosaic extraction.
Optionally match the checkpointed Parquet IDs to report the extracted subset
separately. No cuts or modifications are applied to the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits


JWST_BANDS = (
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W", "F210M",
    "F250M", "F277W", "F300M", "F335M", "F356W", "F410M", "F430M", "F444W",
    "F460M", "F480M", "F560W", "F770W", "F1000W", "F1280W", "F1500W",
    "F1800W", "F2100W", "F2550W",
)
MIRI_BANDS = frozenset(JWST_BANDS[18:])
DEFAULT_CUTS = tuple(range(21, 30))
DEFAULT_SNR_EDGES = (0, 2, 3, 5, 10, 20, 50, 100)


@dataclass
class Catalog:
    field: str
    path: Path
    ids: np.ndarray
    mag: np.ndarray
    snr: dict[str, np.ndarray]
    assumed_njy: list[str]


def _flux(hdu, name: str, assumed_njy: list[str]) -> np.ndarray:
    """Convert a scalar flux column to nJy, retaining invalid/missing values."""
    values = np.asarray(hdu.data[name], dtype=np.float64).copy()
    column = hdu.columns[name]
    if column.null is not None:
        values[values == column.null] = np.nan
    unit = column.unit
    if unit:
        values *= u.Unit(unit).to(u.nJy)
    else:
        assumed_njy.append(f"{hdu.name}.{name}")
    return values


def _aligned(reference, other, label: str) -> None:
    if (len(reference) != len(other) or "ID" not in other.names
            or not np.array_equal(reference["ID"], other["ID"])):
        raise ValueError(f"{label}: catalog IDs are not aligned")


def read_catalog(field: str, path: Path, mag_band: str, bands: list[str],
                 aperture: str) -> Catalog:
    assumed_njy: list[str] = []
    with fits.open(path, memmap=True) as hdus:
        ref = hdus["CIRC"].data
        ids = np.asarray(ref["ID"], dtype=np.int64).copy()
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate source IDs")
        mag_hdu = hdus["MIRI" if mag_band in MIRI_BANDS else "KRON"]
        _aligned(ref, mag_hdu.data, f"{path}:{mag_hdu.name}")
        flux = _flux(mag_hdu, f"{mag_band}_KRON", assumed_njy)
        mag = np.full(len(ids), np.nan)
        good = np.isfinite(flux) & (flux > 0)
        mag[good] = 31.4 - 2.5 * np.log10(flux[good])
        snrs = {}
        checked = set()
        for band in bands:
            name = "MIRI" if band in MIRI_BANDS else ("KRON" if aperture == "KRON" else "CIRC")
            snrs[band] = np.full(len(ids), np.nan)
            if name not in hdus:
                continue
            hdu = hdus[name]
            if name not in checked:
                _aligned(ref, hdu.data, f"{path}:{name}")
                checked.add(name)
            fcol, ecol = f"{band}_{aperture}", f"{band}_{aperture}_e"
            if fcol not in hdu.columns.names or ecol not in hdu.columns.names:
                continue
            flux = _flux(hdu, fcol, assumed_njy)
            error = _flux(hdu, ecol, assumed_njy)
            valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
            with np.errstate(over="ignore", invalid="ignore"):
                np.divide(flux, error, out=snrs[band], where=valid)
            snrs[band][~np.isfinite(snrs[band])] = np.nan
    return Catalog(field, path, ids, mag, snrs, assumed_njy)


def resolve_catalogs(raw_root: Path, manifest: Path | None) -> list[tuple[str, Path]]:
    entries = (json.loads(manifest.read_text())["catalogs"] if manifest else
               [{"field": f, "path": f"catalogs/{f}.fits"} for f in ("GOODS-S", "GOODS-N")])
    result = [(e["field"], raw_root / e["path"]) for e in entries]
    if not result or len({f for f, _ in result}) != len(result):
        raise ValueError("Expected one catalog per unique field")
    if any(f == "ALL" for f, _ in result):
        raise ValueError("ALL is reserved for the combined summary")
    if len({p.resolve() for _, p in result}) != len(result):
        raise ValueError("The same catalog must not be counted twice")
    return result


def discover_bands(entries: list[tuple[str, Path]], aperture: str) -> list[str]:
    available = set()
    for _, path in entries:
        with fits.open(path, memmap=True) as hdus:
            for hdu_name in ("CIRC" if aperture != "KRON" else "KRON", "MIRI"):
                if hdu_name not in hdus:
                    continue
                names = set(hdus[hdu_name].columns.names)
                for band in JWST_BANDS:
                    if f"{band}_{aperture}" in names and f"{band}_{aperture}_e" in names:
                        available.add(band)
    return [b for b in JWST_BANDS if b in available]


def extracted_ids(scratch: Path, catalogs: list[Catalog]) -> tuple[dict[str, np.ndarray], dict]:
    """Read only IDs from files published AND committed by the extractor.

    In-progress .part files and a Parquet renamed before its checkpoint are
    excluded. Membership is recomputed on each run; no dataset files are written.
    """
    import pyarrow.parquet as pq

    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    known = {c.field: set(c.ids.tolist()) for c in catalogs}
    retained: dict[str, set[int]] = {f: set() for f in known}
    files, markers = 0, 0
    for checkpoint in sorted(scratch.rglob("*.progress.json")):
        field = checkpoint.name.removesuffix(".progress.json")
        if field not in known:
            raise ValueError(f"Unknown field in {checkpoint}")
        state = json.loads(checkpoint.read_text())
        if not isinstance(state.get("next_shard"), int) or state["next_shard"] < 0:
            raise ValueError(f"Invalid next_shard in {checkpoint}")
        rows = 0
        for index in range(state["next_shard"]):
            path = checkpoint.parent / f"{field}_{index:05d}.parquet"
            table = pq.read_table(path, columns=["field", "object_id"], use_threads=False)
            for actual_field, object_id in zip(table["field"].to_pylist(), table["object_id"].to_pylist()):
                prefix, number = object_id.rsplit(":", 1)
                source_id = int(number)
                if actual_field != field or prefix != field or source_id not in known[field]:
                    raise ValueError(f"{path}: unknown or inconsistent source {object_id}")
                if source_id in retained[field]:
                    raise ValueError(f"{path}: duplicate extracted source {object_id}")
                retained[field].add(source_id)
                rows += 1
            files += 1
        if rows != state["count"]:
            raise ValueError(f"{checkpoint}: row count differs from checkpoint")
        markers += int(checkpoint.with_name(f"{field}.done").exists())
    masks = {c.field: np.isin(c.ids, list(retained[c.field])) for c in catalogs}
    return masks, {"committed_parquet_files": files, "completed_field_shards": markers,
                   "snapshot_only": True,
                   "warning": "Partial extraction is not a representative sample of the final dataset."}


def selections(mag: np.ndarray, cuts: list[float]):
    yield "all", None, None, np.ones(len(mag), dtype=bool)
    for upper in cuts:
        yield "cumulative", None, upper, np.isfinite(mag) & (mag < upper)
    for lower, upper in zip([None, *cuts], [*cuts, None]):
        mask = np.isfinite(mag)
        if lower is not None:
            mask &= mag >= lower
        if upper is not None:
            mask &= mag < upper
        yield "interval", lower, upper, mask


def _percent(numerator: int, denominator: int):
    return 100.0 * numerator / denominator if denominator else None


def _max_snr(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.full(matrix.shape[0], np.nan)
    result = np.max(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
    result[~np.isfinite(result)] = np.nan
    return result


def summarize(catalogs: list[Catalog], bands: list[str], cuts: list[float],
              snr_edges: list[float], retained: dict[str, np.ndarray] | None = None) -> dict:
    fields = np.concatenate([np.full(len(c.ids), c.field) for c in catalogs])
    mag = np.concatenate([c.mag for c in catalogs])
    matrix = np.concatenate([np.column_stack([c.snr[b] for b in bands]) for c in catalogs])
    groups = {"JWST": list(range(len(bands))),
              "NIRCam": [i for i, b in enumerate(bands) if b not in MIRI_BANDS],
              "MIRI": [i for i, b in enumerate(bands) if b in MIRI_BANDS]}
    metrics = {b: matrix[:, i] for i, b in enumerate(bands)}
    metrics.update({f"max_{group}": _max_snr(matrix[:, indices]) for group, indices in groups.items()})
    populations = {"catalog": np.ones(len(mag), dtype=bool)}
    if retained is not None:
        populations["extracted"] = np.concatenate([retained[c.field] for c in catalogs])
    histogram_edges = [-np.inf, *snr_edges, np.inf]
    count_rows, bin_rows, detection_rows = [], [], []
    for population, population_mask in populations.items():
        for field in ["ALL", *[c.field for c in catalogs]]:
            base = population_mask & (True if field == "ALL" else fields == field)
            parent_count = int(base.sum())
            for selection, lower, upper, selected in selections(mag, cuts):
                mask = base & selected
                n = int(mask.sum())
                common = dict(population=population, field=field, selection=selection,
                              mag_lower=lower, mag_upper=upper, n_selected=n)
                count_rows.append(dict(common, n_population=parent_count,
                                       percent_population=_percent(n, parent_count),
                                       n_valid_mag=int(np.isfinite(mag[mask]).sum()),
                                       n_missing_mag=int((~np.isfinite(mag[mask])).sum())))
                for metric, values in metrics.items():
                    values = values[mask]
                    valid = np.isfinite(values)
                    nvalid = int(valid.sum())
                    counts, _ = np.histogram(values[valid], bins=histogram_edges)
                    for index, count in enumerate(counts):
                        lo, hi = histogram_edges[index:index + 2]
                        label = (f"<{hi:g}" if not np.isfinite(lo) else
                                 f">={lo:g}" if not np.isfinite(hi) else f"[{lo:g},{hi:g})")
                        bin_rows.append(dict(common, metric=metric, snr_bin=label,
                                             n_bin=int(count), n_valid_snr=nvalid,
                                             percent_selected=_percent(int(count), n),
                                             percent_valid_snr=_percent(int(count), nvalid)))
                    bin_rows.append(dict(common, metric=metric, snr_bin="missing",
                                         n_bin=n - nvalid, n_valid_snr=nvalid,
                                         percent_selected=_percent(n - nvalid, n), percent_valid_snr=None))
                for group, indices in groups.items():
                    submatrix = matrix[mask][:, indices]
                    for threshold in (2, 3, 5, 10):
                        ndetected = np.sum(submatrix >= threshold, axis=1)
                        for minimum_bands in (1, 2, 4):
                            passed = int((ndetected >= minimum_bands).sum())
                            detection_rows.append(dict(common, instrument=group, snr_threshold=threshold,
                                                       min_bands=minimum_bands, n_pass=passed,
                                                       percent_selected=_percent(passed, n)))
    return {"counts": count_rows, "snr_bins": bin_rows, "detections": detection_rows}


def write_reports(output: Path, tables: dict, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        temporary = output / f"{name}.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / f"{name}.csv")
    summary = dict(metadata, counts=tables["counts"])
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    lines = ["JADES selection audit", f"Magnitude: {metadata['magnitude_definition']}",
             f"SNR: {metadata['snr_definition']}", "",
             "population field m_upper N(m<m_upper) %population"]
    for row in tables["counts"]:
        if row["selection"] == "cumulative":
            percentage = row["percent_population"]
            lines.append(f"{row['population']:10s} {row['field']:7s} {row['mag_upper']:7g} "
                         f"{row['n_selected']:12d} {percentage if percentage is not None else 'NA'}")
    lines.extend(["", "SNR bins: [lower, upper); last bin has no upper bound.",
                  "Missing SNR is separate; negative flux/SNR is retained.",
                  "Catalog counts include all sources; no star/quality or image-coverage cut is imposed.",
                  "Extracted counts, when requested, are a snapshot of checkpointed source IDs.",
                  "Catalog aperture SNR is not pixel SNR; SNR remains per-band."])
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def plot_reports(output: Path, tables: dict, mag_band: str, population: str = "catalog") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = [r for r in tables["counts"] if r["population"] == population and r["field"] == "ALL"]
    cumulative = [r for r in counts if r["selection"] == "cumulative"]
    intervals = [r for r in counts if r["selection"] == "interval"]
    cuts = [r["mag_upper"] for r in cumulative]
    with plt.rc_context({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "mathtext.fontset": "stix"}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes[0, 0].plot(cuts, [r["n_selected"] for r in cumulative], ".-", color="0.15")
        axes[0, 0].set(xlabel=f"{mag_band} limiting AB magnitude", ylabel="Cumulative source count")
        labels = [f"<{r['mag_upper']:g}" if r["mag_lower"] is None else
                  f">={r['mag_lower']:g}" if r["mag_upper"] is None else
                  f"{r['mag_lower']:g}–{r['mag_upper']:g}" for r in intervals]
        axes[0, 1].bar(labels, [r["n_selected"] for r in intervals], color="0.4")
        axes[0, 1].tick_params(axis="x", rotation=55)
        axes[0, 1].set(xlabel=f"{mag_band} AB magnitude interval", ylabel="Source count")
        bins = [r for r in tables["snr_bins"] if r["population"] == population and r["field"] == "ALL"
                and r["selection"] == "cumulative" and r["metric"] == mag_band]
        labels = list(dict.fromkeys(r["snr_bin"] for r in bins))
        bottom = np.zeros(len(cuts))
        for index, label in enumerate(labels):
            values = np.array([r["percent_selected"] if r["percent_selected"] is not None else np.nan
                               for r in bins if r["snr_bin"] == label])
            axes[1, 0].bar(cuts, values, bottom=bottom, width=0.8, label=label,
                           color="0.85" if label == "missing" else plt.get_cmap("viridis")(index / max(1, len(labels) - 2)))
            bottom += values
        axes[1, 0].set(xlabel=f"{mag_band} limiting AB magnitude", ylabel=f"{mag_band} SNR fraction (%)", ylim=(0, 100))
        axes[1, 0].legend(fontsize=7, ncol=5, frameon=False,
                          loc="upper center", bbox_to_anchor=(0.5, -0.24))
        for threshold in (2, 3, 5, 10):
            rows = [r for r in tables["detections"] if r["population"] == population and r["field"] == "ALL"
                    and r["selection"] == "cumulative" and r["instrument"] == "JWST"
                    and r["snr_threshold"] == threshold and r["min_bands"] == 4]
            axes[1, 1].plot(cuts, [np.nan if r["percent_selected"] is None else r["percent_selected"] for r in rows],
                            ".-", label=rf"SNR $\geq {threshold}$")
        axes[1, 1].set(xlabel=f"{mag_band} limiting AB magnitude", ylabel="Sources detected in ≥4 bands (%)", ylim=(0, 100))
        axes[1, 1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"JADES DR5 — {population} sources")
        fig.savefig(output / f"selection_{population}.png", dpi=180)
        fig.savefig(output / f"selection_{population}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mag-band", type=str.upper, choices=JWST_BANDS, default="F277W")
    parser.add_argument("--mag-cuts", type=float, nargs="+", default=DEFAULT_CUTS)
    parser.add_argument("--bands", type=str.upper, choices=JWST_BANDS, nargs="+")
    parser.add_argument("--aperture", choices=("CIRC1", "CIRC2", "CIRC3", "CIRC4", "KRON"), default="CIRC3")
    parser.add_argument("--snr-edges", type=float, nargs="+", default=DEFAULT_SNR_EDGES)
    parser.add_argument("--scratch-dir", type=Path, help="Also report the checkpointed extracted subset (IDs only)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Write scientific PNG/PDF figures; needs matplotlib")
    args = parser.parse_args()
    cuts = list(args.mag_cuts)
    edges = list(args.snr_edges)
    for name, values in (("mag-cuts", cuts), ("snr-edges", edges)):
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
            parser.error(f"--{name} must be finite and strictly increasing")
    entries = resolve_catalogs(args.raw_root, args.manifest)
    bands = list(dict.fromkeys(args.bands or discover_bands(entries, args.aperture)))
    if args.mag_band not in bands:
        bands.append(args.mag_band)
    catalogs = []
    for field, path in entries:
        print(f"Reading scalar catalog columns: {field} ({path})", flush=True)
        catalogs.append(read_catalog(field, path, args.mag_band, bands, args.aperture))
    retained, snapshot = extracted_ids(args.scratch_dir, catalogs) if args.scratch_dir else (None, None)
    tables = summarize(catalogs, bands, cuts, edges, retained)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "bands": bands, "mag_cuts": cuts,
        "magnitude_band": args.mag_band, "snr_edges": edges,
        "magnitude_definition": f"{args.mag_band} native KRON: m_AB = 31.4 - 2.5 log10(flux_nJy), flux > 0",
        "snr_definition": f"catalog {args.aperture} flux / error; signed; finite flux and error > 0",
        "bin_convention": "Magnitude cuts: m < cut; intervals and SNR bins: [lower, upper).",
        "percent_denominators": {"percent_population": "all sources in the named population and field",
                                 "percent_selected": "all sources passing the magnitude selection, including missing SNR",
                                 "percent_valid_snr": "selected sources with finite SNR in this metric"},
        "scope": "Catalog sources, without star/quality filtering. Scalar photometry does not imply image coverage.",
        "inputs": [{"field": c.field, "path": str(c.path.resolve()), "objects": len(c.ids),
                    "size_bytes": c.path.stat().st_size, "mtime_ns": c.path.stat().st_mtime_ns,
                    "columns_assumed_njy_without_tunit": c.assumed_njy} for c in catalogs],
        "extraction_snapshot": snapshot,
    }
    write_reports(args.output_dir, tables, metadata)
    if args.plots:
        for population in (["catalog", "extracted"] if retained is not None else ["catalog"]):
            plot_reports(args.output_dir, tables, args.mag_band, population)
    print(f"Reports written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
